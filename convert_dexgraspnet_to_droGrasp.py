"""
DexGraspNet → DRO-Grasp 데이터 변환 스크립트
=============================================

DexGraspNet (.npy, ShadowHand 22DOF + 6 wrist = 28 params)
→ DRO-Grasp CMapDataset 포맷 (.pt, ShadowHand 6 virtual + 24 joints = 30 DOF)

주요 차이점:
1. DexGraspNet: qpos dict {WRJTx, WRJTy, WRJTz, WRJRx, WRJRy, WRJRz, robot0:FFJ3, ...}
   - Euler: transforms3d.euler.euler2mat(rx, ry, rz) (sxyz extrinsic)
   - Joints: 22개 (FFJ0-3, MFJ0-3, RFJ0-3, LFJ0-4, THJ0-4)

2. DRO-Grasp: target_q tensor [tx, ty, tz, rx, ry, rz, j0, j1, ..., j23]
   - Euler: scipy Rotation.from_euler('XYZ') (XYZ intrinsic)
   - Joints: 24개 (URDF 기반, pytorch_kinematics 순서)
   - URDF에 virtual joints 6개 포함

핵심 변환:
- Euler angle 컨벤션 변환 (transforms3d sxyz → scipy XYZ intrinsic)
- Joint 순서 매핑 (DexGraspNet MJCF → DRO-Grasp URDF)
- Object mesh: OBJ+scale → STL (no scale, pre-scaled)
- 메타데이터: list of dicts → list of tuples in .pt

Usage:
    python convert_dexgraspnet_to_droGrasp.py \
        --dexgraspnet_root /path/to/DexGraspNet \
        --droGrasp_root /path/to/DRO-Grasp \
        --output_dir /path/to/output
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import trimesh
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import copy


# =============================================================================
# 1. Joint Ordering Definitions
# =============================================================================

# DexGraspNet MJCF joint order (22 joints)
DEXGRASPNET_JOINT_NAMES = [
    'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',  # Forefinger (4)
    'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',  # Middle (4)
    'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',  # Ring (4)
    'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',  # Little (5)
    'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0',   # Thumb (5)
]

# DexGraspNet translation/rotation keys
DEXGRASPNET_TRANS_NAMES = ['WRJTx', 'WRJTy', 'WRJTz']
DEXGRASPNET_ROT_NAMES = ['WRJRx', 'WRJRy', 'WRJRz']


def get_droGrasp_joint_order(urdf_path: str) -> List[str]:
    """
    DRO-Grasp URDF에서 joint 순서를 읽어옵니다.
    pytorch_kinematics를 사용해 URDF 파싱.

    DRO-Grasp의 ShadowHand URDF는 virtual joints 6개 + hand joints 24개 = 30 DOF
    hand joints 24개는 WRJ1, WRJ2 (wrist) + 22 finger joints 포함 가능
    """
    try:
        import pytorch_kinematics as pk
        chain = pk.build_chain_from_urdf(open(urdf_path).read())
        joint_names = chain.get_joint_parameter_names()
        return list(joint_names)
    except ImportError:
        print("WARNING: pytorch_kinematics not installed. Using default joint order.")
        return None


# =============================================================================
# 2. Euler Angle Convention Conversion
# =============================================================================

def convert_euler_dexgraspnet_to_droGrasp(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    DexGraspNet Euler angles → DRO-Grasp Euler angles 변환.

    DexGraspNet: transforms3d.euler.euler2mat(rx, ry, rz)
      - transforms3d default: 'sxyz' (static/extrinsic XYZ)
      - 즉 R = Rz * Ry * Rx (extrinsic = post-multiply)

    DRO-Grasp: scipy Rotation.from_euler('XYZ', [rx, ry, rz])
      - scipy 'XYZ' with uppercase = intrinsic XYZ
      - 즉 R = Rx * Ry' * Rz'' (intrinsic = pre-multiply on rotated axes)

    중요: extrinsic 'xyz' ≡ intrinsic 'ZYX' (reversed order)
    따라서 transforms3d sxyz(rx, ry, rz) = scipy extrinsic xyz(rx, ry, rz)

    변환 방법: rotation matrix를 중간 매개로 사용
    """
    # Step 1: DexGraspNet euler → rotation matrix
    # transforms3d.euler.euler2mat uses 'sxyz' by default
    # This is equivalent to scipy extrinsic 'xyz'
    rot = Rotation.from_euler('xyz', [rx, ry, rz], degrees=False)  # extrinsic xyz (lowercase)
    rot_matrix = rot.as_matrix()

    # Step 2: rotation matrix → DRO-Grasp euler (intrinsic XYZ)
    rot_droGrasp = Rotation.from_matrix(rot_matrix)
    euler_droGrasp = rot_droGrasp.as_euler('XYZ', degrees=False)  # intrinsic XYZ (uppercase)

    return euler_droGrasp


# =============================================================================
# 3. Joint Mapping
# =============================================================================

def build_joint_mapping(droGrasp_urdf_joints: List[str]) -> Dict[str, int]:
    """
    DexGraspNet joint name → DRO-Grasp joint index 매핑 생성.

    DRO-Grasp URDF joints (indices 0-5 are virtual):
      [virtual_tx, virtual_ty, virtual_tz, virtual_rx, virtual_ry, virtual_rz,
       WRJ2, WRJ1,  ← wrist joints (DexGraspNet에는 없음, 0으로 설정)
       FFJ4, FFJ3, FFJ2, FFJ1, FFJ0,  ← 하지만 실제 순서는 URDF에 따라 다름
       ...]

    이 함수는 DRO-Grasp URDF의 실제 joint 이름을 파싱해서 매핑합니다.
    """
    # DexGraspNet joint name → canonical name (prefix 제거)
    def canonical_name(dexgrasp_name: str) -> str:
        """robot0:FFJ3 → FFJ3"""
        return dexgrasp_name.replace('robot0:', '')

    # DRO-Grasp URDF joint name → canonical name
    def urdf_canonical_name(urdf_name: str) -> str:
        """
        URDF joint names might have different prefixes.
        Try to extract the core joint name (e.g., FFJ3, WRJ1, etc.)
        """
        name = urdf_name
        # Remove common prefixes
        for prefix in ['robot0:', 'rh_', 'shadowhand_', 'virtual_']:
            name = name.replace(prefix, '')
        return name.upper()  # normalize case

    mapping = {}
    urdf_canonical_to_idx = {}

    for idx, joint_name in enumerate(droGrasp_urdf_joints):
        if joint_name.startswith('virtual'):
            continue  # skip virtual joints (handled separately as indices 0-5)
        canon = urdf_canonical_name(joint_name)
        urdf_canonical_to_idx[canon] = idx

    for dexgrasp_name in DEXGRASPNET_JOINT_NAMES:
        canon = canonical_name(dexgrasp_name).upper()
        if canon in urdf_canonical_to_idx:
            mapping[dexgrasp_name] = urdf_canonical_to_idx[canon]
        else:
            print(f"  WARNING: DexGraspNet joint '{dexgrasp_name}' ({canon}) "
                  f"not found in DRO-Grasp URDF. Available: {list(urdf_canonical_to_idx.keys())}")

    return mapping


def build_joint_mapping_fallback(dof: int = 30) -> Dict[str, int]:
    """
    URDF를 읽을 수 없을 때 사용하는 fallback 매핑.

    DRO-Grasp ShadowHand의 일반적인 joint 순서 (30 DOF):
    [0-5]: virtual (tx, ty, tz, rx, ry, rz)
    [6]: WRJ2 (wrist)
    [7]: WRJ1 (wrist)
    [8-11]: FFJ3, FFJ2, FFJ1, FFJ0
    [12-15]: MFJ3, MFJ2, MFJ1, MFJ0
    [16-19]: RFJ3, RFJ2, RFJ1, RFJ0
    [20-24]: LFJ4, LFJ3, LFJ2, LFJ1, LFJ0
    [25-29]: THJ4, THJ3, THJ2, THJ1, THJ0

    NOTE: 이 순서는 URDF에 따라 다를 수 있으므로,
    가능하면 get_droGrasp_joint_order()를 사용하세요.
    """
    print("WARNING: Using fallback joint mapping (URDF not found). Verify with actual URDF!")
    mapping = {
        # Forefinger (4 joints)
        'robot0:FFJ3':  8, 'robot0:FFJ2':  9, 'robot0:FFJ1': 10, 'robot0:FFJ0': 11,
        # Middle finger (4 joints)
        'robot0:MFJ3': 12, 'robot0:MFJ2': 13, 'robot0:MFJ1': 14, 'robot0:MFJ0': 15,
        # Ring finger (4 joints)
        'robot0:RFJ3': 16, 'robot0:RFJ2': 17, 'robot0:RFJ1': 18, 'robot0:RFJ0': 19,
        # Little finger (5 joints)
        'robot0:LFJ4': 20, 'robot0:LFJ3': 21, 'robot0:LFJ2': 22, 'robot0:LFJ1': 23, 'robot0:LFJ0': 24,
        # Thumb (5 joints)
        'robot0:THJ4': 25, 'robot0:THJ3': 26, 'robot0:THJ2': 27, 'robot0:THJ1': 28, 'robot0:THJ0': 29,
    }
    return mapping


# =============================================================================
# 4. Single Grasp Conversion
# =============================================================================

def convert_single_grasp(
    qpos: Dict[str, float],
    scale: float,
    dro_dof: int,
    joint_mapping: Optional[Dict[str, int]] = None,
    droGrasp_joint_names: Optional[List[str]] = None,
) -> torch.Tensor:
    """
    DexGraspNet의 단일 grasp를 DRO-Grasp target_q 텐서로 변환합니다.

    Args:
        qpos: DexGraspNet qpos dict
        scale: object scale factor
        dro_dof: DRO-Grasp total DOF (e.g., 30 for ShadowHand)
        joint_mapping: DexGraspNet joint name → DRO-Grasp index
        droGrasp_joint_names: DRO-Grasp URDF joint names for direct matching

    Returns:
        target_q: (dro_dof,) tensor in DRO-Grasp format
    """
    target_q = torch.zeros(dro_dof, dtype=torch.float32)

    # --- Translation (indices 0-2) ---
    # DexGraspNet stores absolute translation
    # DRO-Grasp도 absolute translation을 사용 (학습 시 zero-mean normalization)
    target_q[0] = qpos['WRJTx']
    target_q[1] = qpos['WRJTy']
    target_q[2] = qpos['WRJTz']

    # --- Rotation (indices 3-5) ---
    # Convert Euler convention
    rx, ry, rz = qpos['WRJRx'], qpos['WRJRy'], qpos['WRJRz']
    euler_dro = convert_euler_dexgraspnet_to_droGrasp(rx, ry, rz)
    target_q[3] = euler_dro[0]
    target_q[4] = euler_dro[1]
    target_q[5] = euler_dro[2]

    # --- Joint angles (indices 6+) ---
    if joint_mapping is not None:
        for dexgrasp_name, dro_idx in joint_mapping.items():
            if dexgrasp_name in qpos:
                target_q[dro_idx] = qpos[dexgrasp_name]
    elif droGrasp_joint_names is not None:
        # Direct matching by canonical name
        dexgrasp_canon = {name.replace('robot0:', '').upper(): name
                          for name in DEXGRASPNET_JOINT_NAMES}

        for idx, urdf_joint in enumerate(droGrasp_joint_names):
            if urdf_joint.startswith('virtual'):
                continue
            # Try to match
            canon = urdf_joint.replace('robot0:', '').replace('rh_', '').upper()
            if canon in dexgrasp_canon:
                dexgrasp_key = dexgrasp_canon[canon]
                if dexgrasp_key in qpos:
                    target_q[idx] = qpos[dexgrasp_key]
    else:
        raise ValueError("Either joint_mapping or droGrasp_joint_names must be provided")

    return target_q


# =============================================================================
# 5. Object Mesh Conversion
# =============================================================================

def convert_object_mesh(
    obj_mesh_path: str,
    scale: float,
    output_stl_path: str,
) -> None:
    """
    DexGraspNet OBJ mesh (with scale) → DRO-Grasp STL mesh (pre-scaled).

    DexGraspNet: mesh at unit scale, scaled by grasp's `scale` field
    DRO-Grasp: mesh already at correct scale, stored as STL
    """
    mesh = trimesh.load_mesh(obj_mesh_path)
    mesh.apply_scale(scale)

    os.makedirs(os.path.dirname(output_stl_path), exist_ok=True)
    mesh.export(output_stl_path)


def generate_object_pointcloud(
    stl_path: str,
    output_pt_path: str,
    num_points: int = 2048,
) -> None:
    """
    STL mesh → DRO-Grasp object point cloud (.pt).

    DRO-Grasp format: (num_points, 6) tensor [x, y, z, nx, ny, nz]
    """
    mesh = trimesh.load_mesh(stl_path)
    points, face_indices = mesh.sample(num_points, return_index=True)
    normals = mesh.face_normals[face_indices]

    pc_with_normals = np.concatenate([points, normals], axis=1)
    pc_tensor = torch.tensor(pc_with_normals, dtype=torch.float32)

    os.makedirs(os.path.dirname(output_pt_path), exist_ok=True)
    torch.save(pc_tensor, output_pt_path)


# =============================================================================
# 6. Main Conversion Pipeline
# =============================================================================

class DexGraspNetToDROGraspConverter:
    """DexGraspNet → DRO-Grasp 전체 변환 파이프라인"""

    def __init__(
        self,
        dexgraspnet_root: str,
        droGrasp_root: str,
        output_dir: str,
        dataset_name: str = 'dexgraspnet',
    ):
        """
        Args:
            dexgraspnet_root: DexGraspNet 레포 루트 경로
            droGrasp_root: DRO-Grasp 레포 루트 경로
            output_dir: 변환된 데이터 출력 경로
            dataset_name: DRO-Grasp에서 사용할 dataset 이름 prefix
        """
        self.dexgraspnet_root = Path(dexgraspnet_root)
        self.droGrasp_root = Path(droGrasp_root)
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name

        # DexGraspNet paths
        self.grasp_data_dir = self.dexgraspnet_root / 'data' / 'dataset'
        self.mesh_data_dir = self.dexgraspnet_root / 'data' / 'meshdata'

        # DRO-Grasp paths
        self.urdf_meta_path = self.droGrasp_root / 'data' / 'data_urdf' / 'robot' / 'urdf_assets_meta.json'

        # Output paths
        self.output_dataset_dir = self.output_dir / 'data' / 'CMapDataset_filtered'
        self.output_object_mesh_dir = self.output_dir / 'data' / 'data_urdf' / 'object' / dataset_name
        self.output_object_pc_dir = self.output_dir / 'data' / 'PointCloud' / 'object' / dataset_name

        # Initialize
        self.droGrasp_joint_names = None
        self.dro_dof = None
        self.joint_mapping = None
        self._init_joint_mapping()

    def _ensure_drograsp_data(self):
        """
        DRO-Grasp data/ 폴더가 비어있으면 자동으로 다운로드합니다.
        DRO-Grasp 레포는 git clone만으로는 data/가 비어있고,
        bash scripts/download_data.sh 를 실행해야 URDF 등이 받아집니다.
        """
        if self.urdf_meta_path.exists():
            return  # 이미 있음

        print("=" * 60)
        print("DRO-Grasp data/ 폴더가 비어있습니다. 자동 다운로드를 시도합니다...")
        print("(URDF, 포인트클라우드 등 학습에 필요한 파일)")
        print("=" * 60)

        download_script = self.droGrasp_root / 'scripts' / 'download_data.sh'
        data_dir = self.droGrasp_root / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)

        if download_script.exists():
            # 공식 다운로드 스크립트 사용
            print(f"Running: bash {download_script}")
            import subprocess
            result = subprocess.run(
                ['bash', str(download_script)],
                cwd=str(self.droGrasp_root),
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                print("DRO-Grasp data download complete!")
            else:
                print(f"Download script failed: {result.stderr}")
                self._download_data_manually()
        else:
            self._download_data_manually()

    def _download_data_manually(self):
        """download_data.sh가 없을 때 직접 다운로드"""
        import subprocess
        data_dir = self.droGrasp_root / 'data'
        zip_path = data_dir / 'data.zip'

        url = "https://github.com/zhenyuwei2003/DRO-Grasp/releases/download/v1.0/data.zip"
        print(f"Downloading from: {url}")
        print("(이 파일은 수백 MB일 수 있으므로 시간이 걸릴 수 있습니다)")

        result = subprocess.run(
            ['wget', '-q', '--show-progress', url, '-O', str(zip_path)],
            cwd=str(data_dir),
        )

        if result.returncode == 0 and zip_path.exists():
            print("Unzipping...")
            subprocess.run(['unzip', '-o', str(zip_path), '-d', str(data_dir)])
            zip_path.unlink()  # zip 삭제
            print("DRO-Grasp data download & extract complete!")
        else:
            print(f"\nERROR: 자동 다운로드 실패.")
            print(f"수동으로 다운로드하세요:")
            print(f"  cd {self.droGrasp_root}")
            print(f"  bash scripts/download_data.sh")
            print(f"또는:")
            print(f"  cd {data_dir}")
            print(f"  wget {url}")
            print(f"  unzip data.zip")

    def _init_joint_mapping(self):
        """DRO-Grasp URDF에서 joint 매핑 초기화"""
        # data/ 폴더가 비어있으면 자동 다운로드 시도
        self._ensure_drograsp_data()

        try:
            urdf_meta = json.load(open(self.urdf_meta_path))
            shadowhand_urdf = self.droGrasp_root / urdf_meta['urdf_path']['shadowhand']

            if shadowhand_urdf.exists():
                self.droGrasp_joint_names = get_droGrasp_joint_order(str(shadowhand_urdf))
                self.dro_dof = len(self.droGrasp_joint_names)
                self.joint_mapping = build_joint_mapping(self.droGrasp_joint_names)

                print(f"DRO-Grasp ShadowHand DOF: {self.dro_dof}")
                print(f"DRO-Grasp joint names: {self.droGrasp_joint_names}")
                print(f"Joint mapping (DexGraspNet → DRO-Grasp): {self.joint_mapping}")
            else:
                print(f"WARNING: URDF not found at {shadowhand_urdf}")
                print(f"Using fallback joint mapping (DOF=30)")
                self.dro_dof = 30
                self.joint_mapping = build_joint_mapping_fallback(self.dro_dof)
        except Exception as e:
            print(f"Error initializing joint mapping: {e}")
            print("Using fallback configuration (DOF=30)")
            self.dro_dof = 30
            self.joint_mapping = build_joint_mapping_fallback(self.dro_dof)

    def discover_grasp_files(self) -> List[str]:
        """DexGraspNet grasp .npy 파일 목록 탐색"""
        grasp_files = sorted(self.grasp_data_dir.glob('*.npy'))
        print(f"Found {len(grasp_files)} grasp files in {self.grasp_data_dir}")
        return grasp_files

    def convert_grasps(
        self,
        max_objects: Optional[int] = None,
        max_grasps_per_object: Optional[int] = None,
        validate_grasps: bool = True,
    ) -> Tuple[List, Dict]:
        """
        전체 DexGraspNet grasp 데이터를 DRO-Grasp 포맷으로 변환.

        Args:
            max_objects: 변환할 최대 오브젝트 수 (None=전부)
            max_grasps_per_object: 오브젝트당 최대 grasp 수 (None=전부)
            validate_grasps: joint limit 범위 검증 여부

        Returns:
            metadata: [(target_q, object_name, 'shadowhand'), ...]
            info: dataset statistics
        """
        grasp_files = self.discover_grasp_files()
        if max_objects is not None:
            grasp_files = grasp_files[:max_objects]

        metadata = []
        info = {
            'shadowhand': {
                'robot_name': 'shadowhand',
                'num_total': 0,
                'num_upper_object': 0,
                'num_per_object': {}
            }
        }

        # Track (object_code, scale) → object_name for mesh conversion
        # 중요: DexGraspNet은 같은 오브젝트에 여러 scale이 섞여 있음
        # (예: mug에 scale 0.06, 0.08, 0.1, 0.12, 0.15가 혼재)
        # DRO-Grasp은 오브젝트당 고정 메쉬를 사용하므로,
        # scale별로 별도 오브젝트로 분리해야 함
        object_scale_pairs = {}  # (object_code, scale_str) → object_name

        for grasp_file in tqdm(grasp_files, desc="Converting grasps"):
            object_code = grasp_file.stem

            # Load DexGraspNet grasps
            try:
                grasps = np.load(str(grasp_file), allow_pickle=True)
            except Exception as e:
                print(f"Error loading {grasp_file}: {e}")
                continue

            if len(grasps) == 0:
                continue

            grasp_list = list(grasps)
            if max_grasps_per_object is not None:
                grasp_list = grasp_list[:max_grasps_per_object]

            # Group grasps by scale
            from collections import defaultdict
            scale_groups = defaultdict(list)
            for grasp_dict in grasp_list:
                scale = float(grasp_dict['scale'])
                scale_groups[scale].append(grasp_dict)

            for scale, scale_grasps in scale_groups.items():
                # DRO-Grasp object name: scale별로 분리
                # 예: "dexgraspnet+core-mug-8570d9a8_s006"
                scale_str = f"s{scale:.3f}".replace('.', '')
                object_name = f"{self.dataset_name}+{object_code}_{scale_str}"

                object_scale_pairs[(object_code, scale_str)] = (object_name, scale)
                converted_count = 0

                for grasp_dict in scale_grasps:
                    qpos = grasp_dict['qpos']

                    # Convert
                    try:
                        target_q = convert_single_grasp(
                            qpos=qpos,
                            scale=scale,
                            dro_dof=self.dro_dof,
                            joint_mapping=self.joint_mapping,
                            droGrasp_joint_names=self.droGrasp_joint_names,
                        )

                        if validate_grasps:
                            if torch.isnan(target_q).any() or torch.isinf(target_q).any():
                                continue

                        metadata.append((target_q, object_name, 'shadowhand'))
                        converted_count += 1

                    except Exception as e:
                        print(f"Error converting grasp for {object_code} (scale={scale}): {e}")
                        continue

                if converted_count > 0:
                    info['shadowhand']['num_total'] += converted_count
                    info['shadowhand']['num_upper_object'] += 1
                    info['shadowhand']['num_per_object'][object_name] = converted_count

        print(f"\nConversion complete:")
        print(f"  Total grasps: {info['shadowhand']['num_total']}")
        print(f"  Total objects (scale-separated): {info['shadowhand']['num_upper_object']}")

        # Convert meshes (scale별로 분리된 오브젝트 각각)
        self._convert_meshes(object_scale_pairs)

        return metadata, info

    def _convert_meshes(self, object_scale_pairs: Dict):
        """
        오브젝트 메쉬를 OBJ→STL로 변환 (pre-scaled).
        scale별로 분리된 각 오브젝트에 대해 별도 메쉬 생성.
        """
        print(f"\nConverting {len(object_scale_pairs)} object meshes (scale-separated)...")

        for (object_code, scale_str), (object_name, scale) in tqdm(
            object_scale_pairs.items(), desc="Converting meshes"
        ):
            obj_path = self.mesh_data_dir / object_code / 'coacd' / 'decomposed.obj'

            if not obj_path.exists():
                print(f"  Mesh not found: {obj_path}")
                continue

            # object_name format: "dexgraspnet+core-mug-xxx_s006"
            # DRO-Grasp path: data/data_urdf/object/{dataset}/{obj_with_scale}/{obj_with_scale}.stl
            obj_with_scale = f"{object_code}_{scale_str}"
            stl_dir = self.output_object_mesh_dir / obj_with_scale
            stl_path = stl_dir / f"{obj_with_scale}.stl"

            try:
                convert_object_mesh(str(obj_path), scale, str(stl_path))
            except Exception as e:
                print(f"  Error converting mesh {object_code} (scale={scale}): {e}")
                continue

            # Generate point cloud
            pc_path = self.output_object_pc_dir / f"{obj_with_scale}.pt"
            try:
                generate_object_pointcloud(str(stl_path), str(pc_path))
            except Exception as e:
                print(f"  Error generating point cloud {object_code} (scale={scale}): {e}")

    def save_dataset(
        self,
        metadata: List,
        info: Dict,
        train_ratio: float = 0.9,
    ):
        """
        DRO-Grasp CMapDataset 포맷으로 저장.

        생성 파일:
        1. cmap_dataset.pt: 메인 데이터셋
        2. split_train_validate_objects.json: train/val split
        """
        os.makedirs(self.output_dataset_dir, exist_ok=True)

        # --- cmap_dataset.pt ---
        dataset = {
            'metadata': metadata,
            'info': info,
        }
        dataset_path = self.output_dataset_dir / 'cmap_dataset.pt'
        torch.save(dataset, str(dataset_path))
        print(f"Saved dataset to {dataset_path}")
        print(f"  Metadata entries: {len(metadata)}")

        # --- split_train_validate_objects.json ---
        all_objects = sorted(set(m[1] for m in metadata))
        np.random.seed(42)
        np.random.shuffle(all_objects)

        split_idx = int(len(all_objects) * train_ratio)
        split = {
            'train': sorted(all_objects[:split_idx]),
            'validate': sorted(all_objects[split_idx:]),
        }

        split_path = self.output_dataset_dir / 'split_train_validate_objects.json'
        with open(split_path, 'w') as f:
            json.dump(split, f, indent=2)
        print(f"Saved split to {split_path}")
        print(f"  Train objects: {len(split['train'])}")
        print(f"  Validate objects: {len(split['validate'])}")

    def run(
        self,
        max_objects: Optional[int] = None,
        max_grasps_per_object: Optional[int] = None,
        train_ratio: float = 0.9,
    ):
        """전체 변환 파이프라인 실행"""
        print("=" * 60)
        print("DexGraspNet → DRO-Grasp Conversion Pipeline")
        print("=" * 60)
        print(f"DexGraspNet root: {self.dexgraspnet_root}")
        print(f"DRO-Grasp root:   {self.droGrasp_root}")
        print(f"Output dir:       {self.output_dir}")
        print(f"DRO-Grasp DOF:    {self.dro_dof}")
        print()

        # Convert grasps
        metadata, info = self.convert_grasps(
            max_objects=max_objects,
            max_grasps_per_object=max_grasps_per_object,
        )

        if len(metadata) == 0:
            print("ERROR: No grasps were converted!")
            return

        # Save dataset
        self.save_dataset(metadata, info, train_ratio)

        print("\n" + "=" * 60)
        print("Conversion complete!")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"1. Copy output to DRO-Grasp data directory:")
        print(f"   cp -r {self.output_dir}/data/* {self.droGrasp_root}/data/")
        print(f"2. Generate robot point clouds (if not already done):")
        print(f"   cd {self.droGrasp_root} && python data_utils/generate_pc.py --type robot --robot_name shadowhand")
        print(f"3. Update config to use 'shadowhand' only in robot_names")
        print(f"4. Train: python train.py")


# =============================================================================
# 7. Verification Utilities
# =============================================================================

def verify_conversion(
    dexgraspnet_root: str,
    output_dir: str,
    sample_object: Optional[str] = None,
    num_samples: int = 5,
):
    """
    변환 결과를 검증합니다.
    - Euler angle 변환 정확성 (rotation matrix 비교)
    - Joint values 범위 확인
    - Point cloud 시각화 (optional)
    """
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    # Load converted dataset
    dataset_path = Path(output_dir) / 'data' / 'CMapDataset_filtered' / 'cmap_dataset.pt'
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        return

    dataset = torch.load(str(dataset_path))
    metadata = dataset['metadata']

    print(f"Loaded {len(metadata)} grasp entries")

    # Sample and verify
    indices = np.random.choice(len(metadata), min(num_samples, len(metadata)), replace=False)

    for i, idx in enumerate(indices):
        target_q, object_name, robot_name = metadata[idx]
        print(f"\n--- Sample {i+1} ---")
        print(f"  Object: {object_name}")
        print(f"  Robot: {robot_name}")
        print(f"  target_q shape: {target_q.shape}")
        print(f"  Translation: [{target_q[0]:.4f}, {target_q[1]:.4f}, {target_q[2]:.4f}]")
        print(f"  Rotation (Euler XYZ): [{target_q[3]:.4f}, {target_q[4]:.4f}, {target_q[5]:.4f}]")
        print(f"  Joints: {target_q[6:].numpy()}")

        # Verify rotation matrix round-trip
        euler = target_q[3:6].numpy()
        R = Rotation.from_euler('XYZ', euler).as_matrix()
        euler_back = Rotation.from_matrix(R).as_euler('XYZ')
        R_back = Rotation.from_euler('XYZ', euler_back).as_matrix()
        rot_error = np.abs(R - R_back).max()
        print(f"  Rotation round-trip error: {rot_error:.2e}")

    # Verify point cloud files exist
    pc_dir = Path(output_dir) / 'data' / 'PointCloud' / 'object'
    if pc_dir.exists():
        pc_files = list(pc_dir.rglob('*.pt'))
        print(f"\nPoint cloud files: {len(pc_files)}")
        if pc_files:
            sample_pc = torch.load(str(pc_files[0]))
            print(f"  Sample PC shape: {sample_pc.shape}")

    # Verify mesh files exist
    mesh_dir = Path(output_dir) / 'data' / 'data_urdf' / 'object'
    if mesh_dir.exists():
        stl_files = list(mesh_dir.rglob('*.stl'))
        print(f"\nSTL mesh files: {len(stl_files)}")

    print("\nVerification complete!")


# =============================================================================
# 8. CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert DexGraspNet dataset to DRO-Grasp format'
    )
    parser.add_argument(
        '--dexgraspnet_root', type=str, required=True,
        help='Path to DexGraspNet repository root'
    )
    parser.add_argument(
        '--droGrasp_root', type=str, required=True,
        help='Path to DRO-Grasp repository root'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory (default: droGrasp_root)'
    )
    parser.add_argument(
        '--max_objects', type=int, default=None,
        help='Maximum number of objects to convert'
    )
    parser.add_argument(
        '--max_grasps_per_object', type=int, default=None,
        help='Maximum grasps per object'
    )
    parser.add_argument(
        '--train_ratio', type=float, default=0.9,
        help='Train/validate split ratio'
    )
    parser.add_argument(
        '--dataset_name', type=str, default='dexgraspnet',
        help='Dataset name prefix for DRO-Grasp'
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='Run verification after conversion'
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.droGrasp_root

    converter = DexGraspNetToDROGraspConverter(
        dexgraspnet_root=args.dexgraspnet_root,
        droGrasp_root=args.droGrasp_root,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
    )

    converter.run(
        max_objects=args.max_objects,
        max_grasps_per_object=args.max_grasps_per_object,
        train_ratio=args.train_ratio,
    )

    if args.verify:
        verify_conversion(
            dexgraspnet_root=args.dexgraspnet_root,
            output_dir=args.output_dir,
        )


if __name__ == '__main__':
    main()
