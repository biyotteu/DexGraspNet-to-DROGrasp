"""
DRO-Grasp Inference on DexGraspNet Objects
==========================================

DRO-Grasp pretrained 모델을 사용하여 DexGraspNet 오브젝트에 대한
ShadowHand grasp pose를 예측하는 스크립트.

Pipeline:
    1. DexGraspNet 오브젝트 메쉬 로드 (OBJ → point cloud)
    2. DRO-Grasp pretrained 모델 로드 (model_3robots.pth)
    3. ShadowHand 모델 로드
    4. D(R,O) 예측 → Multilateration → SE(3) Registration → IK Optimization
    5. 결과: predict_q (30 DOF) = [tx, ty, tz, rx, ry, rz, j0..j23]

Usage:
    # 단일 오브젝트
    python inference_dexgraspnet.py \
        --drograsp_root /path/to/DRO-Grasp \
        --object_mesh /path/to/meshdata/sem-Bottle-.../coacd/decomposed.obj \
        --object_scale 0.06 \
        --num_grasps 10

    # 변환된 STL 사용 (convert_dexgraspnet_to_droGrasp.py 출력물)
    python inference_dexgraspnet.py \
        --drograsp_root /path/to/DRO-Grasp \
        --object_stl /path/to/output/data/data_urdf/object/dexgraspnet/obj_name/obj_name.stl \
        --num_grasps 10

    # DexGraspNet 전체 데이터셋 배치 처리
    python inference_dexgraspnet.py \
        --drograsp_root /path/to/DRO-Grasp \
        --meshdata_dir /path/to/meshdata \
        --batch_objects 50 \
        --num_grasps 5 \
        --output_dir ./inference_results

    # 변환 결과 디렉토리에서 배치 처리
    python inference_dexgraspnet.py \
        --drograsp_root /path/to/DRO-Grasp \
        --converted_dir /path/to/converted_output/data \
        --num_grasps 10 \
        --output_dir ./inference_results

Requirements:
    - DRO-Grasp 레포 클론 및 의존성 설치
    - pretrained checkpoint: bash scripts/download_ckpt.sh
    - DRO-Grasp data: bash scripts/download_data.sh
    - GPU (CUDA) 권장
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
from types import SimpleNamespace
from tqdm import tqdm
import time
import glob


# =============================================================================
# DRO-Grasp 임포트 (sys.path에 DRO-Grasp 루트 추가 필요)
# =============================================================================

def setup_drograsp_imports(drograsp_root: str):
    """DRO-Grasp 모듈을 임포트할 수 있도록 sys.path 설정"""
    drograsp_root = os.path.abspath(drograsp_root)
    if drograsp_root not in sys.path:
        sys.path.insert(0, drograsp_root)

    # 필수 파일 확인
    required = [
        'model/network.py',
        'utils/multilateration.py',
        'utils/se3_transform.py',
        'utils/optimization.py',
        'utils/hand_model.py',
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(drograsp_root, f))]
    if missing:
        raise FileNotFoundError(
            f"DRO-Grasp 필수 파일 누락:\n" +
            "\n".join(f"  - {f}" for f in missing) +
            f"\n\nDRO-Grasp 루트: {drograsp_root}"
        )

    return drograsp_root


# =============================================================================
# Object Point Cloud 생성
# =============================================================================

def load_object_pointcloud_from_obj(
    obj_path: str,
    scale: float = 1.0,
    num_points: int = 1024,
) -> np.ndarray:
    """
    DexGraspNet OBJ 메쉬 → 포인트 클라우드 변환.

    Args:
        obj_path: OBJ 메쉬 파일 경로 (e.g., meshdata/.../coacd/decomposed.obj)
        scale: DexGraspNet에서 사용된 스케일 (e.g., 0.06)
        num_points: 샘플링할 포인트 수

    Returns:
        points: (num_points, 3) numpy array
    """
    mesh = trimesh.load(obj_path, force='mesh')

    # DexGraspNet은 scale을 곱해서 사용
    if scale != 1.0:
        mesh.apply_scale(scale)

    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points.astype(np.float32)


def load_object_pointcloud_from_stl(
    stl_path: str,
    num_points: int = 1024,
) -> np.ndarray:
    """
    변환된 STL 메쉬 (이미 스케일 적용됨) → 포인트 클라우드.

    Args:
        stl_path: STL 메쉬 파일 경로
        num_points: 샘플링할 포인트 수

    Returns:
        points: (num_points, 3) numpy array
    """
    mesh = trimesh.load(stl_path, force='mesh')
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points.astype(np.float32)


def load_object_pointcloud_from_pt(
    pt_path: str,
    num_points: int = 1024,
) -> np.ndarray:
    """
    DRO-Grasp .pt 포인트클라우드 파일에서 로드.

    Args:
        pt_path: .pt 파일 경로 (data/PointCloud/object/.../*.pt)
        num_points: 사용할 포인트 수

    Returns:
        points: (num_points, 3) numpy array
    """
    pc_data = torch.load(pt_path, map_location='cpu')
    if isinstance(pc_data, torch.Tensor):
        points = pc_data[:, :3].numpy()
    else:
        points = np.array(pc_data)[:, :3]

    # 랜덤 서브샘플링
    if len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        points = points[indices]
    elif len(points) < num_points:
        # 부족하면 반복 샘플링
        indices = np.random.choice(len(points), num_points, replace=True)
        points = points[indices]

    return points.astype(np.float32)


# =============================================================================
# Inference Pipeline
# =============================================================================

class DROGraspInference:
    """
    DRO-Grasp pretrained 모델을 사용한 grasp pose 추론 클래스.

    사용 흐름:
        1. inference = DROGraspInference(drograsp_root, device='cuda:0')
        2. results = inference.predict_grasps(object_pc, num_grasps=10)
        3. results['predict_q']  → (num_grasps, 30) tensor
    """

    def __init__(
        self,
        drograsp_root: str,
        checkpoint: str = 'model_3robots',
        robot_name: str = 'shadowhand',
        device: str = 'cuda:0',
        num_points: int = 1024,
    ):
        """
        Args:
            drograsp_root: DRO-Grasp 레포 루트 경로
            checkpoint: 체크포인트 이름 (ckpt/model/ 하위)
            robot_name: 로봇 이름 ('shadowhand', 'allegro', 'barrett')
            device: 디바이스 ('cuda:0' 또는 'cpu')
            num_points: 포인트 클라우드 포인트 수
        """
        self.drograsp_root = os.path.abspath(drograsp_root)
        self.robot_name = robot_name
        self.device = torch.device(device)
        self.num_points = num_points

        # DRO-Grasp 임포트 설정
        setup_drograsp_imports(drograsp_root)

        # 원래 작업 디렉토리 저장 후 DRO-Grasp 루트로 이동
        # (상대 경로 참조하는 코드가 있을 수 있으므로)
        self._orig_cwd = os.getcwd()
        os.chdir(self.drograsp_root)

        print(f"[Init] DRO-Grasp root: {self.drograsp_root}")
        print(f"[Init] Robot: {robot_name}")
        print(f"[Init] Device: {device}")
        print(f"[Init] Checkpoint: {checkpoint}")

        # 모델 로드
        self._load_network(checkpoint)
        self._load_hand_model()

        print(f"[Init] 초기화 완료!")

    def _load_network(self, checkpoint: str):
        """네트워크 로드 및 체크포인트 적용"""
        from model.network import create_network

        ckpt_path = os.path.join(self.drograsp_root, f'ckpt/model/{checkpoint}.pth')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"체크포인트를 찾을 수 없습니다: {ckpt_path}\n"
                f"다운로드: cd {self.drograsp_root} && bash scripts/download_ckpt.sh"
            )

        cfg = SimpleNamespace(
            emb_dim=512,
            latent_dim=64,
            pretrain=None,
            center_pc=True,
            block_computing=True,
        )

        self.network = create_network(cfg, mode='validate').to(self.device)
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.network.load_state_dict(state_dict)
        self.network.eval()
        print(f"[Init] 네트워크 로드 완료: {ckpt_path}")

    def _load_hand_model(self):
        """ShadowHand 모델 로드"""
        from utils.hand_model import create_hand_model

        self.hand = create_hand_model(self.robot_name, self.device)

        # DOF 확인
        joint_names = self.hand.pk_chain.get_joint_parameter_names()
        self.dof = len(joint_names)
        print(f"[Init] Hand model 로드 완료: {self.robot_name} ({self.dof} DOF)")
        print(f"[Init] Joint names: {joint_names[:6]}... (총 {len(joint_names)}개)")

    def get_initial_q(self, batch_size: int) -> torch.Tensor:
        """
        초기 joint configuration 생성.

        DRO-Grasp의 hand_model.get_initial_q()를 사용하여
        joint limits 범위 내 랜덤 초기값 생성.

        Returns:
            initial_q: (batch_size, DOF) tensor
        """
        initial_q_list = []
        for _ in range(batch_size):
            q = self.hand.get_initial_q()  # (DOF,) tensor
            initial_q_list.append(q)

        return torch.stack(initial_q_list, dim=0).to(self.device)

    def predict_grasps(
        self,
        object_pc: np.ndarray,
        num_grasps: int = 10,
        n_iter: int = 64,
        split_batch: int = 25,
    ) -> Dict:
        """
        오브젝트 포인트 클라우드에 대한 grasp pose 예측.

        DRO-Grasp의 generative 특성상, 같은 오브젝트에 대해
        다양한 grasp pose를 생성할 수 있습니다 (CVAE latent sampling).

        Args:
            object_pc: (N, 3) numpy array - 오브젝트 포인트 클라우드
            num_grasps: 생성할 grasp 수 (= batch size)
            n_iter: IK 최적화 반복 수 (기본 64)
            split_batch: GPU 메모리 관리를 위한 배치 분할 크기

        Returns:
            dict with:
                'predict_q': (num_grasps, DOF) tensor - 예측된 joint angles
                    [0:3] = translation (tx, ty, tz)
                    [3:6] = rotation (rx, ry, rz) in XYZ intrinsic Euler
                    [6:]  = hand joint angles (24 joints for ShadowHand)
                'dro': (num_grasps, N, N) tensor - D(R,O) distance matrix
                'mlat_pc': (num_grasps, N, 3) tensor - multilateration 결과
                'time_network': float - 네트워크 추론 시간
                'time_optimization': float - IK 최적화 시간
        """
        from utils.multilateration import multilateration
        from utils.se3_transform import compute_link_pose
        from utils.optimization import process_transform, create_problem, optimization

        # ========================================
        # 1. 데이터 준비
        # ========================================
        # Object PC를 배치로 복제
        obj_pc_tensor = torch.from_numpy(object_pc).float().to(self.device)

        # 포인트 수 조정
        if obj_pc_tensor.shape[0] > self.num_points:
            indices = torch.randperm(obj_pc_tensor.shape[0])[:self.num_points]
            obj_pc_tensor = obj_pc_tensor[indices]
        elif obj_pc_tensor.shape[0] < self.num_points:
            indices = torch.randint(0, obj_pc_tensor.shape[0], (self.num_points,))
            obj_pc_tensor = obj_pc_tensor[indices]

        # (N, 3) → (B, N, 3)
        object_pc_batch = obj_pc_tensor.unsqueeze(0).expand(num_grasps, -1, -1)

        # Robot point cloud (hand의 canonical 포즈에서 샘플링)
        robot_pc_batch = self._get_robot_pc(num_grasps)

        # Initial joint configuration
        initial_q = self.get_initial_q(num_grasps)

        print(f"[Inference] Object PC: {object_pc_batch.shape}")
        print(f"[Inference] Robot PC: {robot_pc_batch.shape}")
        print(f"[Inference] Initial Q: {initial_q.shape}")

        # ========================================
        # 2. Network Forward: D(R,O) 예측
        # ========================================
        t0 = time.time()

        with torch.no_grad():
            output = self.network(robot_pc_batch, object_pc_batch)
            dro = output['dro'].detach()  # (B, N, N)

        time_network = time.time() - t0
        print(f"[Inference] D(R,O) 예측 완료: {dro.shape} ({time_network:.3f}s)")

        # ========================================
        # 3. Multilateration → SE(3) → IK
        # ========================================
        t1 = time.time()

        # Multilateration: D(R,O) + object_pc → predicted robot contact points
        mlat_pc = multilateration(dro, object_pc_batch)  # (B, N, 3)

        # SE(3) Registration: link point clouds ↔ predicted contact points
        transform, _ = compute_link_pose(
            self.hand.links_pc,
            mlat_pc,
            is_train=False
        )

        # Extract translation targets
        optim_transform = process_transform(self.hand.pk_chain, transform)

        # IK Optimization (배치 분할 처리)
        layer = create_problem(self.hand.pk_chain, optim_transform.keys())

        predict_q_list = []
        for i in range(0, num_grasps, split_batch):
            end_idx = min(i + split_batch, num_grasps)
            split_initial_q = initial_q[i:end_idx]
            split_optim = {k: v[i:end_idx] for k, v in optim_transform.items()}

            split_q = optimization(
                self.hand.pk_chain,
                layer,
                split_initial_q,
                split_optim,
                n_iter=n_iter,
            )
            predict_q_list.append(split_q)

        predict_q = torch.cat(predict_q_list, dim=0)  # (B, DOF)

        time_optimization = time.time() - t1
        print(f"[Inference] IK 최적화 완료: {predict_q.shape} ({time_optimization:.3f}s)")

        return {
            'predict_q': predict_q,
            'dro': dro,
            'mlat_pc': mlat_pc,
            'time_network': time_network,
            'time_optimization': time_optimization,
        }

    def _get_robot_pc(self, batch_size: int) -> torch.Tensor:
        """
        Robot (ShadowHand) point cloud를 canonical 포즈에서 가져옴.

        DRO-Grasp는 hand.links_pc에서 모든 링크의 포인트를 합쳐서 사용.
        각 링크에서 균등하게 샘플링하여 총 num_points개의 포인트 생성.

        Returns:
            robot_pc: (batch_size, num_points, 3) tensor
        """
        # links_pc에서 모든 포인트 합치기
        all_points = []
        for link_name, link_pc in self.hand.links_pc.items():
            if isinstance(link_pc, torch.Tensor):
                all_points.append(link_pc.to(self.device))
            else:
                all_points.append(torch.tensor(link_pc, dtype=torch.float32, device=self.device))

        all_points = torch.cat(all_points, dim=0)  # (total_points, 3)

        # num_points개로 서브샘플링
        if all_points.shape[0] > self.num_points:
            indices = torch.randperm(all_points.shape[0])[:self.num_points]
            robot_pc = all_points[indices]
        else:
            indices = torch.randint(0, all_points.shape[0], (self.num_points,))
            robot_pc = all_points[indices]

        # (N, 3) → (B, N, 3)
        return robot_pc.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def predict_q_to_dict(self, predict_q: torch.Tensor) -> List[Dict]:
        """
        predict_q tensor를 읽기 쉬운 dict 형태로 변환.

        Args:
            predict_q: (B, DOF) tensor

        Returns:
            list of dicts, each with:
                'translation': [tx, ty, tz]
                'rotation_euler_XYZ': [rx, ry, rz]
                'joint_angles': {joint_name: value, ...}
        """
        joint_names = list(self.hand.pk_chain.get_joint_parameter_names())
        results = []

        for i in range(predict_q.shape[0]):
            q = predict_q[i].cpu().numpy()
            result = {
                'translation': q[:3].tolist(),
                'rotation_euler_XYZ': q[3:6].tolist(),
                'joint_angles': {},
            }
            for j, name in enumerate(joint_names):
                if j >= 6:  # skip virtual joints
                    result['joint_angles'][name] = float(q[j])

            results.append(result)

        return results

    def cleanup(self):
        """원래 작업 디렉토리 복원"""
        os.chdir(self._orig_cwd)


# =============================================================================
# 배치 처리 함수
# =============================================================================

def find_dexgraspnet_objects(meshdata_dir: str, max_objects: int = None) -> List[Dict]:
    """
    DexGraspNet meshdata 디렉토리에서 오브젝트 목록 탐색.

    Returns:
        list of dicts: [{'code': 'sem-Bottle-...', 'obj_path': '...', 'scales': [0.06, ...]}]
    """
    objects = []
    meshdata_dir = Path(meshdata_dir)

    for obj_dir in sorted(meshdata_dir.iterdir()):
        if not obj_dir.is_dir():
            continue

        obj_path = obj_dir / 'coacd' / 'decomposed.obj'
        if not obj_path.exists():
            # 대체 경로
            obj_candidates = list(obj_dir.glob('**/*.obj'))
            if obj_candidates:
                obj_path = obj_candidates[0]
            else:
                continue

        objects.append({
            'code': obj_dir.name,
            'obj_path': str(obj_path),
            'scales': [0.06],  # 기본 스케일 (필요시 수정)
        })

        if max_objects and len(objects) >= max_objects:
            break

    return objects


def find_converted_objects(converted_dir: str) -> List[Dict]:
    """
    convert_dexgraspnet_to_droGrasp.py로 변환된 STL 파일 탐색.

    converted_dir: 변환 출력 data/ 디렉토리
    예: /output/data/data_urdf/object/dexgraspnet/

    Returns:
        list of dicts: [{'name': 'dexgraspnet+obj_name', 'stl_path': '...'}]
    """
    objects = []
    stl_base = Path(converted_dir) / 'data_urdf' / 'object' / 'dexgraspnet'

    if not stl_base.exists():
        # 직접 경로 시도
        stl_base = Path(converted_dir)

    for obj_dir in sorted(stl_base.iterdir()):
        if not obj_dir.is_dir():
            continue

        stl_files = list(obj_dir.glob('*.stl'))
        if stl_files:
            objects.append({
                'name': obj_dir.name,
                'stl_path': str(stl_files[0]),
            })

    return objects


def save_results(results: Dict, output_path: str, save_full: bool = False):
    """결과를 .pt (PyTorch) 및 .json 형태로 저장

    Args:
        save_full: True면 dro, mlat_pc도 저장 (매우 큼). 기본 False.
    """
    # PyTorch tensor 저장 (predict_q만 기본 저장)
    save_dict = {'predict_q': results['predict_q'].cpu()}
    if save_full:
        save_dict['dro'] = results['dro'].cpu()
        save_dict['mlat_pc'] = results['mlat_pc'].cpu()
    torch.save(save_dict, output_path + '.pt')

    # JSON은 배치모드에서 생략 (속도 위해)
    print(f"  저장: {output_path}.pt")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DRO-Grasp Inference on DexGraspNet Objects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 단일 OBJ 오브젝트 (DexGraspNet 원본)
  python inference_dexgraspnet.py \\
      --drograsp_root ~/DRO-Grasp \\
      --object_mesh meshdata/sem-Bottle-xxx/coacd/decomposed.obj \\
      --object_scale 0.06 \\
      --num_grasps 10

  # 단일 STL 오브젝트 (변환 결과물)
  python inference_dexgraspnet.py \\
      --drograsp_root ~/DRO-Grasp \\
      --object_stl converted/data/data_urdf/object/dexgraspnet/obj/obj.stl \\
      --num_grasps 10

  # DexGraspNet 배치 처리
  python inference_dexgraspnet.py \\
      --drograsp_root ~/DRO-Grasp \\
      --meshdata_dir ~/DexGraspNet/data/meshdata \\
      --batch_objects 50 \\
      --num_grasps 5 \\
      --output_dir ./results

  # 변환된 데이터 배치 처리
  python inference_dexgraspnet.py \\
      --drograsp_root ~/DRO-Grasp \\
      --converted_dir ~/converted_output/data \\
      --num_grasps 10 \\
      --output_dir ./results
        """
    )

    # 필수 인자
    parser.add_argument('--drograsp_root', type=str, required=True,
                        help='DRO-Grasp 레포 루트 경로')

    # 입력 소스 (택 1)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--object_mesh', type=str,
                             help='단일 OBJ 메쉬 경로 (DexGraspNet 원본)')
    input_group.add_argument('--object_stl', type=str,
                             help='단일 STL 메쉬 경로 (변환 결과물)')
    input_group.add_argument('--object_pt', type=str,
                             help='단일 .pt 포인트클라우드 경로')
    input_group.add_argument('--meshdata_dir', type=str,
                             help='DexGraspNet meshdata 디렉토리 (배치)')
    input_group.add_argument('--converted_dir', type=str,
                             help='변환 결과 data 디렉토리 (배치)')

    # 옵션
    parser.add_argument('--object_scale', type=float, default=0.06,
                        help='OBJ 스케일 (DexGraspNet, 기본 0.06)')
    parser.add_argument('--num_grasps', type=int, default=10,
                        help='오브젝트당 생성할 grasp 수 (기본 10)')
    parser.add_argument('--checkpoint', type=str, default='model_3robots',
                        help='체크포인트 이름 (기본: model_3robots)')
    parser.add_argument('--robot_name', type=str, default='shadowhand',
                        choices=['shadowhand', 'allegro', 'barrett'],
                        help='로봇 이름 (기본: shadowhand)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='디바이스 (기본: cuda:0)')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='결과 저장 디렉토리')
    parser.add_argument('--batch_objects', type=int, default=None,
                        help='처리할 최대 오브젝트 수 (배치 모드)')
    parser.add_argument('--n_iter', type=int, default=64,
                        help='IK 최적화 반복 수 (기본 64)')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='포인트 클라우드 포인트 수 (기본 1024)')
    parser.add_argument('--split_batch', type=int, default=25,
                        help='IK 최적화 배치 분할 크기 (기본 25)')
    parser.add_argument('--save_full', action='store_true',
                        help='D(R,O), mlat_pc도 저장 (용량 큼, 기본 꺼짐)')
    parser.add_argument('--resume', action='store_true',
                        help='이미 결과 파일이 있는 오브젝트 건너뛰기')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='멀티프로세스 워커 수 (기본 1, 실험적)')

    args = parser.parse_args()

    # 모든 경로를 절대경로로 변환 (이후 os.chdir 해도 안전하도록)
    args.drograsp_root = os.path.abspath(args.drograsp_root)
    args.output_dir = os.path.abspath(args.output_dir)
    if args.object_mesh:
        args.object_mesh = os.path.abspath(args.object_mesh)
    if args.object_stl:
        args.object_stl = os.path.abspath(args.object_stl)
    if args.object_pt:
        args.object_pt = os.path.abspath(args.object_pt)
    if args.meshdata_dir:
        args.meshdata_dir = os.path.abspath(args.meshdata_dir)
    if args.converted_dir:
        args.converted_dir = os.path.abspath(args.converted_dir)

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================
    # Inference 엔진 초기화
    # ========================================
    print("=" * 60)
    print("DRO-Grasp Inference on DexGraspNet Objects")
    print("=" * 60)

    inference = DROGraspInference(
        drograsp_root=args.drograsp_root,
        checkpoint=args.checkpoint,
        robot_name=args.robot_name,
        device=args.device,
        num_points=args.num_points,
    )

    try:
        # ========================================
        # 단일 오브젝트 모드
        # ========================================
        if args.object_mesh:
            print(f"\n[Mode] 단일 OBJ 오브젝트: {args.object_mesh}")
            object_pc = load_object_pointcloud_from_obj(
                args.object_mesh, args.object_scale, args.num_points
            )
            results = inference.predict_grasps(
                object_pc, args.num_grasps, args.n_iter, args.split_batch
            )
            obj_name = Path(args.object_mesh).parent.parent.name
            save_results(results, os.path.join(args.output_dir, obj_name))
            _print_summary(results, inference)

        elif args.object_stl:
            print(f"\n[Mode] 단일 STL 오브젝트: {args.object_stl}")
            object_pc = load_object_pointcloud_from_stl(args.object_stl, args.num_points)
            results = inference.predict_grasps(
                object_pc, args.num_grasps, args.n_iter, args.split_batch
            )
            obj_name = Path(args.object_stl).stem
            save_results(results, os.path.join(args.output_dir, obj_name))
            _print_summary(results, inference)

        elif args.object_pt:
            print(f"\n[Mode] 단일 PT 포인트클라우드: {args.object_pt}")
            object_pc = load_object_pointcloud_from_pt(args.object_pt, args.num_points)
            results = inference.predict_grasps(
                object_pc, args.num_grasps, args.n_iter, args.split_batch
            )
            obj_name = Path(args.object_pt).stem
            save_results(results, os.path.join(args.output_dir, obj_name))
            _print_summary(results, inference)

        # ========================================
        # 배치 모드: DexGraspNet meshdata
        # ========================================
        elif args.meshdata_dir:
            print(f"\n[Mode] DexGraspNet 배치 처리: {args.meshdata_dir}")
            objects = find_dexgraspnet_objects(args.meshdata_dir, args.batch_objects)
            print(f"  발견된 오브젝트: {len(objects)}개")

            all_stats = []
            batch_start_time = time.time()
            for idx, obj_info in enumerate(objects):
                for scale in obj_info['scales']:
                    out_name = f"{obj_info['code']}_s{scale:.4f}"
                    if args.resume and os.path.exists(os.path.join(args.output_dir, out_name + '.pt')):
                        continue

                    if idx > 0 and all_stats:
                        elapsed = time.time() - batch_start_time
                        avg_per_obj = elapsed / len(all_stats)
                        remaining = avg_per_obj * (len(objects) - idx)
                        eta_h = int(remaining // 3600)
                        eta_m = int((remaining % 3600) // 60)
                        print(f"\n--- [{idx+1}/{len(objects)}] {obj_info['code']} "
                              f"(avg {avg_per_obj:.1f}s/obj, ETA {eta_h}h {eta_m}m) ---")
                    else:
                        print(f"\n--- [{idx+1}/{len(objects)}] {obj_info['code']} ---")

                    try:
                        object_pc = load_object_pointcloud_from_obj(
                            obj_info['obj_path'], scale, args.num_points
                        )
                        results = inference.predict_grasps(
                            object_pc, args.num_grasps, args.n_iter, args.split_batch
                        )
                        save_results(results, os.path.join(args.output_dir, out_name),
                                     save_full=args.save_full)
                        all_stats.append({
                            'object': obj_info['code'],
                            'scale': scale,
                            'num_grasps': args.num_grasps,
                            'time_network': results['time_network'],
                            'time_optimization': results['time_optimization'],
                        })
                    except Exception as e:
                        print(f"  ERROR: {e}")

            _print_batch_summary(all_stats, args.output_dir)

        # ========================================
        # 배치 모드: 변환된 데이터
        # ========================================
        elif args.converted_dir:
            print(f"\n[Mode] 변환된 데이터 배치 처리: {args.converted_dir}")
            objects = find_converted_objects(args.converted_dir)
            if args.batch_objects:
                objects = objects[:args.batch_objects]
            print(f"  발견된 오브젝트: {len(objects)}개")

            # Resume: 이미 처리된 오브젝트 건너뛰기
            if args.resume:
                original_count = len(objects)
                objects = [
                    o for o in objects
                    if not os.path.exists(os.path.join(args.output_dir, o['name'] + '.pt'))
                ]
                skipped = original_count - len(objects)
                if skipped > 0:
                    print(f"  [Resume] {skipped}개 건너뜀, 남은 오브젝트: {len(objects)}개")

            all_stats = []
            batch_start_time = time.time()
            for idx, obj_info in enumerate(objects):
                # ETA 계산
                if idx > 0:
                    elapsed = time.time() - batch_start_time
                    avg_per_obj = elapsed / idx
                    remaining = avg_per_obj * (len(objects) - idx)
                    eta_h = int(remaining // 3600)
                    eta_m = int((remaining % 3600) // 60)
                    print(f"\n--- [{idx+1}/{len(objects)}] {obj_info['name']} "
                          f"(avg {avg_per_obj:.1f}s/obj, ETA {eta_h}h {eta_m}m) ---")
                else:
                    print(f"\n--- [{idx+1}/{len(objects)}] {obj_info['name']} ---")

                try:
                    object_pc = load_object_pointcloud_from_stl(
                        obj_info['stl_path'], args.num_points
                    )
                    results = inference.predict_grasps(
                        object_pc, args.num_grasps, args.n_iter, args.split_batch
                    )
                    save_results(results, os.path.join(args.output_dir, obj_info['name']),
                                 save_full=args.save_full)
                    all_stats.append({
                        'object': obj_info['name'],
                        'num_grasps': args.num_grasps,
                        'time_network': results['time_network'],
                        'time_optimization': results['time_optimization'],
                    })
                except Exception as e:
                    print(f"  ERROR: {e}")

            _print_batch_summary(all_stats, args.output_dir)

    finally:
        inference.cleanup()

    print("\n" + "=" * 60)
    print("Inference 완료!")
    print("=" * 60)


def _print_summary(results: Dict, inference: DROGraspInference):
    """단일 오브젝트 결과 요약 출력"""
    predict_q = results['predict_q']
    print(f"\n{'='*50}")
    print(f"결과 요약")
    print(f"{'='*50}")
    print(f"  predict_q shape: {predict_q.shape}")
    print(f"  Translation 범위:")
    print(f"    tx: [{predict_q[:, 0].min():.4f}, {predict_q[:, 0].max():.4f}]")
    print(f"    ty: [{predict_q[:, 1].min():.4f}, {predict_q[:, 1].max():.4f}]")
    print(f"    tz: [{predict_q[:, 2].min():.4f}, {predict_q[:, 2].max():.4f}]")
    print(f"  Rotation 범위 (XYZ intrinsic Euler, rad):")
    print(f"    rx: [{predict_q[:, 3].min():.4f}, {predict_q[:, 3].max():.4f}]")
    print(f"    ry: [{predict_q[:, 4].min():.4f}, {predict_q[:, 4].max():.4f}]")
    print(f"    rz: [{predict_q[:, 5].min():.4f}, {predict_q[:, 5].max():.4f}]")
    print(f"  Joint angles 범위:")
    print(f"    min: {predict_q[:, 6:].min():.4f}")
    print(f"    max: {predict_q[:, 6:].max():.4f}")
    print(f"  시간: 네트워크 {results['time_network']:.3f}s, "
          f"최적화 {results['time_optimization']:.3f}s")

    # Grasp diversity (std)
    std = predict_q.std(dim=0).mean().item()
    print(f"  Grasp diversity (mean std): {std:.4f}")

    # 첫 번째 grasp 상세 출력
    print(f"\n  첫 번째 grasp pose:")
    q_dicts = inference.predict_q_to_dict(predict_q[:1])
    for k, v in q_dicts[0].items():
        if k == 'joint_angles':
            print(f"    {k}:")
            for jname, jval in v.items():
                print(f"      {jname}: {jval:.4f}")
        else:
            print(f"    {k}: {v}")


def _print_batch_summary(all_stats: List[Dict], output_dir: str):
    """배치 처리 결과 요약"""
    if not all_stats:
        print("\n처리된 오브젝트가 없습니다.")
        return

    print(f"\n{'='*60}")
    print(f"배치 처리 요약")
    print(f"{'='*60}")
    print(f"  총 오브젝트: {len(all_stats)}개")

    times_net = [s['time_network'] for s in all_stats]
    times_opt = [s['time_optimization'] for s in all_stats]
    print(f"  네트워크 시간: {np.mean(times_net):.3f}s ± {np.std(times_net):.3f}s")
    print(f"  최적화 시간: {np.mean(times_opt):.3f}s ± {np.std(times_opt):.3f}s")
    print(f"  결과 저장: {output_dir}")

    # 통계 저장
    stats_path = os.path.join(output_dir, 'batch_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"  통계 저장: {stats_path}")


if __name__ == '__main__':
    main()
