# DexGraspNet → DRO-Grasp Converter

[DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet) 데이터셋을 [D(R,O) Grasp](https://github.com/zhenyuwei2003/DRO-Grasp) 학습 포맷으로 변환하는 도구
<br/>두 프로젝트는 동일한 ShadowHand를 사용하지만 데이터 포맷이 다르기 때문에 변환이 필요

---

## 준비물

| 항목 | 출처 | 설명 |
|------|------|------|
| `dataset/` | [PKU 미러](https://mirrors.pku.edu.cn/dl-release/DexGraspNet-ICRA2023/) | grasp NPY 파일 (5355개) |
| `meshdata/` | [PKU 미러](https://mirrors.pku.edu.cn/dl-release/DexGraspNet-ICRA2023/) | 오브젝트 OBJ 메쉬 (5355개) |
| DRO-Grasp 레포 + data | `git clone` 후 `bash scripts/download_data.sh` | ShadowHand URDF (joint 매핑에 필요) |

### 의존성

```bash
pip install numpy scipy trimesh torch tqdm
# DRO-Grasp의 URDF 파싱을 위해:
pip install pytorch_kinematics
```

---

## 데이터 셋업 (Step-by-Step)

전체 과정은 크게 5단계입니다. 모든 경로는 프로젝트 루트(`~/project/`)를 기준으로 설명

### Step 1. 레포 클론 & 이 변환 도구 배치

```bash
mkdir -p ~/project && cd ~/project

# 이 변환 도구
git clone <this-repo> Dexgraspnet_to_DROGrasp

# DexGraspNet 레포 (샘플 데이터 5개 포함, 시각화/검증용)
git clone https://github.com/PKU-EPIC/DexGraspNet.git

# DRO-Grasp 레포
git clone https://github.com/zhenyuwei2003/DRO-Grasp.git
```

이 시점의 디렉토리 구조:

```
~/project/
├── Dexgraspnet_to_DROGrasp/   ← 이 변환 도구
├── DexGraspNet/                ← git clone 직후, data/ 안에 샘플 5개만 있음
└── DRO-Grasp/                  ← git clone 직후, data/ 폴더가 비어있음 ⚠
```

### Step 2. DRO-Grasp data 다운로드 (URDF 등)

DRO-Grasp은 `git clone`만으로는 `data/` 폴더가 비어있습니다. ShadowHand URDF, 로봇 포인트클라우드 등 학습에 필요한 파일을 GitHub Releases에서 별도로 다운운

```bash
cd ~/project/DRO-Grasp
bash scripts/download_data.sh
```

이 스크립트는 내부적으로 아래를 실행합니다:

```bash
wget https://github.com/zhenyuwei2003/DRO-Grasp/releases/download/v1.0/data.zip
unzip data.zip -d data/
```

완료 후 확인:

```bash
# 이 파일이 있으면 성공
ls data/data_urdf/robot/urdf_assets_meta.json
ls data/data_urdf/robot/shadowhand/shadowhand.urdf
ls data/PointCloud/robot/shadowhand.pt
```

다운로드된 `data/`에는 다음이 포함:

```
DRO-Grasp/data/
├── data_urdf/
│   ├── robot/
│   │   ├── urdf_assets_meta.json          ← URDF 경로 참조 파일
│   │   ├── shadowhand/shadowhand.urdf     ← joint 매핑에 필수
│   │   ├── allegro/allegro.urdf
│   │   └── barrett/barrett.urdf
│   └── object/
│       ├── contactdb/                      ← 기존 CMapDataset 오브젝트
│       └── ycb/
├── PointCloud/
│   └── robot/
│       ├── shadowhand.pt                   ← 학습 시 로봇 포인트클라우드
│       ├── allegro.pt
│       └── barrett.pt
├── CMapDataset_filtered/                    ← 기존 학습 데이터 (나중에 덮어쓸 예정)
│   ├── cmap_dataset.pt
│   └── split_train_validate_objects.json
└── MultiDex_filtered/                       ← pretraining 데이터
```

### Step 3. DexGraspNet 풀 데이터셋 다운로드

PKU 미러에서 `meshdata`(오브젝트 메쉬)와 `dataset`(grasp 데이터)을 다운로드

```bash
# 브라우저로 접속하여 다운로드:
# https://mirrors.pku.edu.cn/dl-release/DexGraspNet-ICRA2023/
#
# 또는 직접 다운로드 링크를 알고 있다면:
cd ~/downloads
wget <meshdata 다운로드 URL>
wget <dataset 다운로드 URL>
```

다운로드 후 압축 해제

```bash
# 파일명은 실제 다운로드에 맞게 조정하세요
tar -xzf meshdata.tar.gz    # 또는 unzip meshdata.zip
tar -xzf dataset.tar.gz     # 또는 unzip dataset.zip
```

DexGraspNet 레포의 `data/`에 심볼릭 링크로 연결

```bash
cd ~/project/DexGraspNet/data

# 기존 샘플 데이터 백업 (선택사항)
mv dataset dataset_sample_backup
mv meshdata meshdata_sample_backup

# 풀 데이터셋 심볼릭 링크
ln -sfn ~/downloads/dataset  ./dataset
ln -sfn ~/downloads/meshdata ./meshdata
```

완료 후 확인

```bash
# 오브젝트 수 확인 (5355개여야 함)
ls dataset/*.npy | wc -l

# 메쉬 폴더 수 확인 (5355개여야 함)
ls -d meshdata/*/coacd | wc -l

# 오브젝트 이름이 1:1 대응하는지 확인
ls dataset/ | sed 's/.npy//' | sort > /tmp/dataset_objects.txt
ls meshdata/ | sort > /tmp/mesh_objects.txt
diff /tmp/dataset_objects.txt /tmp/mesh_objects.txt
# diff 출력이 없으면 완벽히 매칭
```

이 시점의 전체 구조

```
~/project/
├── DexGraspNet/
│   └── data/
│       ├── dataset → ~/downloads/dataset     (5355개 .npy 파일)
│       └── meshdata → ~/downloads/meshdata   (5355개 오브젝트 폴더)
├── DRO-Grasp/
│   └── data/
│       ├── data_urdf/robot/shadowhand/       (URDF)
│       ├── PointCloud/robot/shadowhand.pt    (로봇 PC)
│       └── CMapDataset_filtered/             (기존 데이터, 나중에 교체)
└── Dexgraspnet_to_DROGrasp/
    └── convert_dexgraspnet_to_droGrasp.py
```

### Step 4. 변환 실행

```bash
cd ~/project/Dexgraspnet_to_DROGrasp

python convert_dexgraspnet_to_droGrasp.py \
    --dexgraspnet_root ../DexGraspNet \
    --droGrasp_root ../DRO-Grasp \
    --output_dir ~/dexgraspnet_converted \
    --verify
```

정상 실행 시 출력 예시:

```
============================================================
DexGraspNet → DRO-Grasp Conversion Pipeline
============================================================
DRO-Grasp ShadowHand DOF: 30           ← URDF에서 읽어옴
Joint mapping (DexGraspNet → DRO-Grasp): {'robot0:FFJ3': 8, ...}
Found 5355 grasp files in .../data/dataset
Converting grasps: 100%|████████████| 5355/5355
  Total grasps: 1320000
  Total objects (scale-separated): ~26775
Converting meshes: 100%|████████████| 26775/26775
Saved dataset to .../cmap_dataset.pt
```

테스트 실행 (소규모):

```bash
# 10개 오브젝트만 먼저 테스트
python convert_dexgraspnet_to_droGrasp.py \
    --dexgraspnet_root ../DexGraspNet \
    --droGrasp_root ../DRO-Grasp \
    --output_dir ~/dexgraspnet_converted_test \
    --max_objects 10 \
    --verify
```

### Step 5. DRO-Grasp에 변환 결과 연결

변환된 데이터를 DRO-Grasp의 `data/` 폴더에 심볼릭 링크로 연결

```bash
cd ~/project/DRO-Grasp

# 기존 CMapDataset 백업
mv data/CMapDataset_filtered data/CMapDataset_filtered_original

# 변환된 데이터 연결
ln -sfn ~/dexgraspnet_converted/data/CMapDataset_filtered  data/CMapDataset_filtered
ln -sfn ~/dexgraspnet_converted/data/data_urdf/object/dexgraspnet data/data_urdf/object/dexgraspnet
ln -sfn ~/dexgraspnet_converted/data/PointCloud/object/dexgraspnet data/PointCloud/object/dexgraspnet
```

완료 후 확인:

```bash
ls -la data/CMapDataset_filtered/cmap_dataset.pt
ls -la data/CMapDataset_filtered/split_train_validate_objects.json
ls data/data_urdf/object/dexgraspnet/ | head -5
ls data/PointCloud/object/dexgraspnet/ | head -5
```

최종 DRO-Grasp 데이터 구조:

```
DRO-Grasp/data/
├── data_urdf/
│   ├── robot/
│   │   └── shadowhand/shadowhand.urdf       (Step 2에서 다운로드)
│   └── object/
│       ├── contactdb/                        (기존)
│       ├── ycb/                              (기존)
│       └── dexgraspnet → (심볼릭 링크)       (Step 5에서 연결) ★
├── PointCloud/
│   ├── robot/shadowhand.pt                   (Step 2에서 다운로드)
│   └── object/
│       ├── contactdb/                        (기존)
│       ├── ycb/                              (기존)
│       └── dexgraspnet → (심볼릭 링크)       (Step 5에서 연결) ★
├── CMapDataset_filtered → (심볼릭 링크)      (Step 5에서 연결) ★
│   ├── cmap_dataset.pt
│   └── split_train_validate_objects.json
├── CMapDataset_filtered_original/             (기존 백업)
└── MultiDex_filtered/                         (pretraining용, 그대로 유지)
```

### Step 6. DRO-Grasp config 수정 & 학습

```bash
cd ~/project/DRO-Grasp
```

`configs/dataset/cmap_dataset.yaml`을 수정합니다:

```yaml
robot_names:
  - 'shadowhand'    # DexGraspNet은 ShadowHand만 포함

batch_size: 8
num_workers: 16
object_pc_type: 'random'
```

학습 실행:

```bash
python train.py
```

---

## 빠른 시작 (요약)

위 Step 1~6을 한번에 정리하면:

```bash
# 레포 클론
cd ~/project
git clone <this-repo> Dexgraspnet_to_DROGrasp
git clone https://github.com/PKU-EPIC/DexGraspNet.git
git clone https://github.com/zhenyuwei2003/DRO-Grasp.git

# DRO-Grasp data 다운로드 (URDF 등)
cd DRO-Grasp && bash scripts/download_data.sh && cd ..

# DexGraspNet 풀 데이터 심볼릭 링크
cd DexGraspNet/data
ln -sfn /path/to/downloaded/dataset  ./dataset
ln -sfn /path/to/downloaded/meshdata ./meshdata
cd ../..

# 변환 실행
cd Dexgraspnet_to_DROGrasp
python convert_dexgraspnet_to_droGrasp.py \
    --dexgraspnet_root ../DexGraspNet \
    --droGrasp_root ../DRO-Grasp \
    --output_dir ~/dexgraspnet_converted \
    --verify

# DRO-Grasp에 연결
cd ../DRO-Grasp
mv data/CMapDataset_filtered data/CMapDataset_filtered_original
ln -sfn ~/dexgraspnet_converted/data/CMapDataset_filtered  data/CMapDataset_filtered
ln -sfn ~/dexgraspnet_converted/data/data_urdf/object/dexgraspnet data/data_urdf/object/dexgraspnet
ln -sfn ~/dexgraspnet_converted/data/PointCloud/object/dexgraspnet data/PointCloud/object/dexgraspnet

# 학습
python train.py
```

---

## 입력 데이터 구조 (DexGraspNet)

### `dataset/{object_code}.npy`

각 파일은 `np.load(path, allow_pickle=True)`로 로드하며, Python dict의 list를 담고 있음

```python
grasps = np.load('core-mug-8570d9a8.npy', allow_pickle=True)
# grasps[0] =
{
    'qpos': {
        'WRJTx':  -0.0208,     # wrist x position (m)
        'WRJTy':  -0.1161,     # wrist y position (m)
        'WRJTz':   0.0471,     # wrist z position (m)
        'WRJRx':   1.4920,     # Euler angle X (rad), extrinsic xyz convention
        'WRJRy':   0.8010,     # Euler angle Y (rad)
        'WRJRz':   3.1083,     # Euler angle Z (rad)
        'robot0:FFJ3': -0.036, # Forefinger abduction
        'robot0:FFJ2':  0.009, # Forefinger proximal
        'robot0:FFJ1':  0.616, # Forefinger middle
        'robot0:FFJ0':  1.109, # Forefinger distal
        'robot0:MFJ3': ...,    # Middle finger (4 joints)
        'robot0:MFJ2': ...,
        'robot0:MFJ1': ...,
        'robot0:MFJ0': ...,
        'robot0:RFJ3': ...,    # Ring finger (4 joints)
        'robot0:RFJ2': ...,
        'robot0:RFJ1': ...,
        'robot0:RFJ0': ...,
        'robot0:LFJ4': ...,    # Little finger (5 joints)
        'robot0:LFJ3': ...,
        'robot0:LFJ2': ...,
        'robot0:LFJ1': ...,
        'robot0:LFJ0': ...,
        'robot0:THJ4': ...,    # Thumb (5 joints)
        'robot0:THJ3': ...,
        'robot0:THJ2': ...,
        'robot0:THJ1': ...,
        'robot0:THJ0': ...,
    },                         # 총 28 keys (3 trans + 3 rot + 22 joints)
    'scale': 0.06,             # 오브젝트 메쉬에 적용할 스케일 팩터
}
```

주의: **같은 오브젝트 파일 안에 여러 scale 값이 섞여 있음** (예: 0.06, 0.08, 0.10, 0.12, 0.15). 동일 메쉬를 다른 크기로 스케일링하여 다양한 grasp을 생성

### `meshdata/{object_code}/coacd/decomposed.obj`

단위 스케일의 OBJ 메쉬 파일입니다. 실제 크기는 `mesh × scale`로 결정

---

## 출력 데이터 구조 (DRO-Grasp 포맷)

### 파일 구조

```
output_dir/
└── data/
    ├── CMapDataset_filtered/
    │   ├── cmap_dataset.pt                       ← 메인 데이터셋
    │   └── split_train_validate_objects.json      ← train/val 오브젝트 분할
    │
    ├── data_urdf/object/dexgraspnet/
    │   ├── core-mug-8570d9a8_s0060/
    │   │   └── core-mug-8570d9a8_s0060.stl       ← scale=0.06 적용된 메쉬
    │   ├── core-mug-8570d9a8_s0080/
    │   │   └── core-mug-8570d9a8_s0080.stl       ← scale=0.08 적용된 메쉬
    │   └── ...
    │
    └── PointCloud/object/dexgraspnet/
        ├── core-mug-8570d9a8_s0060.pt             ← (2048, 6) 포인트클라우드
        ├── core-mug-8570d9a8_s0080.pt
        └── ...
```

### `cmap_dataset.pt`

```python
torch.load('cmap_dataset.pt')
# =
{
    'metadata': [
        (target_q, object_name, robot_name),  # grasp 1개당 tuple 1개
        (target_q, object_name, robot_name),
        ...
    ],
    'info': {
        'shadowhand': {
            'robot_name': 'shadowhand',
            'num_total': 1320000,
            'num_upper_object': 26775,        # scale 분리 후 오브젝트 수
            'num_per_object': {
                'dexgraspnet+core-mug-8570d9a8_s0060': 52,
                'dexgraspnet+core-mug-8570d9a8_s0080': 92,
                ...
            }
        }
    }
}
```

- `target_q`: `torch.Tensor`, shape `(DOF,)` — 아래 상세 설명 참조
- `object_name`: `str`, 형식 `"dexgraspnet+{object_code}_{scale_str}"`
- `robot_name`: `str`, 항상 `"shadowhand"`

### `split_train_validate_objects.json`

```json
{
    "train": ["dexgraspnet+core-mug-8570d9a8_s0060", ...],
    "validate": ["dexgraspnet+sem-Bottle-437678_s0100", ...]
}
```

90/10 비율로 오브젝트 단위 분할 (동일 오브젝트의 다른 scale은 같은 split에 들어갈 수도, 다른 split에 들어갈 수도 있음).

### 오브젝트 포인트클라우드 `.pt`

```python
pc = torch.load('core-mug-8570d9a8_s0060.pt')
# shape: (2048, 6)
# 열 [0:3] = x, y, z 좌표
# 열 [3:6] = nx, ny, nz 법선 벡터
```

---

## target_q 텐서 상세 (핵심)

DRO-Grasp의 `target_q`는 ShadowHand의 전체 configuration을 나타내는 **1D 텐서**. ShadowHand의 경우 **30차원** (6 virtual + 24 hand joints)

```
target_q = [tx, ty, tz, rx, ry, rz, j0, j1, j2, ..., j23]
            ─────────  ─────────  ──────────────────────────
             [0:3]       [3:6]              [6:30]
           translation  rotation         hand joints
```

### [0:3] Virtual Translation

World frame에서 손목(wrist)의 3D 위치

| index | 의미 | 단위 |
|-------|------|------|
| 0 | tx | meter |
| 1 | ty | meter |
| 2 | tz | meter |

학습 시에는 포인트 클라우드를 zero-mean 정규화하므로 실질적으로 무시되지만, grasp execution 단계에서 사용

### [3:6] Virtual Rotation (Euler XYZ Intrinsic)

손목의 3D 회전. **scipy `Rotation.from_euler('XYZ')` 컨벤션** intrinsic.

| index | 의미 | 단위 | 컨벤션 |
|-------|------|------|--------|
| 3 | rx | radian | X축 먼저 회전 |
| 4 | ry | radian | 회전된 Y'축으로 회전 |
| 5 | rz | radian | 다시 회전된 Z''축으로 회전 |

DexGraspNet은 `transforms3d` extrinsic xyz 컨벤션을 사용하므로 변환이 필요:

```python
# DexGraspNet: R = Rz · Ry · Rx (고정 축 기준)
# DRO-Grasp:  R = Rx · Ry'· Rz'' (회전된 축 기준)
#
# 같은 숫자를 넣으면 다른 rotation matrix가 나옴!
#
# 변환: rotation matrix를 중간 매개로 사용
R = Rotation.from_euler('xyz', [rx, ry, rz]).as_matrix()   # DexGraspNet → matrix
euler_dro = Rotation.from_matrix(R).as_euler('XYZ')         # matrix → DRO-Grasp
#
# 예시:
# DexGraspNet (1.492, 0.801, 3.108) → DRO-Grasp (-1.516, -0.023, 2.340)
# 서로 다른 3개 숫자이지만, 동일한 rotation matrix를 나타냄
```

### [6:29] Hand Joints (24개)

URDF에 정의된 순서대로 나열. DRO-Grasp의 ShadowHand URDF는 24개의 hand joint을 가지며, 이 중 22개는 DexGraspNet과 공유하고 2개(WRJ1, WRJ2)는 DexGraspNet에 없어 0으로 설정

일반적인 순서 (실제 순서는 URDF에 따라 다를 수 있음):

| index | joint | 설명 | DexGraspNet 대응 |
|-------|-------|------|-----------------|
| 6 | WRJ2 | 손목 좌우 | 없음 → 0.0 |
| 7 | WRJ1 | 손목 상하 | 없음 → 0.0 |
| 8 | FFJ3 | 검지 abduction | robot0:FFJ3 |
| 9 | FFJ2 | 검지 proximal | robot0:FFJ2 |
| 10 | FFJ1 | 검지 middle | robot0:FFJ1 |
| 11 | FFJ0 | 검지 distal | robot0:FFJ0 |
| 12 | MFJ3 | 중지 abduction | robot0:MFJ3 |
| 13 | MFJ2 | 중지 proximal | robot0:MFJ2 |
| 14 | MFJ1 | 중지 middle | robot0:MFJ1 |
| 15 | MFJ0 | 중지 distal | robot0:MFJ0 |
| 16 | RFJ3 | 약지 abduction | robot0:RFJ3 |
| 17 | RFJ2 | 약지 proximal | robot0:RFJ2 |
| 18 | RFJ1 | 약지 middle | robot0:RFJ1 |
| 19 | RFJ0 | 약지 distal | robot0:RFJ0 |
| 20 | LFJ4 | 소지 metacarpal | robot0:LFJ4 |
| 21 | LFJ3 | 소지 abduction | robot0:LFJ3 |
| 22 | LFJ2 | 소지 proximal | robot0:LFJ2 |
| 23 | LFJ1 | 소지 middle | robot0:LFJ1 |
| 24 | LFJ0 | 소지 distal | robot0:LFJ0 |
| 25 | THJ4 | 엄지 abduction | robot0:THJ4 |
| 26 | THJ3 | 엄지 proximal | robot0:THJ3 |
| 27 | THJ2 | 엄지 middle | robot0:THJ2 |
| 28 | THJ1 | 엄지 distal | robot0:THJ1 |
| 29 | THJ0 | 엄지 tip | robot0:THJ0 |

스크립트는 DRO-Grasp URDF를 `pytorch_kinematics`로 파싱해서 실제 joint name 기반으로 자동 매핑핑. URDF가 없으면 위 기본 순서를 사용용

---

## DRO-Grasp이 target_q를 사용하는 방식

학습 시 target_q로부터 생성되는 데이터 플로우:

```
target_q (30,)
    │
    ├──→ Forward Kinematics ──→ robot_pc_target (512, 3)
    │                                │
    │                                ├── object_pc (512, 3)
    │                                │
    │                                └──→ D(R,O) = cdist(robot_pc, object_pc)
    │                                              shape: (512, 512)
    │                                              ← 학습 타겟 (distance matrix)
    │
    └──→ get_initial_q() ──→ initial_q (30,)
                                │
                                └──→ robot_pc_initial (512, 3) ← 네트워크 입력
```

네트워크는 `robot_pc_initial + object_pc`를 입력받아 `D(R,O)` distance matrix를 예측하고, 이로부터 multilateration으로 grasp pose를 복원

---

## 변환 시 주의사항

### Scale 분리

DexGraspNet은 하나의 오브젝트에 여러 scale (예: 0.06, 0.08, 0.10, 0.12, 0.15)을 적용해서 grasp을 생성<br/>DRO-Grasp은 오브젝트당 고정 메쉬를 사용하므로, scale별로 별도 오브젝트로 분리

```
core-mug-8570d9a8 (246 grasps, 5 scales)
  ├── dexgraspnet+core-mug-8570d9a8_s0060 (52 grasps, scale=0.06)
  ├── dexgraspnet+core-mug-8570d9a8_s0080 (92 grasps, scale=0.08)
  ├── dexgraspnet+core-mug-8570d9a8_s0100 (55 grasps, scale=0.10)
  ├── dexgraspnet+core-mug-8570d9a8_s0120 (35 grasps, scale=0.12)
  └── dexgraspnet+core-mug-8570d9a8_s0150 (12 grasps, scale=0.15)
```

### Euler Angle Convention

두 프로젝트가 서로 다른 Euler 컨벤션을 사용

- DexGraspNet: `transforms3d` **extrinsic xyz** (= scipy `'xyz'` 소문자)
- DRO-Grasp: scipy **intrinsic XYZ** (= scipy `'XYZ'` 대문자)

테스트 결과, 100/100 랜덤 각도에서 두 컨벤션이 서로 다른 rotation matrix를 생성하므로 변환이 반드시 필요 => `test_euler_conversion.py`로 검증 가능능

### Joint 수 차이 (22 vs 24)

DexGraspNet MJCF는 22개 finger joint만 정의하고, DRO-Grasp URDF는 WRJ1, WRJ2 wrist joint 2개가 추가로 있어 24개입니다. 변환 시 WRJ1, WRJ2는 0으로 설정

---

## CLI 옵션

```bash
python convert_dexgraspnet_to_droGrasp.py \
    --dexgraspnet_root /path/to/DexGraspNet \
    --droGrasp_root /path/to/DRO-Grasp \
    --output_dir /path/to/output \
    --max_objects 100 \              # 오브젝트 수 제한 (테스트용)
    --max_grasps_per_object 50 \     # 오브젝트당 grasp 수 제한
    --train_ratio 0.9 \              # train/val 분할 비율
    --dataset_name dexgraspnet \     # DRO-Grasp 내 dataset prefix
    --verify                         # 변환 후 검증 실행
```

---

## Inference: Pretrained 모델로 Grasp Pose 예측

DRO-Grasp의 pretrained 모델(`model_3robots.pth`)을 사용하여 DexGraspNet 오브젝트에 대한 ShadowHand grasp pose를 직접 예측할 수 있습니다. 새로 학습할 필요 없이 바로 사용 가능

### Inference 준비

```bash
# 1. DRO-Grasp 레포 클론 및 의존성 설치
cd ~/project
git clone https://github.com/zhenyuwei2003/DRO-Grasp.git
cd DRO-Grasp
pip install -r requirements.txt

# 2. DRO-Grasp 데이터 다운로드 (ShadowHand URDF, point cloud 등)
bash scripts/download_data.sh

# 3. Pretrained 체크포인트 다운로드
bash scripts/download_ckpt.sh

# 4. 의존성 확인
pip install trimesh numpy scipy torch tqdm pytorch_kinematics
```

다운로드 후 DRO-Grasp 구조:
```
DRO-Grasp/
├── ckpt/model/
│   ├── model_3robots.pth          ← 3개 로봇 통합 모델 (권장)
│   ├── model_shadowhand.pth       ← ShadowHand 전용
│   ├── model_allegro.pth
│   └── model_barrett.pth
├── data/
│   ├── data_urdf/robot/           ← URDF 파일들
│   ├── PointCloud/robot/          ← 로봇 포인트클라우드
│   └── ...
└── ...
```

### Inference 실행

**단일 오브젝트 (DexGraspNet OBJ 원본):**
```bash
python inference_dexgraspnet.py \
    --drograsp_root ~/project/DRO-Grasp \
    --object_mesh ~/DexGraspNet/data/meshdata/sem-Bottle-437678d4ea/coacd/decomposed.obj \
    --object_scale 0.06 \
    --num_grasps 10 \
    --output_dir ./inference_results
```

**단일 오브젝트 (변환된 STL):**
```bash
python inference_dexgraspnet.py \
    --drograsp_root ~/project/DRO-Grasp \
    --object_stl ~/converted/data/data_urdf/object/dexgraspnet/obj_name/obj_name.stl \
    --num_grasps 10
```

**DexGraspNet 배치 처리:**
```bash
python inference_dexgraspnet.py \
    --drograsp_root ~/project/DRO-Grasp \
    --meshdata_dir ~/DexGraspNet/data/meshdata \
    --batch_objects 50 \
    --num_grasps 5 \
    --output_dir ./inference_results
```

**변환 결과물 배치 처리:**
```bash
python inference_dexgraspnet.py \
    --drograsp_root ~/project/DRO-Grasp \
    --converted_dir ~/dexgraspnet_converted/data \
    --num_grasps 10 \
    --output_dir ./inference_results
```

### Inference 결과물 상세

#### 디렉토리 구조

```
inference_results/
├── core-bottle-1071fa4cddb2da2fc8724d5673a063a6_s0060.pt     ← 오브젝트별 결과
├── core-mug-8570d9a8_s0080.pt
├── sem-Bottle-437678d4ea_s0060.pt
├── ...  (오브젝트 수만큼)
└── batch_stats.json                                           ← 전체 배치 통계
```

파일명 규칙: `{오브젝트코드}_s{스케일}.pt`
예: `core-bottle-xxx_s0060` → 오브젝트 `core-bottle-xxx`, 스케일 `0.060`

#### `.pt` 파일 구조

```python
import torch

data = torch.load('core-bottle-xxx_s0060.pt')

data['predict_q']   # (num_grasps, 30) — 핵심 결과: 예측된 grasp pose들
# --save_full 옵션 사용 시 추가 저장:
data['dro']         # (num_grasps, 1024, 1024) — D(R,O) distance matrix
data['mlat_pc']     # (num_grasps, 1024, 3) — multilateration 복원 포인트
```

#### `predict_q` 텐서 상세 (핵심)

`predict_q`는 `(num_grasps, 30)` shape의 PyTorch float32 tensor입니다.
각 행(row)이 하나의 grasp pose이며, 30개 값은 아래와 같이 구성됩니다.

```
predict_q[i] = [tx, ty, tz, rx, ry, rz, WRJ2, WRJ1, FFJ3, FFJ2, FFJ1, FFJ0, MFJ3, MFJ2, MFJ1, MFJ0, RFJ3, RFJ2, RFJ1, RFJ0, LFJ4, LFJ3, LFJ2, LFJ1, LFJ0, THJ4, THJ3, THJ2, THJ1, THJ0]
```

인덱스별 상세:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Index │ Name   │ 설명                          │ 단위   │ 범위 (일반적) │
├───────┼────────┼───────────────────────────────┼────────┼──────────────┤
│   0   │ tx     │ 손목 X 위치 (오른쪽+)          │ meter  │ -0.3 ~ 0.3  │
│   1   │ ty     │ 손목 Y 위치 (앞쪽+)            │ meter  │ -0.3 ~ 0.3  │
│   2   │ tz     │ 손목 Z 위치 (위쪽+)            │ meter  │ -0.3 ~ 0.3  │
├───────┼────────┼───────────────────────────────┼────────┼──────────────┤
│   3   │ rx     │ 손목 회전 X (Roll)             │ radian │ -π ~ π      │
│   4   │ ry     │ 손목 회전 Y (Pitch)            │ radian │ -π ~ π      │
│   5   │ rz     │ 손목 회전 Z (Yaw)              │ radian │ -π ~ π      │
├───────┼────────┼───────────────────────────────┼────────┼──────────────┤
│   6   │ WRJ2   │ 손목 좌우 기울기 (Wrist Deviate)│ radian │ -0.52 ~ 0.17│
│   7   │ WRJ1   │ 손목 굽힘 (Wrist Flexion)      │ radian │ -0.69 ~ 0.49│
├───────┼────────┼───────────────────────────────┼────────┼──────────────┤
│   8   │ FFJ3   │ 검지 벌림 (Abduction)          │ radian │ -0.35 ~ 0.35│
│   9   │ FFJ2   │ 검지 첫째마디 굽힘 (Proximal)   │ radian │  0.0 ~ 1.57 │
│  10   │ FFJ1   │ 검지 둘째마디 굽힘 (Middle)     │ radian │  0.0 ~ 1.57 │
│  11   │ FFJ0   │ 검지 끝마디 굽힘 (Distal)      │ radian │  0.0 ~ 1.57 │
├───────┼────────┼───────────────────────────────┼────────┼──────────────┤
│  12   │ MFJ3   │ 중지 벌림                      │ radian │ -0.35 ~ 0.35│
│  13   │ MFJ2   │ 중지 첫째마디 굽힘              │ radian │  0.0 ~ 1.57 │
│  14   │ MFJ1   │ 중지 둘째마디 굽힘              │ radian │  0.0 ~ 1.57 │
│  15   │ MFJ0   │ 중지 끝마디 굽힘                │ radian │  0.0 ~ 1.57 │
├───────┼────────┼───────────────────────────────┼────────┼──────────────┤
│  16   │ RFJ3   │ 약지 벌림                      │ radian │ -0.35 ~ 0.35│
│  17   │ RFJ2   │ 약지 첫째마디 굽힘              │ radian │  0.0 ~ 1.57 │
│  18   │ RFJ1   │ 약지 둘째마디 굽힘              │ radian │  0.0 ~ 1.57 │
│  19   │ RFJ0   │ 약지 끝마디 굽힘                │ radian │  0.0 ~ 1.57 │
├───────┼────────┼───────────────────────────────┼────────┼──────────────┤
│  20   │ LFJ4   │ 새끼 회전 (Little Finger Rot)  │ radian │  0.0 ~ 0.79 │
│  21   │ LFJ3   │ 새끼 벌림                      │ radian │ -0.35 ~ 0.35│
│  22   │ LFJ2   │ 새끼 첫째마디 굽힘              │ radian │  0.0 ~ 1.57 │
│  23   │ LFJ1   │ 새끼 둘째마디 굽힘              │ radian │  0.0 ~ 1.57 │
│  24   │ LFJ0   │ 새끼 끝마디 굽힘                │ radian │  0.0 ~ 1.57 │
├───────┼────────┼───────────────────────────────┼────────┼──────────────┤
│  25   │ THJ4   │ 엄지 벌림 (Thumb Abduction)    │ radian │ -1.05 ~ 1.05│
│  26   │ THJ3   │ 엄지 윗회전 (Upper Rotation)   │ radian │  0.0 ~ 1.22 │
│  27   │ THJ2   │ 엄지 중간굽힘 (Middle Flexion) │ radian │ -0.21 ~ 0.52│
│  28   │ THJ1   │ 엄지 첫째마디 굽힘              │ radian │ -0.70 ~ 0.70│
│  29   │ THJ0   │ 엄지 끝마디 굽힘                │ radian │ -1.05 ~ 1.05│
└───────┴────────┴───────────────────────────────┴────────┴──────────────┘
```

#### 좌표계 및 회전 컨벤션

**좌표계**: 오브젝트 중심 기준. 오브젝트의 원점(0,0,0)을 기준으로 손목의 위치(tx,ty,tz)가 결정됩니다.

**회전 컨벤션**: `rx, ry, rz`는 **scipy intrinsic XYZ** Euler angles입니다.

```python
from scipy.spatial.transform import Rotation

# predict_q에서 rotation matrix 복원
rx, ry, rz = predict_q[i, 3], predict_q[i, 4], predict_q[i, 5]
rot_matrix = Rotation.from_euler('XYZ', [rx, ry, rz]).as_matrix()  # (3,3)

# quaternion으로 변환 (시뮬레이터에서 필요할 수 있음)
quat = Rotation.from_euler('XYZ', [rx, ry, rz]).as_quat()  # [qx, qy, qz, qw]
```

주의: DexGraspNet의 **extrinsic xyz** (= `transforms3d.euler.euler2mat` 기본값)와는 다른 컨벤션입니다. 동일한 rotation matrix를 표현하는 Euler angle 값이 달라집니다.

```
DexGraspNet:  Rotation.from_euler('xyz', [rx, ry, rz])  ← 소문자 (extrinsic)
DRO-Grasp:   Rotation.from_euler('XYZ', [rx, ry, rz])  ← 대문자 (intrinsic)
```

#### ShadowHand 손가락 구조 다이어그램

```
                     ShadowHand (24 hand joints)
                     ===========================

          검지(FF)      중지(MF)      약지(RF)      새끼(LF)        엄지(TH)
         ┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐       ┌──────┐
         │ FFJ0 │     │ MFJ0 │     │ RFJ0 │     │ LFJ0 │       │ THJ0 │
         │(끝)  │     │(끝)  │     │(끝)  │     │(끝)  │       │(끝)  │
         ├──────┤     ├──────┤     ├──────┤     ├──────┤       ├──────┤
         │ FFJ1 │     │ MFJ1 │     │ RFJ1 │     │ LFJ1 │       │ THJ1 │
         │(중간)│     │(중간)│     │(중간)│     │(중간)│       │(중간)│
         ├──────┤     ├──────┤     ├──────┤     ├──────┤       ├──────┤
         │ FFJ2 │     │ MFJ2 │     │ RFJ2 │     │ LFJ2 │       │ THJ2 │
         │(근위)│     │(근위)│     │(근위)│     │(근위)│       │(중간)│
         ├──────┤     ├──────┤     ├──────┤     ├──────┤       ├──────┤
         │ FFJ3 │     │ MFJ3 │     │ RFJ3 │     │ LFJ3 │       │ THJ3 │
         │(벌림)│     │(벌림)│     │(벌림)│     │(벌림)│       │(윗회전)│
         └──┬───┘     └──┬───┘     └──┬───┘     ├──────┤       ├──────┤
            │            │            │         │ LFJ4 │       │ THJ4 │
            │            │            │         │(회전)│       │(벌림)│
            │            │            │         └──┬───┘       └──┬───┘
            └────────────┴────────────┴────────────┴──────────────┘
                                      │
                              ┌───────┴───────┐
                              │  WRJ2 (좌우)  │
                              │  WRJ1 (굽힘)  │
                              ├───────────────┤
                              │  손목 (Wrist)  │
                              │  tx, ty, tz   │
                              │  rx, ry, rz   │
                              └───────────────┘
```

Joint 이름 규칙:
- **FF** = Forefinger (검지), **MF** = Middle Finger (중지), **RF** = Ring Finger (약지)
- **LF** = Little Finger (새끼), **TH** = Thumb (엄지)
- **J0** = Distal (끝마디), **J1** = Middle (중간마디), **J2** = Proximal (첫째마디), **J3** = Abduction (벌림)
- **J4** = 새끼/엄지에만 있는 추가 자유도 (회전/벌림)
- **WRJ1** = Wrist Flexion (손목 굽힘), **WRJ2** = Wrist Deviation (손목 좌우)

#### 결과 해석 예시

```python
import torch

data = torch.load('inference_results/core-bottle-xxx_s0060.pt')
predict_q = data['predict_q']  # (100, 30) — 100개 다양한 grasp

# 첫 번째 grasp 해석
q = predict_q[0]
print(f"손목 위치: x={q[0]:.4f}m, y={q[1]:.4f}m, z={q[2]:.4f}m")
print(f"손목 회전: rx={q[3]:.4f}, ry={q[4]:.4f}, rz={q[5]:.4f} (rad)")
print(f"손목 관절: WRJ2={q[6]:.4f}, WRJ1={q[7]:.4f}")
print(f"검지: FFJ3={q[8]:.4f} FFJ2={q[9]:.4f} FFJ1={q[10]:.4f} FFJ0={q[11]:.4f}")
print(f"엄지: THJ4={q[25]:.4f} THJ3={q[26]:.4f} THJ2={q[27]:.4f} THJ1={q[28]:.4f} THJ0={q[29]:.4f}")

# 100개 grasp의 다양성 확인
std = predict_q.std(dim=0)
print(f"\nGrasp diversity (mean joint std): {std[6:].mean():.4f}")
print(f"Translation diversity: tx={std[0]:.4f} ty={std[1]:.4f} tz={std[2]:.4f}")

# Rotation matrix로 변환
from scipy.spatial.transform import Rotation
for i in range(3):
    euler = predict_q[i, 3:6].numpy()
    R = Rotation.from_euler('XYZ', euler).as_matrix()
    print(f"\nGrasp {i} rotation matrix:\n{R}")
```

#### `batch_stats.json` 구조

배치 처리 완료 후 생성되는 전체 통계 파일:

```json
[
  {
    "object": "core-bottle-1071fa4cddb2da2fc8724d5673a063a6_s0060",
    "num_grasps": 100,
    "time_network": 1.234,
    "time_optimization": 45.678
  },
  {
    "object": "core-mug-8570d9a8_s0080",
    "num_grasps": 100,
    "time_network": 1.198,
    "time_optimization": 44.321
  }
]
```

### Inference Pipeline 상세

DRO-Grasp의 inference는 5단계로 진행됩니다:

1. **D(R,O) 예측**: 네트워크가 robot point cloud와 object point cloud를 입력받아 (B, N, N) distance matrix를 예측
2. **Multilateration**: D(R,O) matrix와 object point cloud로부터 robot의 예상 contact point 위치를 복원
3. **SE(3) Registration**: 각 robot link의 point cloud를 predicted contact points에 Procrustes alignment (SVD)
4. **Position Target 추출**: SE(3) transform에서 translation만 추출하여 IK target으로 사용
5. **IK Optimization**: CVXPY 기반 iterative convex optimization으로 joint angles 계산 (기본 64회 반복)

### Python API로 사용

```python
from inference_dexgraspnet import DROGraspInference, load_object_pointcloud_from_obj

# 초기화
inference = DROGraspInference(
    drograsp_root='/path/to/DRO-Grasp',
    checkpoint='model_3robots',
    robot_name='shadowhand',
    device='cuda:0',
)

# 오브젝트 포인트 클라우드 준비
object_pc = load_object_pointcloud_from_obj(
    'meshdata/sem-Bottle-xxx/coacd/decomposed.obj',
    scale=0.06,
    num_points=1024,
)

# Grasp pose 예측
results = inference.predict_grasps(object_pc, num_grasps=10)

predict_q = results['predict_q']  # (10, 30) tensor
print(f"Translation: {predict_q[0, :3]}")
print(f"Rotation:    {predict_q[0, 3:6]}")
print(f"Joints:      {predict_q[0, 6:]}")

# 읽기 쉬운 dict 형태로 변환
q_dicts = inference.predict_q_to_dict(predict_q)
print(q_dicts[0])

inference.cleanup()
```

### Checkpoint 선택 가이드

| Checkpoint | 설명 | 사용 시나리오 |
|------------|------|-------------|
| `model_3robots` | Barrett + Allegro + ShadowHand 통합 | 범용 (권장) |
| `model_shadowhand` | ShadowHand 전용 | ShadowHand만 사용 시 |
| `model_3robots_partial` | 부분 포인트클라우드 대응 | 실제 센서 (occluded view) |

### CLI 옵션 (inference_dexgraspnet.py)

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--drograsp_root` | (필수) | DRO-Grasp 레포 루트 경로 |
| `--object_mesh` | - | 단일 OBJ 메쉬 (DexGraspNet 원본) |
| `--object_stl` | - | 단일 STL 메쉬 (변환 결과물) |
| `--object_pt` | - | 단일 .pt 포인트클라우드 |
| `--meshdata_dir` | - | DexGraspNet meshdata 디렉토리 (배치) |
| `--converted_dir` | - | 변환 결과 data 디렉토리 (배치) |
| `--object_scale` | 0.06 | OBJ 스케일 (DexGraspNet) |
| `--num_grasps` | 10 | 오브젝트당 생성할 grasp 수 |
| `--checkpoint` | model_3robots | 체크포인트 이름 |
| `--robot_name` | shadowhand | 로봇 (shadowhand/allegro/barrett) |
| `--device` | cuda:0 | 디바이스 |
| `--output_dir` | ./inference_results | 결과 저장 경로 |
| `--batch_objects` | 전체 | 배치 모드 최대 오브젝트 수 |
| `--n_iter` | 64 | IK 최적화 반복 수 |
| `--num_points` | 1024 | 포인트 클라우드 포인트 수 |
| `--split_batch` | 25 | IK 배치 분할 크기 (GPU 메모리) |

---

## 파일 설명

| 파일 | 설명 |
|------|------|
| `convert_dexgraspnet_to_droGrasp.py` | 메인 변환 스크립트 (DexGraspNet → DRO-Grasp 포맷) |
| `inference_dexgraspnet.py` | DRO-Grasp pretrained 모델로 grasp pose 추론 |
| `test_euler_conversion.py` | Euler angle 변환 정확성 검증 테스트 |
| `setup_data.sh` | 데이터 다운로드/배치/심볼릭 링크 셋업 가이드 |

---

## 참고 논문

- **DexGraspNet**: Wang et al., "DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation", ICRA 2023. [arXiv:2210.02697](https://arxiv.org/abs/2210.02697)
- **D(R,O) Grasp**: Wei et al., "D(R,O) Grasp: A Unified Representation of Robot and Object Interaction for Cross-Embodiment Dexterous Grasping", 2025. [arXiv:2410.01702](https://arxiv.org/abs/2410.01702)

## 라이선스

DexGraspNet 데이터셋은 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 라이선스