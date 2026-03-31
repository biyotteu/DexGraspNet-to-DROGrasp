#!/bin/bash
# =============================================================================
# DexGraspNet → DRO-Grasp 데이터 셋업 스크립트
# =============================================================================
#
# 사용법:
#   1. 아래 경로를 실제 환경에 맞게 수정
#   2. chmod +x setup_data.sh && ./setup_data.sh
#
# PKU 미러 다운로드:
#   https://mirrors.pku.edu.cn/dl-release/DexGraspNet-ICRA2023/
#   예상 파일: meshdata.tar.gz (또는 유사), dataset.tar.gz (또는 유사)
# =============================================================================

set -e

# ======================== 경로 설정 (수정 필요!) ========================
DEXGRASPNET_ROOT="$HOME/DexGraspNet"          # DexGraspNet 레포 클론 경로
DROGRASP_ROOT="$HOME/DRO-Grasp"               # DRO-Grasp 레포 클론 경로
DOWNLOAD_DIR="$HOME/downloads/dexgraspnet"    # PKU에서 다운받은 파일 위치
OUTPUT_DIR="$HOME/dexgraspnet_converted"      # 변환 결과 출력 경로
# ======================================================================

echo "============================================"
echo " Step 0: 디렉토리 구조 확인"
echo "============================================"

# 필요한 디렉토리 생성
mkdir -p "$DEXGRASPNET_ROOT/data/dataset"
mkdir -p "$DEXGRASPNET_ROOT/data/meshdata"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "============================================"
echo " Step 1: DexGraspNet 데이터 압축 해제"
echo "============================================"
echo ""
echo "PKU 미러에서 다운로드한 파일을 아래와 같이 배치하세요:"
echo ""
echo "  $DOWNLOAD_DIR/"
echo "  ├── meshdata/           ← 오브젝트 메쉬 (OBJ 파일들)"
echo "  │   ├── core-mug-8570d9a8.../coacd/decomposed.obj"
echo "  │   ├── sem-Bottle-437678.../coacd/decomposed.obj"
echo "  │   └── ... (5355개 오브젝트)"
echo "  └── dataset/            ← grasp 데이터 (NPY 파일들)"
echo "      ├── core-mug-8570d9a8....npy"
echo "      ├── sem-Bottle-437678....npy"
echo "      └── ... (5355개 파일)"
echo ""
echo "압축 파일이라면 먼저 해제하세요:"
echo "  tar -xzf meshdata.tar.gz -C $DOWNLOAD_DIR/"
echo "  tar -xzf dataset.tar.gz -C $DOWNLOAD_DIR/"
echo ""

# 파일 존재 확인
if [ ! -d "$DOWNLOAD_DIR/meshdata" ] && [ ! -d "$DOWNLOAD_DIR/dataset" ]; then
    echo "ERROR: $DOWNLOAD_DIR에 meshdata/ 또는 dataset/ 폴더가 없습니다."
    echo "PKU 미러에서 다운로드 후 압축을 해제하세요."
    exit 1
fi

echo "============================================"
echo " Step 2: DexGraspNet 레포에 심볼릭 링크 생성"
echo "============================================"

# DexGraspNet 레포의 data/ 디렉토리에 심볼릭 링크
if [ -d "$DOWNLOAD_DIR/dataset" ]; then
    ln -sfn "$DOWNLOAD_DIR/dataset" "$DEXGRASPNET_ROOT/data/dataset"
    echo "  ✓ dataset → $DEXGRASPNET_ROOT/data/dataset"
fi

if [ -d "$DOWNLOAD_DIR/meshdata" ]; then
    ln -sfn "$DOWNLOAD_DIR/meshdata" "$DEXGRASPNET_ROOT/data/meshdata"
    echo "  ✓ meshdata → $DEXGRASPNET_ROOT/data/meshdata"
fi

echo ""
echo "결과 구조:"
echo "  $DEXGRASPNET_ROOT/"
echo "  └── data/"
echo "      ├── dataset → $DOWNLOAD_DIR/dataset (심볼릭 링크)"
echo "      └── meshdata → $DOWNLOAD_DIR/meshdata (심볼릭 링크)"

echo ""
echo "============================================"
echo " Step 3: 변환 실행"
echo "============================================"

echo "변환 명령어:"
echo ""
echo "  python convert_dexgraspnet_to_droGrasp.py \\"
echo "      --dexgraspnet_root $DEXGRASPNET_ROOT \\"
echo "      --droGrasp_root $DROGRASP_ROOT \\"
echo "      --output_dir $OUTPUT_DIR \\"
echo "      --verify"
echo ""

# 실제 실행 (주석 해제하여 사용)
# python convert_dexgraspnet_to_droGrasp.py \
#     --dexgraspnet_root "$DEXGRASPNET_ROOT" \
#     --droGrasp_root "$DROGRASP_ROOT" \
#     --output_dir "$OUTPUT_DIR" \
#     --verify

echo "============================================"
echo " Step 4: DRO-Grasp에 심볼릭 링크 생성"
echo "============================================"
echo ""
echo "변환 완료 후 아래 명령어로 DRO-Grasp에 연결하세요:"
echo ""
echo "  # 변환된 CMapDataset"
echo "  ln -sfn $OUTPUT_DIR/data/CMapDataset_filtered $DROGRASP_ROOT/data/CMapDataset_filtered"
echo ""
echo "  # 변환된 오브젝트 메쉬 (STL)"
echo "  ln -sfn $OUTPUT_DIR/data/data_urdf/object/dexgraspnet $DROGRASP_ROOT/data/data_urdf/object/dexgraspnet"
echo ""
echo "  # 변환된 오브젝트 포인트클라우드"
echo "  ln -sfn $OUTPUT_DIR/data/PointCloud/object/dexgraspnet $DROGRASP_ROOT/data/PointCloud/object/dexgraspnet"
echo ""
echo "또는 전체를 한번에 복사:"
echo "  cp -r $OUTPUT_DIR/data/* $DROGRASP_ROOT/data/"
echo ""

echo "============================================"
echo " Step 5: DRO-Grasp config 수정"
echo "============================================"
echo ""
echo "  $DROGRASP_ROOT/configs/dataset/cmap_dataset.yaml에서:"
echo "    robot_names:"
echo "      - 'shadowhand'    ← shadowhand만 남기기"
echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
