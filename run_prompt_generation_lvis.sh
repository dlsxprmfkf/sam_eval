#!/bin/bash
# LVIS Prompt 생성 스크립트 (ViTDet-H Cascade Mask R-CNN, GPU 1)
# 공식 설정: max_detections=300, score_threshold=0.02
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use explicit python from sam_eval env to avoid activation issues
PYTHON=/data_cast2/members/jeongmo/miniconda3/envs/sam_eval/bin/python

OUTPUT_FILE="prompts/prompts_lvis.json"
IMAGE_DIR="data/val2017_lvis_ver"
CONFIG_FILE="checkpoints/detector/detectron2_configs/LVIS/cascade_mask_rcnn_vitdet_h_100ep.py"
WEIGHTS_FILE="checkpoints/detector/model_final_11bbb7.pkl"

# GPU 설정 (요청: GPU 1)
export CUDA_VISIBLE_DEVICES=1

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=============================================="
echo "  LVIS Prompt 생성 (ViTDet-H)"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "=============================================="

total_images=$(ls -1 "$IMAGE_DIR"/*.jpg 2>/dev/null | wc -l)
echo -e "전체 이미지 수: ${GREEN}$total_images${NC}"

if [ -f "$OUTPUT_FILE" ]; then
    existing_count=$(OUTPUT_FILE_ENV="$OUTPUT_FILE" python3 - << 'EOF'
import json, os
path = os.environ.get('OUTPUT_FILE_ENV', '')
try:
    with open(path, 'r') as f:
        data = json.load(f)
    print(len({d['image_id'] for d in data}))
except Exception:
    print(0)
EOF
    )
    echo -e "기존 처리된 이미지 수: ${YELLOW}$existing_count${NC}"
else
    existing_count=0
    echo -e "기존 prompt 파일: ${RED}없음${NC}"
fi

echo ""
read -p "처리할 이미지 개수 입력 (최대 $total_images): " TARGET
if ! [[ "$TARGET" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}오류: 숫자를 입력해주세요.${NC}"
    exit 1
fi
if [ "$TARGET" -gt "$total_images" ]; then
    echo -e "${YELLOW}경고: 입력값이 전체 이미지 수보다 큽니다. 전체(${total_images})로 설정합니다.${NC}"
    TARGET=$total_images
fi

if [ "$existing_count" -ge "$TARGET" ]; then
    echo -e "${GREEN}이미 $existing_count 개 처리됨. 추가 작업 없음.${NC}"
    exit 0
fi

remaining=$((TARGET - existing_count))
echo ""
echo "=============================================="
echo -e "목표 이미지 수: ${GREEN}$TARGET${NC}"
echo -e "추가 처리 필요: ${GREEN}$remaining${NC}"
echo "=============================================="
echo ""

read -p "prompt 생성을 시작하시겠습니까? (y/n): " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "취소되었습니다."
    exit 0
fi

echo ""
echo -e "${GREEN}Prompt 생성을 시작합니다...${NC}"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Official ViTDet config: test_topk_per_image=300, test_score_thresh=0.02
$PYTHON scripts/generate_lvis_prompts.py \
    --image-dir "$IMAGE_DIR" \
    --output "$OUTPUT_FILE" \
    --config "$CONFIG_FILE" \
    --weights "$WEIGHTS_FILE" \
    --max-images "$TARGET" \
    --max-detections 300 \
    --score-threshold 0.02

echo ""
echo -e "${GREEN}완료!${NC}"
