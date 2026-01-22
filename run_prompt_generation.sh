#!/bin/bash
# COCO Prompt 생성 스크립트 (FocalNet-DINO 기반)
# GPU 1번 사용
# 공식 설정: max_detections=300 (num_select=300)

set -e

# 기본 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use explicit python from sam_eval env to avoid activation issues
PYTHON=/data_cast2/members/jeongmo/miniconda3/envs/sam_eval/bin/python

OUTPUT_FILE="prompts/prompts_coco.json"
IMAGE_DIR="data/val2017"

# GPU 설정
export CUDA_VISIBLE_DEVICES=1

# 색상 설정
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "  COCO Prompt 생성 (FocalNet-DINO)"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "=============================================="

# 전체 이미지 수 확인
TOTAL_IMAGES=$(ls -1 "$IMAGE_DIR"/*.jpg 2>/dev/null | wc -l)
echo -e "전체 이미지 수: ${GREEN}$TOTAL_IMAGES${NC}"

# 기존 prompt 파일 확인
if [ -f "$OUTPUT_FILE" ]; then
    # JSON 파일에서 unique image_id 개수 확인
    EXISTING_COUNT=$(python3 -c "
import json
with open('$OUTPUT_FILE', 'r') as f:
    data = json.load(f)
unique_ids = len(set(d['image_id'] for d in data))
print(unique_ids)
" 2>/dev/null || echo "0")
    echo -e "기존 처리된 이미지 수: ${YELLOW}$EXISTING_COUNT${NC}"
else
    EXISTING_COUNT=0
    echo -e "기존 prompt 파일: ${RED}없음${NC}"
fi

# 원하는 개수 입력
echo ""
read -p "처리할 이미지 개수 입력 (최대 $TOTAL_IMAGES): " TARGET_COUNT

# 입력 검증
if ! [[ "$TARGET_COUNT" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}오류: 숫자를 입력해주세요.${NC}"
    exit 1
fi

if [ "$TARGET_COUNT" -gt "$TOTAL_IMAGES" ]; then
    echo -e "${YELLOW}경고: 입력값이 전체 이미지 수보다 큽니다. 전체 이미지($TOTAL_IMAGES)로 설정합니다.${NC}"
    TARGET_COUNT=$TOTAL_IMAGES
fi

# 처리 필요 여부 확인
if [ "$EXISTING_COUNT" -ge "$TARGET_COUNT" ]; then
    echo ""
    echo -e "${GREEN}이미 $EXISTING_COUNT 개의 이미지가 처리되어 있습니다.${NC}"
    echo -e "${GREEN}추가 처리가 필요하지 않습니다.${NC}"
    exit 0
fi

REMAINING=$((TARGET_COUNT - EXISTING_COUNT))
echo ""
echo "=============================================="
echo -e "목표 이미지 수: ${GREEN}$TARGET_COUNT${NC}"
echo -e "기존 처리 완료: ${YELLOW}$EXISTING_COUNT${NC}"
echo -e "추가 처리 필요: ${GREEN}$REMAINING${NC}"
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

# Python 스크립트 실행 (처리할 최대 개수 전달)
$PYTHON scripts/generate_all_prompts.py --max_images "$TARGET_COUNT"

echo ""
echo -e "${GREEN}완료!${NC}"
