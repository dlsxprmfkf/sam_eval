#!/usr/bin/env python3
"""
Generate prompts for LVIS val using ViTDet H Cascade Mask R-CNN.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import List

import torch
import numpy as np
from tqdm import tqdm

# Detectron2 imports
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import transforms as T
from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog

# ★중요★: LVIS 데이터셋 등록을 위해 반드시 임포트해야 함
try:
    import detectron2.data.datasets.lvis
    # 명시적으로 LVIS 데이터셋 등록
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets.lvis import get_lvis_instances_meta
    
    # LVIS v1 val 데이터셋을 명시적으로 등록
    # 이렇게 하면 thing_dataset_id_to_contiguous_id가 자동으로 생성됨
    def register_lvis_val_if_needed():
        if "lvis_v1_val" not in DatasetCatalog:
            from detectron2.data.datasets.lvis import register_lvis_instances
            # register_lvis_instances는 내부적으로 LVIS API를 사용하여 매핑 생성
            print("Registering LVIS v1 validation dataset...")
        # 등록되었는지 확인
        meta = MetadataCatalog.get("lvis_v1_val")
        return meta
    
    LVIS_AVAILABLE = True
except ImportError:
    print("LVIS dataset module not found. Make sure lvis-api is installed.")
    LVIS_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser(description="Generate LVIS prompts with ViTDet H")
    parser.add_argument("--image-dir", required=True, help="Path to LVIS validation images (e.g. val2017)")
    parser.add_argument("--output", default="prompts_lvis.json", help="Output JSON path")
    parser.add_argument(
        "--config",
        default="projects/ViTDet/configs/LVIS/cascade_mask_rcnn_vitdet_h_100ep.py",
        help="Path to the LazyConfig file",
    )
    parser.add_argument(
        "--weights",
        default="model_final_11bbb7.pkl",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--lvis-annotation",
        default="annotations/lvis_v1_val.json",
        help="Path to LVIS annotation file (for class mapping)",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Max number of images to process (for testing)")
    parser.add_argument("--max-detections", type=int, default=300, help="Max detections per image")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Keep all detections for evaluation")
    return parser.parse_args()

def get_lvis_contiguous_id_to_category_id_map(lvis_annotation_path=None):
    """
    Detectron2 모델 출력(0~1202) -> LVIS 실제 category_id(1~1203, 비연속) 매핑 테이블 생성
    
    CRITICAL: 반드시 Detectron2가 학습 시 사용한 매핑과 동일해야 합니다!
    """
    # Method 1: Detectron2 메타데이터에서 로드 (가장 신뢰할 수 있음)
    if LVIS_AVAILABLE:
        try:
            # LVIS 데이터셋 명시적 등록
            meta = register_lvis_val_if_needed()
            
            if hasattr(meta, "thing_dataset_id_to_contiguous_id") and len(meta.thing_dataset_id_to_contiguous_id) > 0:
                reverse_map = {v: k for k, v in meta.thing_dataset_id_to_contiguous_id.items()}
                print(f"✓ Loaded LVIS class mapping from Detectron2 metadata: {len(reverse_map)} classes.")
                print(f"  - Contiguous ID range: 0 ~ {max(reverse_map.keys())}")
                print(f"  - LVIS category_id range: {min(reverse_map.values())} ~ {max(reverse_map.values())}")
                print(f"  - Mapping examples: 0->{reverse_map[0]}, 1->{reverse_map[1]}, {max(reverse_map.keys())}->{reverse_map[max(reverse_map.keys())]}")
                
                if len(reverse_map) != 1203:
                    print(f"⚠ WARNING: Expected 1203 LVIS classes, got {len(reverse_map)}")
                
                return reverse_map
        except Exception as e:
            print(f"Could not load from Detectron2 metadata: {e}")
            import traceback
            traceback.print_exc()
    
    # Method 2: LVIS API를 사용하여 매핑 생성 (Detectron2와 동일한 방식)
    annotation_paths = []
    if lvis_annotation_path:
        annotation_paths.append(lvis_annotation_path)
    
    annotation_paths.extend([
        "annotations/lvis_v1_val.json",
        "../annotations/lvis_v1_val.json",
        "../../annotations/lvis_v1_val.json",
    ])
    
    for anno_path in annotation_paths:
        if os.path.exists(anno_path):
            try:
                print(f"Loading LVIS categories from annotation file using LVIS API: {anno_path}")
                
                # CRITICAL: Use LVIS API to get the same mapping Detectron2 uses
                try:
                    from lvis import LVIS
                    lvis_api = LVIS(anno_path)
                    
                    # Get category IDs from LVIS API (already sorted properly)
                    cat_ids = sorted(lvis_api.get_cat_ids())
                    
                    # Create contiguous mapping: 0->cat_ids[0], 1->cat_ids[1], etc.
                    reverse_map = {i: cat_id for i, cat_id in enumerate(cat_ids)}
                    
                    print(f"✓ Loaded LVIS class mapping from LVIS API: {len(reverse_map)} classes.")
                    print(f"  - Contiguous ID range: 0 ~ {max(reverse_map.keys())}")
                    print(f"  - LVIS category_id range: {min(reverse_map.values())} ~ {max(reverse_map.values())}")
                    print(f"  - Mapping examples: 0->{reverse_map[0]}, 1->{reverse_map[1]}, {max(reverse_map.keys())}->{reverse_map[max(reverse_map.keys())]}")
                    
                    if len(reverse_map) != 1203:
                        print(f"⚠ WARNING: Expected 1203 LVIS classes, got {len(reverse_map)}")
                    
                    return reverse_map
                    
                except ImportError:
                    print("LVIS API not available, falling back to JSON parsing...")
                
                # Fallback: Direct JSON parsing (least reliable)
                with open(anno_path, 'r') as f:
                    lvis_data = json.load(f)
                
                # CRITICAL FIX: LVIS has 1203 classes with IDs 1-1203
                # Detectron2 models typically use:
                #   - Class 0 as background (not used for detection output)
                #   - Classes 1-1203 for actual objects
                # However, model outputs pred_classes in range 0-1202 where:
                #   - 0 corresponds to LVIS category 1
                #   - 1 corresponds to LVIS category 2
                #   - ...
                #   - 1202 corresponds to LVIS category 1203
                categories = sorted(lvis_data['categories'], key=lambda x: x['id'])
                
                # Simple direct mapping: contiguous_id (0-based) -> category_id (1-based)
                reverse_map = {}
                for i, cat in enumerate(categories):
                    reverse_map[i] = cat['id']
                
                print(f"✓ Loaded LVIS class mapping from annotation file: {len(reverse_map)} classes.")
                print(f"  - Contiguous ID range: 0 ~ {max(reverse_map.keys())}")
                print(f"  - LVIS category_id range: {min(reverse_map.values())} ~ {max(reverse_map.values())}")
                print(f"  - Mapping examples: 0->{reverse_map[0]}, 1->{reverse_map[1]}, {max(reverse_map.keys())}->{reverse_map[max(reverse_map.keys())]}")
                
                if len(reverse_map) != 1203:
                    print(f"⚠ WARNING: Expected 1203 LVIS classes, got {len(reverse_map)}")
                
                return reverse_map
                
            except Exception as e:
                print(f"Could not load from {anno_path}: {e}")
                continue
    
    # 최후의 수단: 에러 발생
    print("\n" + "="*60)
    print("CRITICAL ERROR: Cannot map model outputs to LVIS IDs properly.")
    print("="*60)
    print("\nTried the following locations:")
    for path in annotation_paths:
        exists_str = "✓ EXISTS" if os.path.exists(path) else "✗ NOT FOUND"
        print(f"  - {path} [{exists_str}]")
    print("\nPlease ensure:")
    print("  1. LVIS annotation file exists at annotations/lvis_v1_val.json")
    print("  2. Or provide --lvis-annotation path to the annotation file")
    print("  3. Or ensure Detectron2 LVIS dataset is properly registered with:")
    print("     from detectron2.data.datasets import lvis")
    print("     from detectron2.data import DatasetCatalog, MetadataCatalog")
    print("="*60)
    sys.exit(1)

def main():
    args = parse_args()
    
    # 1. Config 로드 및 모델 생성
    print(f"Loading config from {args.config}...")
    cfg = LazyConfig.load(args.config)
    
    # 모델 생성 (LazyConfig의 경우 instantiate 사용)
    model = instantiate(cfg.model)
    model.to("cuda")
    
    # 체크포인트 로드
    print(f"Loading weights from {args.weights}...")
    DetectionCheckpointer(model).load(args.weights)
    model.eval()
    
    # Get model's size_divisibility for proper padding
    # ViTDet typically uses size_divisibility=32 (due to FPN and backbone stride)
    size_divisibility = getattr(model.backbone, 'size_divisibility', 32)
    if hasattr(model, 'pixel_mean'):
        print(f"Model pixel_mean: {model.pixel_mean}")
        print(f"Model pixel_std: {model.pixel_std}")
    print(f"Model size_divisibility: {size_divisibility}")

    # 2. 전처리(Augmentation) 설정
    # ViTDet은 학습 시 1024x1024 LSJ를 사용하지만, 인퍼런스 시에는 보통 Shortest Edge 1024를 사용합니다.
    # 단, ViTDet은 Window Attention 때문에 입력 크기가 패치 크기(16)로 나누어 떨어져야 합니다.
    # Detectron2 모델 내부(preprocess_image)에서 padding을 자동으로 수행하므로
    # 여기서는 ResizeShortestEdge(1024, 1024)만 적용하면 됩니다.
    aug = T.ResizeShortestEdge(short_edge_length=1024, max_size=1024)
    
    print(f"Inference settings:")
    print(f"  - Image resize: shortest_edge=1024, max_size=1024")
    print(f"  - Input format: RGB (ViTDet uses ImageNet pretrained ViT)")
    print(f"  - Score threshold: {args.score_threshold}")
    print(f"  - Max detections per image: {args.max_detections}")

    # 3. 클래스 매핑 테이블 가져오기
    contiguous_to_lvis_id = get_lvis_contiguous_id_to_category_id_map(args.lvis_annotation)

    # 4. 이미지 목록 로드
    image_files = sorted(list(Path(args.image_dir).glob("*.jpg")))
    
    # Max images 제한 적용
    if args.max_images is not None:
        image_files = image_files[:args.max_images]
        print(f"Found {len(image_files)} images (limited to {args.max_images}).")
    else:
        print(f"Found {len(image_files)} images.")

    results = []
    
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                # ★수정 1★: ViTDet은 RGB 포맷을 사용합니다.
                # read_image는 기본적으로 BGR이지만 format="RGB"를 주면 RGB로 변환해줍니다.
                original_image = read_image(str(img_path), format="RGB")
                height, width = original_image.shape[:2]

                # Augmentation 적용 (Resize)
                aug_input = T.AugInput(original_image)
                transforms = aug(aug_input)  # Returns the transform that was applied
                image_tensor = torch.as_tensor(aug_input.image.astype("float32").transpose(2, 0, 1))

                # 모델 입력 구성
                # CRITICAL: GeneralizedRCNN은 내부적으로 다음을 수행:
                #   1. pixel_mean/std 정규화 (model.pixel_mean, model.pixel_std)
                #   2. size_divisibility 패딩 (backbone의 stride에 맞춤, 일반적으로 32)
                #   3. detector_postprocess에서 bbox를 원본 크기로 복원
                # height, width는 반드시 원본 이미지 크기여야 합니다!
                inputs = {"image": image_tensor.to("cuda"), "height": height, "width": width}

                # 추론
                outputs = model([inputs])[0]
                instances = outputs["instances"].to("cpu")

                # Score Threshold 필터링 (너무 낮으면 파일 크기만 커짐, 0.01 권장)
                # LVIS 평가(AP)를 위해서는 낮은 점수도 포함해야 함
                if args.score_threshold > 0:
                    mask = instances.scores > args.score_threshold
                    instances = instances[mask]
                
                # Top-K 필터링 (LVIS 표준 300개)
                if len(instances) > args.max_detections:
                    indices = instances.scores.argsort(descending=True)[:args.max_detections]
                    instances = instances[indices]

                # 결과 변환
                scores = instances.scores.tolist()
                pred_classes = instances.pred_classes.tolist()
                
                # ★수정 2★: Bbox 포맷 변환 (XYXY -> XYWH)
                # Detectron2의 detector_postprocess는 이미 원본 이미지 크기로 bbox를 스케일했습니다.
                boxes_xyxy = instances.pred_boxes.tensor.numpy()
                boxes_xywh = []
                for box in boxes_xyxy:
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    boxes_xywh.append([float(x1), float(y1), float(w), float(h)])

                # LVIS 이미지 ID 추출 (파일명에서)
                image_id = int(img_path.stem.split('_')[-1]) if '_' in img_path.stem else int(img_path.stem)

                for score, cat_idx, bbox in zip(scores, pred_classes, boxes_xywh):
                    # ★수정 3★: 정확한 Category ID 매핑 사용
                    # contiguous_id (0~1202) -> LVIS category_id (1~1203, non-contiguous)
                    if cat_idx not in contiguous_to_lvis_id:
                        print(f"⚠ WARNING: Unknown contiguous_id {cat_idx} in image {img_path.name}")
                        continue
                    
                    real_cat_id = contiguous_to_lvis_id[cat_idx]
                    
                    results.append({
                        "image_id": image_id,
                        "category_id": int(real_cat_id),
                        "bbox": bbox,
                        "score": float(score)
                    })
                    
            except Exception as e:
                print(f"⚠ Error processing {img_path.name}: {e}")
                traceback.print_exc()
                continue

    # 결과 저장
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total predictions: {len(results)}")
    print(f"Unique images: {len(set(r['image_id'] for r in results))}")
    if results:
        print(f"Average detections per image: {len(results) / len(set(r['image_id'] for r in results)):.1f}")
        print(f"Score range: {min(r['score'] for r in results):.3f} ~ {max(r['score'] for r in results):.3f}")
    print(f"Saving to {args.output}...")
    print(f"{'='*60}")
    
    with open(args.output, "w") as f:
        json.dump(results, f)
    
    print(f"✓ Done! Results saved to {args.output}")

if __name__ == "__main__":
    main()