#!/usr/bin/env python3
"""
Standard LVIS Evaluation for SAM Segmentation with Pruning Support.

This extends lvis_evaluation.py to support pruned SAM models.
"""

import os
import sys

# Add parent directory to path for _utils import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import time
import copy
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

try:
    from lvis import LVIS, LVISEval, LVISResults
    HAS_LVIS = True
except ImportError:
    print("Error: lvis library not installed.")
    print("Install with: pip install lvis")
    sys.exit(1)

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("Error: segment_anything not installed.")
    sys.exit(1)

# Import pruning utilities
from _utils import apply_sam_pruning


class PruningArgs:
    """Container for pruning arguments"""
    def __init__(self, sparsity_type='none', prune_mlp_ratio=0.0, prune_attn_ratio=0.0, 
                 rebuild_freq=1, act_mode='rms'):
        self.sparsity_type = sparsity_type
        self.prune_mlp_ratio = prune_mlp_ratio
        self.prune_attn_ratio = prune_attn_ratio
        self.rebuild_freq = rebuild_freq
        self.act_mode = act_mode


def instances_to_coco_json(
    masks: List[np.ndarray],
    boxes: List[List[float]],
    scores: List[float],
    category_ids: List[int],
    image_id: int
) -> List[Dict[str, Any]]:
    """Convert predictions to COCO JSON format."""
    results = []
    
    for mask, box, score, cat_id in zip(masks, boxes, scores, category_ids):
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle['counts'] = rle['counts'].decode('utf-8')
        
        result = {
            'image_id': int(image_id),
            'category_id': int(cat_id),
            'bbox': [float(x) for x in box],
            'score': float(score),
            'segmentation': rle,
        }
        results.append(result)
    
    return results


class StandardLVISEvaluatorWithPruning:
    """LVIS Evaluation pipeline for SAM with pruning support."""
    
    def __init__(
        self,
        sam_checkpoint: str,
        model_type: str,
        lvis_annotation_file: str,
        prompt_file: str,
        image_dir: str,
        device: str = "cuda",
        score_threshold: float = 0.0,
        max_detections_per_image: int = 300,
        output_dir: str = "eval_results_lvis",
        save_predictions: bool = False,
        pruning_args: Optional[PruningArgs] = None,
    ):
        self.device = device
        self.score_threshold = score_threshold
        self.max_detections_per_image = max_detections_per_image
        self.output_dir = output_dir
        self.save_predictions = save_predictions
        
        # Load SAM model
        print(f"Loading SAM model: {model_type}")
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device)
        
        # Apply pruning if specified
        if pruning_args and pruning_args.sparsity_type != 'none':
            print(f"\n{'='*60}")
            print("Applying Pruning Configuration:")
            print(f"  Sparsity Type: {pruning_args.sparsity_type}")
            print(f"  MLP Ratio: {pruning_args.prune_mlp_ratio}")
            print(f"  Attn Ratio: {pruning_args.prune_attn_ratio}")
            print(f"  Rebuild Freq: {pruning_args.rebuild_freq}")
            print(f"  Act Mode: {pruning_args.act_mode}")
            print(f"{'='*60}\n")
            sam = apply_sam_pruning(sam, pruning_args)
        
        sam.eval()
        self.predictor = SamPredictor(sam)
        print("  SAM model loaded!")
        
        # Load LVIS ground truth
        print(f"Loading LVIS ground truth: {lvis_annotation_file}")
        self.lvis_gt = LVIS(lvis_annotation_file)
        self.coco_gt = COCO(lvis_annotation_file)  # Also load as COCO for image loading
        print(f"  Loaded {len(self.coco_gt.getImgIds())} images")
        
        # Ensure iscrowd field exists (LVIS doesn't have iscrowd)
        for ann_id in self.coco_gt.anns:
            if 'iscrowd' not in self.coco_gt.anns[ann_id]:
                self.coco_gt.anns[ann_id]['iscrowd'] = 0
        
        # Load prompts
        print(f"Loading detector prompts: {prompt_file}")
        with open(prompt_file, 'r') as f:
            prompts = json.load(f)
        
        self.prompts_by_image = defaultdict(list)
        for prompt in prompts:
            self.prompts_by_image[prompt['image_id']].append(prompt)
        
        print(f"  Loaded {len(prompts)} prompts for {len(self.prompts_by_image)} images")
        
        self.image_dir = image_dir
    
    def run_sam_on_boxes_batch(
        self,
        image: np.ndarray,
        boxes: List[List[float]]
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Run SAM inference on a batch of bounding boxes."""
        if len(boxes) == 0:
            return [], np.array([])
        
        # Convert COCO format [x, y, w, h] to SAM format [x1, y1, x2, y2]
        boxes_xyxy = []
        for box in boxes:
            x, y, w, h = box
            boxes_xyxy.append([x, y, x + w, y + h])
        
        input_boxes = torch.tensor(boxes_xyxy, dtype=torch.float32)
        
        image_shape = image.shape[:2]
        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            input_boxes, image_shape
        ).to(self.device)
        
        masks, iou_predictions, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=True,
        )
        
        # Select best mask for each box
        best_mask_indices = torch.argmax(iou_predictions, dim=1)
        pred_masks = masks[torch.arange(len(masks)), best_mask_indices]
        pred_iou_scores = iou_predictions[torch.arange(len(masks)), best_mask_indices]
        
        pred_masks_np = pred_masks.cpu().numpy()
        binary_masks = [(mask > 0.5).astype(np.uint8) for mask in pred_masks_np]
        
        return binary_masks, pred_iou_scores.cpu().numpy()
    
    def evaluate(self, max_images: Optional[int] = None, batch_size: int = 1) -> Dict:
        """Run full evaluation pipeline with optional batching.
        
        Args:
            max_images: Maximum number of images to evaluate
            batch_size: Number of images to process in parallel (default: 1 for sequential)
        """
        image_ids = sorted(self.prompts_by_image.keys())
        if max_images is not None and max_images > 0:
            image_ids = image_ids[:max_images]
        
        print(f"\nEvaluating {len(image_ids)} images (batch_size={batch_size})...")
        
        all_predictions = []
        
        with torch.no_grad():
            # Process images in batches
            for batch_start in tqdm(range(0, len(image_ids), batch_size), desc="Processing batches"):
                batch_end = min(batch_start + batch_size, len(image_ids))
                batch_image_ids = image_ids[batch_start:batch_end]
                
                for image_id in batch_image_ids:
                    try:
                        # LVIS file_name can be 'coco/val2017/000000123456.jpg' format
                        img_info = self.coco_gt.loadImgs(image_id)[0]
                        fname = img_info.get('file_name', f"{image_id:012d}.jpg")
                        fname = os.path.basename(fname)
                        image_path = os.path.join(self.image_dir, fname)
                        
                        if not os.path.exists(image_path):
                            print(f"Warning: Image not found: {image_path}")
                            continue
                        
                        # Load and preprocess image
                        img_pil = Image.open(image_path)
                        img = ImageOps.exif_transpose(img_pil).convert('RGB')
                        image = np.array(img)
                        
                        self.predictor.set_image(image)
                        
                        # Get prompts for this image
                        prompts = self.prompts_by_image[image_id]
                        prompts = [p for p in prompts if p['score'] >= self.score_threshold]
                        prompts = sorted(prompts, key=lambda x: x['score'], reverse=True)
                        prompts = prompts[:self.max_detections_per_image]
                        
                        if len(prompts) == 0:
                            continue
                        
                        boxes = [p['bbox'] for p in prompts]
                        det_scores = [p['score'] for p in prompts]
                        category_ids = [p['category_id'] for p in prompts]
                        
                        # Run SAM
                        masks, sam_scores = self.run_sam_on_boxes_batch(image, boxes)
                        
                        # Combine detection and SAM scores
                        final_scores = [float(ds) * float(ss) for ds, ss in zip(det_scores, sam_scores)]
                        
                        # Convert to COCO format
                        predictions = instances_to_coco_json(
                            masks=masks,
                            boxes=boxes,
                            scores=final_scores,
                            category_ids=category_ids,
                            image_id=image_id,
                        )
                        
                        all_predictions.extend(predictions)
                        
                    except Exception as e:
                        print(f"Error processing image {image_id}: {e}")
                        continue
        
        print(f"\nGenerated {len(all_predictions)} predictions")
        
        # Run COCO evaluation
        if len(all_predictions) == 0:
            print("Warning: No predictions generated!")
            return {}
        
        # Save predictions if requested
        if self.save_predictions:
            os.makedirs(self.output_dir, exist_ok=True)
            pred_file = os.path.join(self.output_dir, 'lvis_predictions.json')
            with open(pred_file, 'w') as f:
                json.dump(all_predictions, f)
            print(f"Predictions saved to {pred_file}")
        
        # Evaluate using LVIS API
        import tempfile
        
        # Filter LVIS GT to only evaluated images
        pred_image_ids = set(p['image_id'] for p in all_predictions)
        print(f"Filtering LVIS GT: {len(pred_image_ids)} images out of {len(self.lvis_gt.imgs)} total")
        
        # Create filtered LVIS GT
        filtered_images = [img for img in self.lvis_gt.dataset['images'] if img['id'] in pred_image_ids]
        filtered_annotations = [ann for ann in self.lvis_gt.dataset['annotations'] if ann['image_id'] in pred_image_ids]
        
        filtered_dataset = {
            'images': filtered_images,
            'annotations': filtered_annotations,
            'categories': self.lvis_gt.dataset['categories'],
        }
        
        for key in ['info', 'licenses']:
            if key in self.lvis_gt.dataset:
                filtered_dataset[key] = self.lvis_gt.dataset[key]
        
        print(f"  Filtered GT: {len(filtered_images)} images, {len(filtered_annotations)} annotations")
        
        # Save filtered GT and predictions to temp files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(filtered_dataset, f)
            temp_gt_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(all_predictions, f)
            temp_pred_path = f.name
        
        try:
            filtered_lvis = LVIS(temp_gt_path)
            lvis_results = LVISResults(filtered_lvis, temp_pred_path, max_dets=self.max_detections_per_image)
            lvis_eval = LVISEval(filtered_lvis, lvis_results, 'segm')
            lvis_eval.run()
            lvis_eval.print_results()
            
            # Get results
            results_dict = lvis_eval.get_results()
            results = {
                'AP': float(results_dict.get('AP', 0.0) * 100),
                'AP50': float(results_dict.get('AP50', 0.0) * 100),
                'AP75': float(results_dict.get('AP75', 0.0) * 100),
                'APs': float(results_dict.get('APs', 0.0) * 100),
                'APm': float(results_dict.get('APm', 0.0) * 100),
                'APl': float(results_dict.get('APl', 0.0) * 100),
                'APr': float(results_dict.get('APr', 0.0) * 100),
                'APc': float(results_dict.get('APc', 0.0) * 100),
                'APf': float(results_dict.get('APf', 0.0) * 100),
                'num_predictions': len(all_predictions),
                'num_images': len(pred_image_ids),
            }
        except Exception as e:
            print(f"LVIS evaluation error: {e}")
            import traceback
            traceback.print_exc()
            results = {}
        finally:
            if os.path.exists(temp_gt_path):
                os.remove(temp_gt_path)
            if os.path.exists(temp_pred_path):
                os.remove(temp_pred_path)
        
        return results
    
    def print_results(self, results: Dict):
        """Print evaluation results."""
        print("\n" + "=" * 60)
        print("LVIS Evaluation Results")
        print("=" * 60)
        print(f"  AP          : {results.get('AP', 0):.3f}")
        print(f"  AP50        : {results.get('AP50', 0):.3f}")
        print(f"  AP75        : {results.get('AP75', 0):.3f}")
        print("-" * 60)
        print("  Size-based:")
        print(f"    APs (small) : {results.get('APs', 0):.3f}")
        print(f"    APm (medium): {results.get('APm', 0):.3f}")
        print(f"    APl (large) : {results.get('APl', 0):.3f}")
        print("-" * 60)
        print("  Frequency-based:")
        print(f"    APr (rare)    : {results.get('APr', 0):.3f}")
        print(f"    APc (common)  : {results.get('APc', 0):.3f}")
        print(f"    APf (frequent): {results.get('APf', 0):.3f}")
        print("=" * 60)
        print(f"  Images evaluated: {results.get('num_images', 0)}")
        print(f"  Total predictions: {results.get('num_predictions', 0)}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Standard LVIS Evaluation for SAM with Pruning Support'
    )
    parser.add_argument('--sam-checkpoint', type=str, required=True)
    parser.add_argument('--model-type', type=str, default='vit_l',
                        choices=['vit_h', 'vit_l', 'vit_b'])
    parser.add_argument('--lvis-annotation', type=str, required=True)
    parser.add_argument('--prompt-file', type=str, required=True)
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--max-images', type=int, default=-1,
                        help='Maximum number of images to evaluate (-1 for all)')
    parser.add_argument('--score-threshold', type=float, default=0.0)
    parser.add_argument('--max-detections', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Number of images to process in parallel (default: 1)')
    parser.add_argument('--output-dir', type=str, default='eval_results_lvis')
    parser.add_argument('--output-file', type=str, default='lvis_results.json')
    parser.add_argument('--save-predictions', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sparsity-type', type=str, default='none',
                        choices=['none', 'sp', 'unstructured'],
                        help='Pruning type: none, sp (structured), or unstructured')
    parser.add_argument('--prune-mlp-ratio', type=float, default=0.0,
                        help='MLP pruning ratio (0.0-1.0)')
    parser.add_argument('--prune-attn-ratio', type=float, default=0.0,
                        help='Attention pruning ratio (0.0-1.0)')
    parser.add_argument('--rebuild-freq', type=int, default=1,
                        help='Rebuild frequency for pruning masks')
    parser.add_argument('--act-mode', type=str, default='rms',
                        help='Activation mode for importance calculation')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create pruning args
    pruning_args = PruningArgs(
        sparsity_type=args.sparsity_type,
        prune_mlp_ratio=args.prune_mlp_ratio,
        prune_attn_ratio=args.prune_attn_ratio,
        rebuild_freq=args.rebuild_freq,
        act_mode=args.act_mode,
    )
    
    evaluator = StandardLVISEvaluatorWithPruning(
        sam_checkpoint=args.sam_checkpoint,
        model_type=args.model_type,
        lvis_annotation_file=args.lvis_annotation,
        prompt_file=args.prompt_file,
        image_dir=args.image_dir,
        device=args.device,
        score_threshold=args.score_threshold,
        max_detections_per_image=args.max_detections,
        output_dir=args.output_dir,
        save_predictions=args.save_predictions,
        pruning_args=pruning_args,
    )
    
    print("\n" + "=" * 60)
    print("Starting LVIS Evaluation with Pruning")
    print("=" * 60)
    
    start_time = time.time()
    max_imgs = None if args.max_images == -1 else args.max_images
    results = evaluator.evaluate(max_images=max_imgs, batch_size=args.batch_size)
    elapsed_time = time.time() - start_time
    
    if results:
        evaluator.print_results(results)
        print(f"\nTotal time: {elapsed_time:.2f}s")
        
        # Save results
        results_file = os.path.join(args.output_dir, args.output_file)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
