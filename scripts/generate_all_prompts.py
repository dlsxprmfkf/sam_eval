#!/usr/bin/env python3
"""
Generate FocalNet-DINO prompts for all COCO val2017 images with batch processing and progress saving
"""

import sys
sys.path.insert(0, 'checkpoints/detector/FocalNet-DINO')

import os
import json
import torch
import argparse
from PIL import Image, ImageOps
from tqdm import tqdm
import torchvision.transforms as T

from util.slconfig import SLConfig
from models.registry import MODULE_BUILD_FUNCS

# Argument parser
parser = argparse.ArgumentParser(description='Generate COCO prompts using FocalNet-DINO')
parser.add_argument('--max_images', type=int, default=None,
                    help='Maximum number of images to process (default: all)')
parsed_args = parser.parse_args()

# Config
config_path = 'checkpoints/detector/FocalNet-DINO/config/DINO/DINO_5scale_focalnet_large_fl4.py'
checkpoint_path = 'checkpoints/detector/focalnet_large_fl4_o365_finetuned_on_coco.pth'
image_dir = 'data/val2017'
output_path = 'prompts/prompts_coco.json'
# Official FocalNet-DINO config uses num_select=300 for evaluation
max_detections = 300
score_threshold = 0.0
batch_size = 1
save_interval = 500

class RandomResize:
    def __init__(self, sizes, max_size=None):
        self.sizes = sizes
        self.max_size = max_size
    
    def __call__(self, img):
        size = self.sizes[0]
        w, h = img.size
        if self.max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > self.max_size:
                size = int(round(self.max_size * min_original_size / max_original_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return img.resize((ow, oh), Image.BILINEAR)

transform = T.Compose([
    RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load model
print('Building model...')
cfg = SLConfig.fromfile(config_path)
cfg_dict = cfg._cfg_dict.to_dict()

class Args:
    pass

model_args = Args()
for k, v in cfg_dict.items():
    setattr(model_args, k, v)

model_args.device = 'cuda'
model_args.masks = False
model_args.fix_size = False
model_args.skip_backbone_pretrain = True

build_func = MODULE_BUILD_FUNCS.get(model_args.modelname)
model, criterion, postprocessors = build_func(model_args)

checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

# Remove 'module.' prefix
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict, strict=False)
model = model.cuda().eval()
print('Model loaded!')

# Get images
images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
print(f'Found {len(images)} images')

# Load existing results if any
existing_results = []
existing_image_ids = set()
if os.path.exists(output_path):
    with open(output_path, 'r') as f:
        existing_results = json.load(f)
        existing_image_ids = set(r['image_id'] for r in existing_results)
    print(f'Loaded {len(existing_results)} existing results from {output_path}')

# Filter to unprocessed images
images_to_process = []
for img_name in images:
    try:
        image_id = int(os.path.splitext(img_name)[0])
        if image_id not in existing_image_ids:
            images_to_process.append((image_id, img_name))
    except ValueError:
        print(f"Skipping non-integer filename: {img_name}")

# Apply max_images limit
if parsed_args.max_images is not None:
    total_needed = parsed_args.max_images - len(existing_image_ids)
    if total_needed <= 0:
        print(f'Already have {len(existing_image_ids)} images processed. Target is {parsed_args.max_images}. Nothing to do.')
        sys.exit(0)
    images_to_process = images_to_process[:total_needed]

print(f'Processing {len(images_to_process)} new images (skipping {len(existing_image_ids)} existing)')

results = existing_results.copy()

# Process images
with torch.no_grad():
    for idx, (image_id, img_name) in enumerate(tqdm(images_to_process)):
        try:
            img_path = os.path.join(image_dir, img_name)
            
            img_pil = Image.open(img_path)
            img = ImageOps.exif_transpose(img_pil).convert('RGB')
            orig_w, orig_h = img.size
            
            img_tensor = transform(img).unsqueeze(0).cuda()
            
            outputs = model(img_tensor)
            
            prob = outputs['pred_logits'].sigmoid()
            boxes = outputs['pred_boxes']
            
            scores, labels = prob[0].max(-1)
            
            boxes = boxes[0]
            boxes_coco = boxes.clone()
            boxes_coco[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * orig_w
            boxes_coco[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * orig_h
            boxes_coco[:, 2] = boxes[:, 2] * orig_w
            boxes_coco[:, 3] = boxes[:, 3] * orig_h
            
            valid = scores > score_threshold
            scores = scores[valid]
            labels = labels[valid]
            boxes_coco = boxes_coco[valid]
            
            sorted_idx = scores.argsort(descending=True)[:max_detections]
            
            for idx_i in sorted_idx:
                # Use label directly as category_id
                real_category_id = int(labels[idx_i].item())

                results.append({
                    'image_id': image_id,
                    'category_id': real_category_id,
                    'bbox': boxes_coco[idx_i].tolist(),
                    'score': scores[idx_i].item()
                })
            
            # Intermediate save
            if (idx + 1) % save_interval == 0:
                with open(output_path, 'w') as f:
                    json.dump(results, f)
                print(f'\n  Saved {len(results)} total results')
                
        except Exception as e:
            print(f'Error processing image {image_id}: {e}')
            continue

# Final save
with open(output_path, 'w') as f:
    json.dump(results, f)

print(f'\n' + '='*60)
print(f'Completed!')
print(f'='*60)
print(f'Total results saved: {len(results)}')
unique_ids_count = len(set(r["image_id"] for r in results))
print(f'Unique images: {unique_ids_count}')
if unique_ids_count > 0:
    print(f'Avg prompts per image: {len(results)/unique_ids_count:.1f}')

if len(results) > 0:
    scores = [d['score'] for d in results]
    print(f'\nScore distribution:')
    print(f'  Min: {min(scores):.3f}')
    print(f'  Max: {max(scores):.3f}')
    print(f'  Mean: {sum(scores)/len(scores):.3f}')
    print(f'  High confidence (>0.5): {len([s for s in scores if s > 0.5])}')

print(f'\nSaved to: {output_path}')