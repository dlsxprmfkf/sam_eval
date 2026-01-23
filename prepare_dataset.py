#!/usr/bin/env python3
"""
SAM Evaluation Dataset Preparation Script
Downloads and organizes all required datasets, annotations, and model checkpoints.
"""

import os
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Set
import argparse


class DatasetPreparer:
    """Handles dataset and checkpoint preparation for SAM evaluation."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.annotations_dir = self.base_dir / "annotations"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.prompts_dir = self.base_dir / "prompts"
        self.eval_results_dir = self.base_dir / "eval_results_pruning"
        
    def create_directories(self):
        """Create necessary directory structure."""
        print("Creating directory structure...")
        dirs = [
            self.data_dir,
            self.data_dir / "train2017",
            self.data_dir / "val2017",
            self.data_dir / "val2017_lvis_ver",
            self.annotations_dir,
            self.checkpoints_dir / "sam",
            self.checkpoints_dir / "detector",
            self.prompts_dir,
            self.eval_results_dir,
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {dir_path.relative_to(self.base_dir)}")
    
    def download_file(self, url: str, dest_path: Path, desc: str = ""):
        """Download a file with progress indication."""
        if dest_path.exists():
            print(f"  ⊙ {desc} already exists, skipping download")
            return
        
        print(f"  ⬇ Downloading {desc}...")
        print(f"    URL: {url}")
        
        try:
            def reporthook(count, block_size, total_size):
                percent = min(int(count * block_size * 100 / total_size), 100)
                print(f"\r    Progress: {percent}%", end='', flush=True)
            
            urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
            print(f"\n  ✓ Downloaded to {dest_path.relative_to(self.base_dir)}")
        except Exception as e:
            print(f"\n  ✗ Failed to download {desc}: {e}")
            if dest_path.exists():
                dest_path.unlink()
            raise
    
    def extract_zip(self, zip_path: Path, extract_to: Path, desc: str = ""):
        """Extract a zip file."""
        print(f"  📦 Extracting {desc}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"  ✓ Extracted to {extract_to.relative_to(self.base_dir)}")
    
    def download_annotations(self):
        """Download COCO and LVIS annotations."""
        print("\n[1/5] Downloading Annotations...")
        
        # COCO annotations
        coco_anno_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        coco_anno_zip = self.annotations_dir / "annotations_trainval2017.zip"
        self.download_file(coco_anno_url, coco_anno_zip, "COCO annotations")
        
        if not (self.annotations_dir / "instances_val2017.json").exists():
            self.extract_zip(coco_anno_zip, self.annotations_dir.parent, "COCO annotations")
        
        # LVIS annotations
        lvis_urls = {
            "lvis_v1_val.json": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json",
            "lvis_v1_train.json": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json",
        }
        
        for filename, url in lvis_urls.items():
            dest = self.annotations_dir / filename
            self.download_file(url, dest, f"LVIS {filename}")
    
    def download_sam_checkpoints(self):
        """Download SAM model checkpoints."""
        print("\n[2/5] Downloading SAM Checkpoints...")
        
        sam_models = {
            "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "sam_vit_l_0b3195.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "sam_vit_b_01ec64.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        }
        
        for filename, url in sam_models.items():
            dest = self.checkpoints_dir / "sam" / filename
            self.download_file(url, dest, f"SAM {filename}")
    
    def download_detector_checkpoints(self):
        """Download detector model checkpoints."""
        print("\n[3/5] Downloading Detector Checkpoints...")
        
        # FocalNet-DINO (COCO)
        focalnet_url = "https://projects4jw.blob.core.windows.net/focalnet/release/detection/focalnet_large_fl4_o365_finetuned_on_coco.pth"
        focalnet_dest = self.checkpoints_dir / "detector" / "focalnet_large_fl4_o365_finetuned_on_coco.pth"
        self.download_file(focalnet_url, focalnet_dest, "FocalNet-DINO (COCO)")
        
        # ViTDet-H Cascade (LVIS)
        vitdet_url = "https://dl.fbaipublicfiles.com/detectron2/LVIS-InstanceSegmentation/cascade_mask_rcnn_vitdet_h/194405522/model_final_11bbb7.pkl"
        vitdet_dest = self.checkpoints_dir / "detector" / "model_final_11bbb7.pkl"
        self.download_file(vitdet_url, vitdet_dest, "ViTDet-H Cascade (LVIS)")
        
        print("\n  ℹ Config files are already included in the repository:")
        print(f"    - FocalNet-DINO: checkpoints/detector/FocalNet-DINO/")
        print(f"    - ViTDet: checkpoints/detector/detectron2_configs/")
    
    def download_coco_images(self):
        """Download COCO train2017 and val2017 images."""
        print("\n[4/5] Downloading COCO Images...")
        print("  This will take significant time and disk space (~25GB)...")
        
        coco_images = {
            "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
            "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
        }
        
        for filename, url in coco_images.items():
            zip_path = self.data_dir / filename
            self.download_file(url, zip_path, f"COCO {filename}")
            
            # Extract if not already extracted
            image_dir = self.data_dir / filename.replace('.zip', '')
            if not image_dir.exists() or len(list(image_dir.glob('*.jpg'))) == 0:
                self.extract_zip(zip_path, self.data_dir, f"COCO {filename}")
    
    def create_lvis_val_subset(self):
        """Create val2017_lvis_ver with images used in LVIS validation."""
        print("\n[5/5] Creating LVIS Validation Image Subset...")
        
        lvis_anno_path = self.annotations_dir / "lvis_v1_val.json"
        if not lvis_anno_path.exists():
            print("  ✗ LVIS annotation file not found. Run annotation download first.")
            return
        
        # Load LVIS annotations
        print("  📖 Loading LVIS v1 validation annotations...")
        with open(lvis_anno_path, 'r') as f:
            lvis_data = json.load(f)
        
        # Get image IDs and filenames
        image_info = {}
        for img in lvis_data['images']:
            image_info[img['id']] = {
                'file_name': img['file_name'],
                'coco_url': img.get('coco_url', '')
            }
        
        print(f"  ℹ Total LVIS validation images: {len(image_info)}")
        
        # Separate images by source (train2017 vs val2017)
        train_images = []
        val_images = []
        
        for img_id, info in image_info.items():
            if 'train2017' in info['coco_url']:
                train_images.append(info['file_name'])
            else:  # val2017
                val_images.append(info['file_name'])
        
        print(f"  ℹ From train2017: {len(train_images)} images")
        print(f"  ℹ From val2017: {len(val_images)} images")
        
        # Copy images to val2017_lvis_ver
        dest_dir = self.data_dir / "val2017_lvis_ver"
        dest_dir.mkdir(exist_ok=True)
        
        copied_count = 0
        skipped_count = 0
        missing_count = 0
        
        print("  📁 Copying images from train2017...")
        for filename in train_images:
            src = self.data_dir / "train2017" / filename
            dst = dest_dir / filename
            
            if dst.exists():
                skipped_count += 1
                continue
            
            if src.exists():
                shutil.copy2(src, dst)
                copied_count += 1
                if copied_count % 1000 == 0:
                    print(f"    Progress: {copied_count}/{len(train_images)} from train2017")
            else:
                missing_count += 1
        
        print("  📁 Copying images from val2017...")
        train_copied = copied_count
        for filename in val_images:
            src = self.data_dir / "val2017" / filename
            dst = dest_dir / filename
            
            if dst.exists():
                skipped_count += 1
                continue
            
            if src.exists():
                shutil.copy2(src, dst)
                copied_count += 1
                if (copied_count - train_copied) % 1000 == 0:
                    print(f"    Progress: {copied_count - train_copied}/{len(val_images)} from val2017")
            else:
                missing_count += 1
        
        print(f"\n  ✓ LVIS validation subset created:")
        print(f"    - Copied: {copied_count} images")
        print(f"    - Skipped (already exist): {skipped_count} images")
        print(f"    - Missing: {missing_count} images")
        print(f"    - Total in {dest_dir.relative_to(self.base_dir)}: {len(list(dest_dir.glob('*.jpg')))} images")
    
    def run_all(self, skip_images: bool = False):
        """Run all preparation steps."""
        print("="*60)
        print("  SAM Evaluation Dataset Preparation")
        print("="*60)
        
        self.create_directories()
        self.download_annotations()
        self.download_sam_checkpoints()
        self.download_detector_checkpoints()
        
        if not skip_images:
            self.download_coco_images()
            self.create_lvis_val_subset()
        else:
            print("\n[4/5] Skipping COCO image download (--skip-images)")
            print("[5/5] Skipping LVIS subset creation (--skip-images)")
        
        print("\n" + "="*60)
        print("  ✓ Dataset Preparation Complete!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Generate prompts: ./run_prompt_generation_coco.sh")
        print("  2. Generate prompts: ./run_prompt_generation_lvis.sh")
        print("  3. Run evaluation: ./run_pruning_evaluation.sh")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets and checkpoints for SAM evaluation"
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip COCO image download (useful for testing or when images already exist)"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory for the project (default: current directory)"
    )
    
    args = parser.parse_args()
    
    preparer = DatasetPreparer(base_dir=args.base_dir)
    preparer.run_all(skip_images=args.skip_images)


if __name__ == "__main__":
    main()
