# SAM Evaluation & Pruning (sam_eval)

A pipeline for generating prompts with FocalNet-DINO and ViTDet-H, and evaluating SAM and pruned SAM models on COCO/LVIS datasets.

## Directory Overview

```
sam_eval/
├── _utils/                       # Pruning utilities (sam_pruning.py, wrappers.py)
├── scripts/
│   ├── generate_all_prompts.py   # COCO prompt generation (FocalNet-DINO)
│   ├── generate_lvis_prompts.py  # LVIS prompt generation (ViTDet-H)
│   ├── coco_evaluation_pruning.py# COCO evaluation (with pruning support)
│   └── lvis_evaluation_pruning.py# LVIS evaluation (with pruning support)
├── prompts/                      # Generated prompts
│   ├── prompts_coco.json
│   └── prompts_lvis.json
├── checkpoints/                  # SAM and detector weights
├── annotations/                  # COCO/LVIS annotations
├── run_prompt_generation.sh      # COCO prompt generation wrapper
├── run_prompt_generation_lvis.sh # LVIS prompt generation wrapper
├── run_pruning_evaluation.sh     # Batch pruning evaluation script
└── README.md
```

## Installation

### Step 1: Environment Setup

```bash
# Create and activate conda environment
conda create -n sam_eval python=3.9
conda activate sam_eval

# Install PyTorch (CUDA 12.1)
pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121

# Install Python dependencies
pip install -r requirements.txt

# Install Detectron2
pip install --no-build-isolation \
  detectron2@git+https://github.com/facebookresearch/detectron2.git@fd27788985af0f4ca800bca563acdb700bb890e2

# Install Detrex
pip install --no-build-isolation \
  detrex@git+https://github.com/IDEA-Research/detrex.git@e244e6c3da3e84566728c52c21fb061d23ce0e2f

# Install Segment Anything
pip install \
  segment_anything@git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf
```

**Note:** Install PyTorch before `pip install -r requirements.txt` to avoid build issues with meta repos.

### Step 2: Dataset Preparation

Use the automated dataset preparation script to download and organize all required data:

```bash
python prepare_dataset.py
```

This will automatically:
1. Create directory structure (`data/`, `annotations/`, `checkpoints/`, `prompts/`)
2. Download annotations (minimal)
  - COCO: `instances_val2017.json`
  - LVIS: `lvis_v1_val.json`
3. Download SAM checkpoints (vit_h, vit_l, vit_b) → `checkpoints/sam/`
4. Download detector checkpoints → `checkpoints/detector/`
  - FocalNet-DINO weights (COCO)
  - ViTDet-H Cascade weights (LVIS)
  - Clone FocalNet-DINO repo (for COCO prompt generation)
  - Note: ViTDet config files are already in the repository
5. Download COCO images (train2017 + val2017, ~25GB) → `data/`
6. Create LVIS validation subset (`val2017_lvis_ver/` with 19,809 images)

**Options:**
- `--skip-images`: Skip image download (useful if you already have COCO images)
- `--base-dir`: Specify custom base directory (default: current directory)

```bash
# Skip image download (if you already have them)
python prepare_dataset.py --skip-images

# Use custom directory
python prepare_dataset.py --base-dir /path/to/project
```

**Manual Setup (Alternative):**

If you prefer manual setup or the script fails, follow these steps:

1. **Create directories:**
   ```bash
   mkdir -p data/{train2017,val2017,val2017_lvis_ver}
   mkdir -p annotations checkpoints/{sam,detector} prompts
   ```

2. **Download COCO images and annotations:**
   ```bash
   cd data
   wget http://images.cocodataset.org/zips/train2017.zip
   wget http://images.cocodataset.org/zips/val2017.zip
   unzip train2017.zip && unzip val2017.zip
   cd ../annotations
   wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   unzip annotations_trainval2017.zip
   ```

3. **Download LVIS annotations:**
   ```bash
   cd annotations
   wget https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json
   wget https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json
   ```

4. **Download SAM checkpoints:**
   ```bash
   cd checkpoints/sam
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
   ```

5. **Download detector checkpoints:**
   ```bash
   cd ../detector
   # FocalNet-DINO (COCO)
   wget https://projects4jw.blob.core.windows.net/focalnet/release/detection/focalnet_large_fl4_o365_finetuned_on_coco.pth
   
   # ViTDet-H Cascade (LVIS)
   wget https://dl.fbaipublicfiles.com/detectron2/LVIS-InstanceSegmentation/cascade_mask_rcnn_vitdet_h/194405522/model_final_11bbb7.pkl
   ```

6. **Create LVIS validation subset:**
   ```bash
   # This copies images referenced in lvis_v1_val.json from train2017 and val2017
   python -c "
   import json, shutil
   from pathlib import Path
   
   # Load LVIS annotations
   with open('annotations/lvis_v1_val.json') as f:
       data = json.load(f)
   
   # Get image info
   for img in data['images']:
       src_dir = 'data/train2017' if 'train2017' in img['coco_url'] else 'data/val2017'
       src = Path(src_dir) / img['file_name']
       dst = Path('data/val2017_lvis_ver') / img['file_name']
       dst.parent.mkdir(exist_ok=True)
       if src.exists():
           shutil.copy2(src, dst)
   "
   ```

## Prompt Generation

### COCO (FocalNet-DINO)
```bash
./run_prompt_generation.sh
```
- Input: data/val2017
- Output: prompts/prompts_coco.json
- Default GPU: CUDA_VISIBLE_DEVICES=1

### LVIS (ViTDet-H)
```bash
./run_prompt_generation_lvis.sh
```
- Input: data/val2017_lvis_ver
- Output: prompts/prompts_lvis.json
- Default GPU: CUDA_VISIBLE_DEVICES=1

## Pruning Evaluation (Batch Mode)

```bash
./run_pruning_evaluation.sh
```

This script runs multiple evaluation configurations defined in `test_list`. Edit the script to customize:
- Dataset (coco, lvis)
- Model size (h: vit_h, l: vit_l, b: vit_b)
- Number of images (e.g., 50, 100, -1 for all)
- Pruning method (none, sp, unstructured)
- Pruning ratios (MLP and Attention)
- Batch size (number of images to process in parallel)

### Batch Processing

Edit `BATCH_SIZE` in `run_pruning_evaluation.sh`:
```bash
BATCH_SIZE=4  # Number of images to process in parallel
```

## Manual Evaluation

### COCO
```bash
python3 scripts/coco_evaluation_pruning.py \
  --sam-checkpoint checkpoints/sam/sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --coco-annotation annotations/instances_val2017.json \
  --prompt-file prompts/prompts_coco.json \
  --image-dir data/val2017 \
  --max-images 50 \
  --batch-size 4 \
  --sparsity-type sp \
  --prune-mlp-ratio 0.5 \
  --prune-attn-ratio 0.0 \
  --output-dir eval_results_pruning/coco_vit_h_sp_m0.5_a0.0_n50
```

### LVIS
```bash
python3 scripts/lvis_evaluation_pruning.py \
  --sam-checkpoint checkpoints/sam/sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --lvis-annotation annotations/lvis_v1_val.json \
  --prompt-file prompts/prompts_lvis.json \
  --image-dir data/val2017_lvis_ver \
  --max-images 50 \
  --batch-size 4 \
  --sparsity-type unstructured \
  --prune-mlp-ratio 0.3 \
  --prune-attn-ratio 0.3 \
  --output-dir eval_results_pruning/lvis_vit_h_unst_m0.3_a0.3_n50
```

## Key Parameters

### Pruning Parameters
- `--sparsity-type`: Pruning method (none | sp | unstructured)
  - `none`: Baseline (no pruning)
  - `sp`: Structured pruning (removes entire neurons)
  - `unstructured`: Unstructured pruning (zeros individual weights)
- `--prune-mlp-ratio`: MLP pruning ratio (0.0-1.0)
- `--prune-attn-ratio`: Attention pruning ratio (0.0-1.0)

### Evaluation Parameters
- `--max-images`: Number of images to evaluate (-1 for all)
- `--max-detections`: Max detections per image (default: 300 for LVIS, 100 for COCO)
- `--batch-size`: Number of images to process in parallel (default: 1)
- `--score-threshold`: Detection score threshold (default: 0.0)

### Model Parameters
- `--model-type`: SAM model size (vit_h, vit_l, vit_b)
- `--device`: Compute device (default: cuda)
- `--rebuild-freq`: How often to rebuild pruning masks (default: 1)
- `--act-mode`: Activation mode for importance calculation (default: rms)

## Required Directory Structure

After running `python prepare_dataset.py`, your directory structure should look like:

```
sam_eval/
├── data/
│   ├── train2017/                # COCO training images (~118K images)
│   ├── val2017/                  # COCO validation images (~5K images)
│   └── val2017_lvis_ver/         # LVIS validation subset (19,809 images)
│       ├── (15,000 from train2017)
│       └── (4,809 from val2017)
├── annotations/
│   ├── instances_val2017.json    # COCO validation annotations (downloaded)
│   └── lvis_v1_val.json          # LVIS validation annotations (downloaded)
├── checkpoints/
│   ├── sam/
│   │   ├── sam_vit_h_4b8939.pth  # SAM ViT-H (huge)
│   │   ├── sam_vit_l_0b3195.pth  # SAM ViT-L (large)
│   │   └── sam_vit_b_01ec64.pth  # SAM ViT-B (base)
│   └── detector/
│       ├── focalnet_large_fl4_o365_finetuned_on_coco.pth  # (downloaded)
│       ├── model_final_11bbb7.pkl                         # (downloaded)
│       ├── FocalNet-DINO/        # Cloned (ignored by git)
│       └── detectron2_configs/   # ✓ Tracked in git
├── prompts/                      # Generated after running prompt scripts
│   ├── prompts_coco.json
│   └── prompts_lvis.json
└── eval_results_pruning/         # Created during evaluation
```

**Note:** The `prepare_dataset.py` script automatically creates this structure.

**Git Repository:** Only `checkpoints/detector/detectron2_configs/` is tracked in git (ViTDet config files).
All other checkpoints and data files are ignored by `.gitignore`:
- `checkpoints/sam/` - SAM model weights (downloaded by script)
- `checkpoints/detector/*.pth` - Detector weights (downloaded by script)
- `checkpoints/detector/FocalNet-DINO/` - FocalNet code and configs (optional)

## Output

Results are saved to `eval_results_pruning/` with the following structure:
```
eval_results_pruning/
└── {dataset}_{model}_{pruning_config}_n{num_images}/
    └── results_{...}.json
```

Example output JSON:
```json
{
  "AP": 64.05,
  "AP50": 85.23,
  "AP75": 70.12,
  "APs": 45.67,
  "APm": 68.90,
  "APl": 78.34,
  "num_predictions": 15432,
  "num_images": 50
}
```

## Metrics

### COCO Metrics
- **AP**: Average Precision @ IoU=0.50:0.95
- **AP50**: Average Precision @ IoU=0.50
- **AP75**: Average Precision @ IoU=0.75
- **APs**: AP for small objects
- **APm**: AP for medium objects
- **APl**: AP for large objects

### LVIS Metrics
Includes COCO metrics plus:
- **APr**: AP for rare objects
- **APc**: AP for common objects
- **APf**: AP for frequent objects

## Notes

- LVIS evaluation on NumPy 1.20+ environments may encounter `np.float` deprecation issues. If this occurs, replace `np.float` with `np.float64` in `lvis/eval.py`:
  ```bash
  sed -i 's/astype(dtype=np\.float)/astype(dtype=np.float64)/g' ~/miniconda3/envs/sam_eval/lib/python3.9/site-packages/lvis/eval.py
  ```

- Batch size determines how many images are processed in each batch. Larger batch sizes are faster but require more GPU memory.

- Results are cached in `eval_results_pruning/`. Remove directories to rerun evaluations.

## References

- **SAM**: https://github.com/facebookresearch/segment-anything
- **FocalNet-DINO**: https://github.com/clin1223/FocalNet-DINO
- **Detectron2**: https://github.com/facebookresearch/detectron2
- **LVIS**: https://www.lvisdataset.org/
