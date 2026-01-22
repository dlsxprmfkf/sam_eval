#!/bin/bash

# ==============================================================================
# SAM Pruning Evaluation Script
# ==============================================================================
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=1

# ==============================================================================
# TEST CONFIGURATION LIST
# Format: "dataset model max_images sparsity_type mlp_ratio attn_ratio"
# 
# Parameters:
#   dataset: coco, lvis
#   model: h (vit_h), l (vit_l), b (vit_b)
#   max_images: number of images to evaluate (-1 for all, e.g., 50, 100)
#   sparsity_type: none, sp (structured), unstructured
#   mlp_ratio: MLP pruning ratio (0.0-1.0, e.g., 0.0, 0.3, 0.5, 0.7)
#   attn_ratio: Attention pruning ratio (0.0-1.0, e.g., 0.0, 0.3, 0.5)
# ==============================================================================
test_list=(
    # Baseline (no pruning)
    # "lvis h 50 none 0.0 0.0"
    # "lvis h 50 sp 0.2 0.0"
    "lvis h 50 sp 0.4 0.0"    
)

# ==============================================================================
# GLOBAL SETTINGS
# ==============================================================================
DEVICE="cuda"
REBUILD_FREQ=1
ACT_MODE="rms"
SCORE_THRESHOLD=0.0
BATCH_SIZE=10  # Number of images to process in parallel

# Paths
COCO_ANNOTATION="annotations/instances_val2017.json"
LVIS_ANNOTATION="annotations/lvis_v1_val.json"
# Images under sam_eval/data
COCO_IMAGE_DIR="data/val2017"
LVIS_IMAGE_DIR="data/val2017_lvis_ver"  # LVIS uses COCO images (lvis split)
COCO_PROMPT_FILE="prompts/prompts_coco.json"
LVIS_PROMPT_FILE="prompts/prompts_lvis.json"

# Checkpoints
CHECKPOINT_DIR="checkpoints/sam"
VIT_H_CHECKPOINT="${CHECKPOINT_DIR}/sam_vit_h_4b8939.pth"
VIT_L_CHECKPOINT="${CHECKPOINT_DIR}/sam_vit_l_0b3195.pth"
VIT_B_CHECKPOINT="${CHECKPOINT_DIR}/sam_vit_b_01ec64.pth"

# ==============================================================================
# TEST EXECUTION LOOP
# ==============================================================================
echo "======================================================================"
echo "Starting SAM Pruning Evaluation"
echo "======================================================================"

for test_case in "${test_list[@]}"; do
    read -r DATASET MODEL_FLAG MAX_IMAGES SPARSITY_TYPE MLP_RATIO ATTN_RATIO <<< "$test_case"

    echo ""
    echo "======================================================================"
    echo "▶️  TEST CASE: $test_case"
    echo "======================================================================"

    # Set model type and checkpoint path
    case "$MODEL_FLAG" in
        h) MODEL_TYPE="vit_h"; CHECKPOINT="$VIT_H_CHECKPOINT"; MODEL_SUFFIX="_vit_h" ;;
        l) MODEL_TYPE="vit_l"; CHECKPOINT="$VIT_L_CHECKPOINT"; MODEL_SUFFIX="_vit_l" ;;
        b) MODEL_TYPE="vit_b"; CHECKPOINT="$VIT_B_CHECKPOINT"; MODEL_SUFFIX="_vit_b" ;;
        *) echo "ERROR: Invalid model flag '$MODEL_FLAG'. Skipping."; continue ;;
    esac

    # Check if checkpoint exists
    if [ ! -f "$CHECKPOINT" ]; then
        echo "ERROR: Checkpoint not found: $CHECKPOINT"
        echo "Skipping this test case."
        continue
    fi

    # Set dataset paths
    case "$DATASET" in
        coco)
            ANNOTATION="$COCO_ANNOTATION"
            IMAGE_DIR="$COCO_IMAGE_DIR"
            PROMPT_FILE="$COCO_PROMPT_FILE"
            EVAL_SCRIPT="scripts/coco_evaluation_pruning.py"
            ;;
        lvis)
            ANNOTATION="$LVIS_ANNOTATION"
            IMAGE_DIR="$LVIS_IMAGE_DIR"
            PROMPT_FILE="$LVIS_PROMPT_FILE"
            EVAL_SCRIPT="scripts/lvis_evaluation_pruning.py"
            ;;
        *) echo "ERROR: Invalid dataset '$DATASET'. Skipping."; continue ;;
    esac

    # Check if required files exist
    if [ ! -f "$ANNOTATION" ]; then
        echo "ERROR: Annotation file not found: $ANNOTATION"
        continue
    fi
    if [ ! -f "$PROMPT_FILE" ]; then
        echo "ERROR: Prompt file not found: $PROMPT_FILE"
        continue
    fi
    if [ ! -d "$IMAGE_DIR" ]; then
        echo "ERROR: Image directory not found: $IMAGE_DIR"
        continue
    fi

    # Generate output suffix
    OUTPUT_SUFFIX="${DATASET}${MODEL_SUFFIX}"
    
    if [ "$SPARSITY_TYPE" != "none" ]; then
        PRUNING_TAG="_${SPARSITY_TYPE}_m${MLP_RATIO}_a${ATTN_RATIO}"
        OUTPUT_SUFFIX="${OUTPUT_SUFFIX}${PRUNING_TAG}"
    fi
    
    OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_n${MAX_IMAGES}"
    
    OUTPUT_DIR="eval_results_pruning/${OUTPUT_SUFFIX}"
    OUTPUT_FILE="results_${OUTPUT_SUFFIX}.json"

    echo "Configuration:"
    echo "  Dataset: $DATASET"
    echo "  Model: $MODEL_TYPE"
    echo "  Checkpoint: $CHECKPOINT"
    echo "  Max Images: $MAX_IMAGES"
    echo "  Sparsity Type: $SPARSITY_TYPE"
    echo "  MLP Ratio: $MLP_RATIO"
    echo "  Attn Ratio: $ATTN_RATIO"
    echo "  Output Dir: $OUTPUT_DIR"
    echo ""

    # Run evaluation
    python3 "$EVAL_SCRIPT" \
        --sam-checkpoint "$CHECKPOINT" \
        --model-type "$MODEL_TYPE" \
        --${DATASET}-annotation "$ANNOTATION" \
        --prompt-file "$PROMPT_FILE" \
        --image-dir "$IMAGE_DIR" \
        --max-images $MAX_IMAGES \
        --score-threshold $SCORE_THRESHOLD \
        --max-detections 300 \
        --batch-size $BATCH_SIZE \
        --output-dir "$OUTPUT_DIR" \
        --output-file "$OUTPUT_FILE" \
        --device "$DEVICE" \
        --sparsity-type "$SPARSITY_TYPE" \
        --prune-mlp-ratio $MLP_RATIO \
        --prune-attn-ratio $ATTN_RATIO \
        --rebuild-freq $REBUILD_FREQ \
        --act-mode "$ACT_MODE"

    if [ $? -eq 0 ]; then
        echo "✅ Test case completed successfully"
    else
        echo "❌ Test case failed"
    fi

done

echo ""
echo "======================================================================"
echo "✅ All test cases finished."
echo "======================================================================"
