#!/bin/bash
# Training script for Japanese TTS in Docker

set -e

# Default values
WORLD_SIZE=${WORLD_SIZE:-1}
BASE_LR=${BASE_LR:-0.0001}
NUM_ITERS=${NUM_ITERS:-10000}
SAVE_EVERY_N=${SAVE_EVERY_N:-1000}
MAX_DURATION=${MAX_DURATION:-60}
USE_FP16=${USE_FP16:-1}
WANDB_PROJECT=${WANDB_PROJECT:-zipvoice-japanese}

echo "=========================================="
echo "ZipVoice Japanese Training"
echo "=========================================="
echo "World size: $WORLD_SIZE"
echo "Base LR: $BASE_LR"
echo "Num iters: $NUM_ITERS"
echo "Save every: $SAVE_EVERY_N"
echo "Max duration: $MAX_DURATION"
echo "Use FP16: $USE_FP16"
echo "=========================================="

# Check if k2 is available
python -c "import k2; print(f'k2 version: {k2.__version__}')" || {
    echo "ERROR: k2 is not installed!"
    exit 1
}

# Run training
python -m zipvoice.bin.train_zipvoice \
    --world-size $WORLD_SIZE \
    --use-fp16 $USE_FP16 \
    --finetune 1 \
    --base-lr $BASE_LR \
    --num-iters $NUM_ITERS \
    --save-every-n $SAVE_EVERY_N \
    --max-duration $MAX_DURATION \
    --drop-last 0 \
    --model-config download/zipvoice/model.json \
    --checkpoint download/zipvoice/model.pt \
    --tokenizer japanese \
    --token-file data/tokens_japanese_extended.txt \
    --dataset custom \
    --train-manifest data/fbank/tsukuyomi_cuts_train.jsonl.gz \
    --dev-manifest data/fbank/tsukuyomi_cuts_dev.jsonl.gz \
    --exp-dir exp/zipvoice_japanese \
    --wandb-project $WANDB_PROJECT

echo "Training completed!"
