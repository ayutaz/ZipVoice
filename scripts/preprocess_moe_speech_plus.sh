#!/bin/bash
# Preprocessing script for moe-speech-plus dataset (473 speakers)
# Dataset: https://huggingface.co/datasets/ayousanz/moe-speech-plus

set -e

DATA_DIR="/data/moe-speech-plus"
OUTPUT_DIR="/data/ZipVoice-plus/data"
NUM_JOBS=16

echo "=== Step 1: Extract ZIP files and create TSV ==="
uv run python scripts/preprocess_moe_speech_plus.py \
    --input-dir "$DATA_DIR" \
    --extract-dir "$DATA_DIR/extracted" \
    --output-dir data/raw \
    --dev-ratio 0.05 \
    --num-jobs $NUM_JOBS

echo "=== Step 2: Create manifests (includes resampling to 24kHz) ==="
uv run python -m zipvoice.bin.prepare_dataset \
    --tsv-path data/raw/moe_speech_plus_train.tsv \
    --prefix moe_speech_plus \
    --subset train \
    --num-jobs $NUM_JOBS \
    --output-dir data/manifests

uv run python -m zipvoice.bin.prepare_dataset \
    --tsv-path data/raw/moe_speech_plus_dev.tsv \
    --prefix moe_speech_plus \
    --subset dev \
    --num-jobs $NUM_JOBS \
    --output-dir data/manifests

echo "=== Step 3: Compute fbank features ==="
uv run python -m zipvoice.bin.compute_fbank \
    --source-dir data/manifests \
    --dest-dir data/fbank \
    --dataset moe_speech_plus \
    --subset train \
    --num-jobs $NUM_JOBS

uv run python -m zipvoice.bin.compute_fbank \
    --source-dir data/manifests \
    --dest-dir data/fbank \
    --dataset moe_speech_plus \
    --subset dev \
    --num-jobs $NUM_JOBS

echo "=== Step 4: Pretokenize with Japanese tokenizer ==="
uv run python -m zipvoice.bin.pretokenize_manifest \
    --input-manifest data/fbank/moe_speech_plus_cuts_train.jsonl.gz \
    --output-manifest data/fbank/moe_speech_plus_cuts_train_tokenized.jsonl.gz \
    --token-file data/tokens_japanese_extended.txt \
    --use-accent \
    --num-workers $NUM_JOBS

uv run python -m zipvoice.bin.pretokenize_manifest \
    --input-manifest data/fbank/moe_speech_plus_cuts_dev.jsonl.gz \
    --output-manifest data/fbank/moe_speech_plus_cuts_dev_tokenized.jsonl.gz \
    --token-file data/tokens_japanese_extended.txt \
    --use-accent \
    --num-workers $NUM_JOBS

echo "=== Preprocessing complete! ==="
echo ""
echo "Train manifest: data/fbank/moe_speech_plus_cuts_train_tokenized.jsonl.gz"
echo "Dev manifest: data/fbank/moe_speech_plus_cuts_dev_tokenized.jsonl.gz"
echo "Token file: data/tokens_japanese_extended.txt"
echo ""
echo "Next step: Run training with:"
echo "  uv run python -m zipvoice.bin.train_zipvoice \\"
echo "    --world-size 1 --use-fp16 1 --num-epochs 20 \\"
echo "    --finetune 1 --checkpoint download/zipvoice/model.pt \\"
echo "    --tokenizer japanese --token-file data/tokens_japanese_extended.txt \\"
echo "    --dataset moe_speech_plus --manifest-dir data/fbank \\"
echo "    --base-lr 0.01 --max-duration 100 \\"
echo "    --exp-dir exp/zipvoice_japanese_473speakers"
