#!/bin/bash
# Preprocessing script for MoeSpeech-20speakers dataset
# Dataset: https://huggingface.co/datasets/ayousanz/moe-speech-20speakers-ljspeech

set -e

DATA_DIR="/data/moe-speech-20speakers-ljspeech"
OUTPUT_DIR="/data/ZipVoice-plus/data"
NUM_JOBS=8

echo "=== Step 1: Convert metadata.csv to TSV format ==="
python3 << 'EOF'
import os
import csv

data_dir = "/data/moe-speech-20speakers-ljspeech"
output_dir = "/data/ZipVoice-plus/data/raw"
os.makedirs(output_dir, exist_ok=True)

# Read metadata.csv (LJSpeech format: id|speaker|text)
entries = []
with open(f"{data_dir}/metadata.csv", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) >= 3:
            utt_id = parts[0]
            text = parts[2]
            wav_path = f"{data_dir}/wavs/{utt_id}.wav"
            entries.append((utt_id, text, wav_path))

print(f"Total entries: {len(entries)}")

# Split into train/dev (95%/5%)
dev_count = max(1, int(len(entries) * 0.05))
dev_entries = entries[:dev_count]
train_entries = entries[dev_count:]

# Write TSV files
with open(f"{output_dir}/moe_speech_train.tsv", "w", encoding="utf-8") as f:
    for utt_id, text, wav_path in train_entries:
        f.write(f"{utt_id}\t{text}\t{wav_path}\n")

with open(f"{output_dir}/moe_speech_dev.tsv", "w", encoding="utf-8") as f:
    for utt_id, text, wav_path in dev_entries:
        f.write(f"{utt_id}\t{text}\t{wav_path}\n")

print(f"Train: {len(train_entries)}, Dev: {len(dev_entries)}")
EOF

echo "=== Step 2: Create manifests (includes resampling to 24kHz) ==="
uv run python -m zipvoice.bin.prepare_dataset \
    --tsv-path data/raw/moe_speech_train.tsv \
    --prefix moe_speech \
    --subset train \
    --num-jobs $NUM_JOBS \
    --output-dir data/manifests

uv run python -m zipvoice.bin.prepare_dataset \
    --tsv-path data/raw/moe_speech_dev.tsv \
    --prefix moe_speech \
    --subset dev \
    --num-jobs $NUM_JOBS \
    --output-dir data/manifests

echo "=== Step 3: Compute fbank features ==="
uv run python -m zipvoice.bin.compute_fbank \
    --source-dir data/manifests \
    --dest-dir data/fbank \
    --dataset moe_speech \
    --subset train \
    --num-jobs $NUM_JOBS

uv run python -m zipvoice.bin.compute_fbank \
    --source-dir data/manifests \
    --dest-dir data/fbank \
    --dataset moe_speech \
    --subset dev \
    --num-jobs $NUM_JOBS

echo "=== Step 4: Pretokenize with accent markers ==="
uv run python -m zipvoice.bin.pretokenize_manifest \
    --input-manifest data/fbank/moe_speech_cuts_train.jsonl.gz \
    --output-manifest data/fbank/moe_speech_cuts_train_tokenized.jsonl.gz \
    --token-file data/tokens_japanese_extended.txt \
    --use-accent \
    --num-workers $NUM_JOBS

uv run python -m zipvoice.bin.pretokenize_manifest \
    --input-manifest data/fbank/moe_speech_cuts_dev.jsonl.gz \
    --output-manifest data/fbank/moe_speech_cuts_dev_tokenized.jsonl.gz \
    --token-file data/tokens_japanese_extended.txt \
    --use-accent \
    --num-workers $NUM_JOBS

echo "=== Preprocessing complete! ==="
echo "Train manifest: data/fbank/moe_speech_cuts_train_tokenized.jsonl.gz"
echo "Dev manifest: data/fbank/moe_speech_cuts_dev_tokenized.jsonl.gz"
echo "Token file: data/tokens_japanese_extended.txt"
