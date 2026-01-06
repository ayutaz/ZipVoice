#!/bin/bash
# Preprocessing script for Tsukuyomi-chan dataset
# Dataset location: /data/tsukuyomi-chan-ljspeech/

set -e

DATA_DIR="/data/tsukuyomi-chan-ljspeech"
OUTPUT_DIR="/data/ZipVoice-plus/data"
NUM_JOBS=4

echo "=== Step 1: Convert metadata.csv to TSV format ==="
uv run python3 << 'EOF'
import os

data_dir = "/data/tsukuyomi-chan-ljspeech"
output_dir = "/data/ZipVoice-plus/data/raw"
os.makedirs(output_dir, exist_ok=True)

# Read metadata.csv (LJSpeech format: id|text)
entries = []
with open(f"{data_dir}/metadata.csv", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) >= 2:
            utt_id = parts[0]
            text = parts[1]
            wav_path = f"{data_dir}/wavs/{utt_id}.wav"
            # Verify file exists
            if os.path.exists(wav_path):
                entries.append((utt_id, text, wav_path))

print(f"Total entries: {len(entries)}")

# Split into train/dev (90%/10%)
dev_count = max(1, int(len(entries) * 0.1))
dev_entries = entries[:dev_count]
train_entries = entries[dev_count:]

# Write TSV files
with open(f"{output_dir}/tsukuyomi_train.tsv", "w", encoding="utf-8") as f:
    for utt_id, text, wav_path in train_entries:
        f.write(f"{utt_id}\t{text}\t{wav_path}\n")

with open(f"{output_dir}/tsukuyomi_dev.tsv", "w", encoding="utf-8") as f:
    for utt_id, text, wav_path in dev_entries:
        f.write(f"{utt_id}\t{text}\t{wav_path}\n")

print(f"Train: {len(train_entries)}, Dev: {len(dev_entries)}")
EOF

echo "=== Step 2: Create manifests (includes resampling to 24kHz) ==="
uv run python -m zipvoice.bin.prepare_dataset \
    --tsv-path data/raw/tsukuyomi_train.tsv \
    --prefix tsukuyomi \
    --subset train \
    --num-jobs $NUM_JOBS \
    --output-dir data/manifests

uv run python -m zipvoice.bin.prepare_dataset \
    --tsv-path data/raw/tsukuyomi_dev.tsv \
    --prefix tsukuyomi \
    --subset dev \
    --num-jobs $NUM_JOBS \
    --output-dir data/manifests

echo "=== Step 3: Compute fbank features ==="
uv run python -m zipvoice.bin.compute_fbank \
    --source-dir data/manifests \
    --dest-dir data/fbank \
    --dataset tsukuyomi \
    --subset train \
    --num-jobs $NUM_JOBS

uv run python -m zipvoice.bin.compute_fbank \
    --source-dir data/manifests \
    --dest-dir data/fbank \
    --dataset tsukuyomi \
    --subset dev \
    --num-jobs $NUM_JOBS

echo "=== Step 4: Pretokenize with accent markers ==="
uv run python -m zipvoice.bin.pretokenize_manifest \
    --input-manifest data/fbank/tsukuyomi_cuts_train.jsonl.gz \
    --output-manifest data/fbank/tsukuyomi_cuts_train_tokenized.jsonl.gz \
    --token-file data/tokens_japanese_extended.txt \
    --use-accent \
    --num-workers $NUM_JOBS

uv run python -m zipvoice.bin.pretokenize_manifest \
    --input-manifest data/fbank/tsukuyomi_cuts_dev.jsonl.gz \
    --output-manifest data/fbank/tsukuyomi_cuts_dev_tokenized.jsonl.gz \
    --token-file data/tokens_japanese_extended.txt \
    --use-accent \
    --num-workers $NUM_JOBS

echo "=== Preprocessing complete! ==="
echo "Train manifest: data/fbank/tsukuyomi_cuts_train_tokenized.jsonl.gz"
echo "Dev manifest: data/fbank/tsukuyomi_cuts_dev_tokenized.jsonl.gz"
echo "Token file: data/tokens_japanese_extended.txt"
