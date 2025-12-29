#!/bin/bash
# Preprocessing script for Tsukuyomi-chan dataset
# Run inside Docker container

set -e

echo "=== Step 1: Resample audio to 24kHz ==="
mkdir -p data/raw/tsukuyomi_24k

python -c "
import os
import soundfile as sf
import librosa
from tqdm import tqdm

src_dir = 'data/raw/tsukuyomi_src'
dst_dir = 'data/raw/tsukuyomi_24k'
target_sr = 24000

files = [f for f in os.listdir(src_dir) if f.endswith('.wav')]
print(f'Processing {len(files)} files...')

for fname in tqdm(files, desc='Resampling'):
    src_path = os.path.join(src_dir, fname)
    dst_path = os.path.join(dst_dir, fname)
    audio, sr = librosa.load(src_path, sr=target_sr)
    sf.write(dst_path, audio, target_sr)

print('Done resampling!')
"

echo "=== Step 2: Create train/dev split TSV ==="
python -c "
tsv_in = 'data/raw/tsukuyomi_docker.tsv'
tsv_train = 'data/raw/tsukuyomi_24k_train.tsv'
tsv_dev = 'data/raw/tsukuyomi_24k_dev.tsv'

entries = []
with open(tsv_in, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            utt_id, text = parts[0], parts[1]
            wav_path = f'data/raw/tsukuyomi_24k/{utt_id}.wav'
            entries.append((utt_id, text, wav_path))

dev_count = max(1, int(len(entries) * 0.1))
dev_entries = entries[:dev_count]
train_entries = entries[dev_count:]

with open(tsv_train, 'w', encoding='utf-8') as f:
    for utt_id, text, path in train_entries:
        f.write(f'{utt_id}\t{text}\t{path}\n')

with open(tsv_dev, 'w', encoding='utf-8') as f:
    for utt_id, text, path in dev_entries:
        f.write(f'{utt_id}\t{text}\t{path}\n')

print(f'Train: {len(train_entries)}, Dev: {len(dev_entries)}')
"

echo "=== Step 3: Create manifests ==="
python -m zipvoice.bin.prepare_dataset --tsv-path data/raw/tsukuyomi_24k_train.tsv --prefix tsukuyomi --subset train --num-jobs 4 --output-dir data/manifests

python -m zipvoice.bin.prepare_dataset --tsv-path data/raw/tsukuyomi_24k_dev.tsv --prefix tsukuyomi --subset dev --num-jobs 4 --output-dir data/manifests

echo "=== Step 4: Compute fbank features ==="
python -m zipvoice.bin.compute_fbank --source-dir data/manifests --dest-dir data/fbank --dataset tsukuyomi --subset train --num-jobs 4

python -m zipvoice.bin.compute_fbank --source-dir data/manifests --dest-dir data/fbank --dataset tsukuyomi --subset dev --num-jobs 4

echo "=== Step 5: Create token file with accent tokens ==="
cp data/tokens_japanese_extended.txt data/tokens_tsukuyomi_accent.txt
echo "Token file: data/tokens_tsukuyomi_accent.txt"

echo "=== Step 6: Pretokenize with accent markers ==="
python -m zipvoice.bin.pretokenize_manifest --input-manifest data/fbank/tsukuyomi_cuts_train.jsonl.gz --output-manifest data/fbank/tsukuyomi_cuts_train_tokenized.jsonl.gz --token-file data/tokens_tsukuyomi_accent.txt --use-accent --num-workers 4

python -m zipvoice.bin.pretokenize_manifest --input-manifest data/fbank/tsukuyomi_cuts_dev.jsonl.gz --output-manifest data/fbank/tsukuyomi_cuts_dev_tokenized.jsonl.gz --token-file data/tokens_tsukuyomi_accent.txt --use-accent --num-workers 4

echo "=== Preprocessing complete! ==="
echo "Train manifest: data/fbank/tsukuyomi_cuts_train_tokenized.jsonl.gz"
echo "Dev manifest: data/fbank/tsukuyomi_cuts_dev_tokenized.jsonl.gz"
echo "Token file: data/tokens_tsukuyomi_accent.txt"
