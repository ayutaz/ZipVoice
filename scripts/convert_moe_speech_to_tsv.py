#!/usr/bin/env python3
"""
Convert moe-speech-20speakers-ljspeech dataset to ZipVoice TSV format.

This script:
1. Reads metadata.csv (LJSpeech format: filename|speaker_id|text)
2. Resamples audio from 22050Hz to 24000Hz
3. Splits into train/dev sets (90%/10%)
4. Outputs TSV files for ZipVoice training

Usage:
    python scripts/convert_moe_speech_to_tsv.py \
        --input-dir data/raw/moe-speech \
        --output-dir data/raw \
        --num-jobs 8
"""

import argparse
import os
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import torchaudio


def resample_audio(args):
    """Resample a single audio file from 22050Hz to 24000Hz."""
    input_path, output_path, target_sr = args

    try:
        waveform, sr = torchaudio.load(input_path)

        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, waveform, target_sr)

        return True, input_path
    except Exception as e:
        return False, f"{input_path}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Convert moe-speech to ZipVoice TSV format")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Input directory containing metadata.csv and wavs/")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for TSV files and resampled audio")
    parser.add_argument("--target-sr", type=int, default=24000,
                        help="Target sample rate (default: 24000)")
    parser.add_argument("--train-ratio", type=float, default=0.9,
                        help="Ratio of training data (default: 0.9)")
    parser.add_argument("--num-jobs", type=int, default=8,
                        help="Number of parallel jobs for resampling")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/dev split")
    parser.add_argument("--skip-resample", action="store_true",
                        help="Skip resampling if already done")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    metadata_path = input_dir / "metadata.csv"
    input_wavs_dir = input_dir / "wavs"
    output_wavs_dir = output_dir / "moe_speech_24k"

    # Read metadata
    print(f"Reading metadata from {metadata_path}...")
    entries = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 3:
                filename = parts[0]
                speaker_id = parts[1]
                text = parts[2]
                entries.append({
                    "filename": filename,
                    "speaker_id": speaker_id,
                    "text": text,
                    "input_wav": str(input_wavs_dir / f"{filename}.wav"),
                    "output_wav": str(output_wavs_dir / f"{filename}.wav"),
                })

    print(f"Found {len(entries)} entries")

    # Resample audio files
    if not args.skip_resample:
        print(f"\nResampling audio files to {args.target_sr}Hz...")
        os.makedirs(output_wavs_dir, exist_ok=True)

        resample_tasks = [
            (e["input_wav"], e["output_wav"], args.target_sr)
            for e in entries
        ]

        success_count = 0
        error_count = 0

        with ProcessPoolExecutor(max_workers=args.num_jobs) as executor:
            futures = {executor.submit(resample_audio, task): task for task in resample_tasks}

            with tqdm(total=len(futures), desc="Resampling") as pbar:
                for future in as_completed(futures):
                    success, msg = future.result()
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        if error_count <= 5:
                            print(f"Error: {msg}")
                    pbar.update(1)

        print(f"Resampling complete: {success_count} success, {error_count} errors")
    else:
        print("Skipping resampling (--skip-resample)")

    # Split into train/dev
    print(f"\nSplitting into train/dev (ratio: {args.train_ratio})...")
    random.seed(args.seed)
    random.shuffle(entries)

    split_idx = int(len(entries) * args.train_ratio)
    train_entries = entries[:split_idx]
    dev_entries = entries[split_idx:]

    print(f"Train: {len(train_entries)}, Dev: {len(dev_entries)}")

    # Write TSV files
    # Format: uniq_id\ttext\twav_path
    train_tsv = output_dir / "moe_speech_train.tsv"
    dev_tsv = output_dir / "moe_speech_dev.tsv"

    print(f"\nWriting TSV files...")

    with open(train_tsv, "w", encoding="utf-8") as f:
        for e in train_entries:
            wav_path = e["output_wav"] if not args.skip_resample else e["input_wav"]
            f.write(f"{e['filename']}\t{e['text']}\t{wav_path}\n")

    with open(dev_tsv, "w", encoding="utf-8") as f:
        for e in dev_entries:
            wav_path = e["output_wav"] if not args.skip_resample else e["input_wav"]
            f.write(f"{e['filename']}\t{e['text']}\t{wav_path}\n")

    print(f"Wrote {train_tsv}")
    print(f"Wrote {dev_tsv}")

    # Print statistics
    print("\n=== Summary ===")
    print(f"Total entries: {len(entries)}")
    print(f"Train entries: {len(train_entries)}")
    print(f"Dev entries: {len(dev_entries)}")
    print(f"Sample rate: {args.target_sr}Hz")
    print(f"Output wavs: {output_wavs_dir}")
    print(f"Train TSV: {train_tsv}")
    print(f"Dev TSV: {dev_tsv}")


if __name__ == "__main__":
    main()
