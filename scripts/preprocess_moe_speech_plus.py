#!/usr/bin/env python3
"""
Preprocessing script for moe-speech-plus dataset.
Converts the dataset to TSV format for ZipVoice training.

Dataset structure:
- ZIP files per character (e.g., 01a5575c.zip)
- Inside each ZIP: data/{char_id}/wav/{char_id}_{num}.wav and .json files
- JSON contains: anime_whisper_transcription, parakeet_jp_transcription, etc.
"""

import argparse
import json
import os
import random
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple


def process_zip_file(zip_path: str, extract_dir: str) -> List[Tuple[str, str, str]]:
    """
    Process a single ZIP file and extract entries.

    Returns:
        List of (unique_id, text, wav_path) tuples
    """
    entries = []

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Get list of files
            names = z.namelist()

            # Find WAV files and their corresponding JSON files
            wav_files = [n for n in names if n.endswith('.wav')]

            for wav_name in wav_files:
                # Find corresponding JSON (not .bak.json)
                json_name = wav_name.replace('.wav', '.json')

                if json_name not in names:
                    continue

                try:
                    # Read JSON metadata
                    json_content = z.read(json_name)
                    metadata = json.loads(json_content)

                    # Use anime_whisper_transcription (better for anime voices)
                    text = metadata.get('anime_whisper_transcription', '')
                    if not text:
                        text = metadata.get('parakeet_jp_transcription', '')

                    if not text:
                        continue

                    # Extract WAV file
                    # Path: data/{char_id}/wav/{char_id}_{num}.wav
                    parts = wav_name.split('/')
                    if len(parts) >= 4:
                        char_id = parts[1]
                        wav_filename = parts[-1]
                        unique_id = wav_filename.replace('.wav', '')

                        # Extract WAV to extract_dir
                        wav_extract_path = os.path.join(extract_dir, char_id, wav_filename)
                        os.makedirs(os.path.dirname(wav_extract_path), exist_ok=True)

                        # Extract file
                        with z.open(wav_name) as src:
                            with open(wav_extract_path, 'wb') as dst:
                                dst.write(src.read())

                        entries.append((unique_id, text, wav_extract_path))

                except Exception as e:
                    print(f"Error processing {wav_name}: {e}")
                    continue

    except Exception as e:
        print(f"Error processing {zip_path}: {e}")

    return entries


def main():
    parser = argparse.ArgumentParser(description="Preprocess moe-speech-plus dataset")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing downloaded ZIP files",
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        default=None,
        help="Directory to extract WAV files (default: {input-dir}/extracted)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for TSV files",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.05,
        help="Ratio of data to use for dev set (default: 0.05)",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=8,
        help="Number of parallel jobs for extraction",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/dev split",
    )
    args = parser.parse_args()

    # Setup directories
    input_dir = Path(args.input_dir)
    extract_dir = args.extract_dir or str(input_dir / "extracted")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)

    # Find all ZIP files
    zip_files = sorted(input_dir.glob("*.zip"))
    print(f"Found {len(zip_files)} ZIP files")

    # Process ZIP files in parallel
    all_entries = []
    with ProcessPoolExecutor(max_workers=args.num_jobs) as executor:
        futures = {
            executor.submit(process_zip_file, str(zf), extract_dir): zf
            for zf in zip_files
        }

        for i, future in enumerate(as_completed(futures)):
            zip_file = futures[future]
            try:
                entries = future.result()
                all_entries.extend(entries)
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(zip_files)} ZIP files, {len(all_entries)} entries so far")
            except Exception as e:
                print(f"Error processing {zip_file}: {e}")

    print(f"Total entries: {len(all_entries)}")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(all_entries)

    dev_count = max(1, int(len(all_entries) * args.dev_ratio))
    dev_entries = all_entries[:dev_count]
    train_entries = all_entries[dev_count:]

    print(f"Train: {len(train_entries)}, Dev: {len(dev_entries)}")

    # Write TSV files
    train_tsv = output_dir / "moe_speech_plus_train.tsv"
    dev_tsv = output_dir / "moe_speech_plus_dev.tsv"

    with open(train_tsv, "w", encoding="utf-8") as f:
        for unique_id, text, wav_path in train_entries:
            f.write(f"{unique_id}\t{text}\t{wav_path}\n")

    with open(dev_tsv, "w", encoding="utf-8") as f:
        for unique_id, text, wav_path in dev_entries:
            f.write(f"{unique_id}\t{text}\t{wav_path}\n")

    print(f"\nOutput files:")
    print(f"  Train: {train_tsv}")
    print(f"  Dev: {dev_tsv}")
    print(f"\nNext steps:")
    print(f"  1. Create manifests: uv run python -m zipvoice.bin.prepare_dataset --tsv-path {train_tsv} ...")
    print(f"  2. Compute fbank: uv run python -m zipvoice.bin.compute_fbank ...")
    print(f"  3. Tokenize: uv run python -m zipvoice.bin.pretokenize_manifest ...")


if __name__ == "__main__":
    main()
