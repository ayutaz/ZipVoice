#!/usr/bin/env python3
"""
Prepare TSV file from Tsukuyomi-chan Corpus for ZipVoice training.

Usage:
    uv run python egs/zipvoice/local/prepare_tsv_tsukuyomi.py \
        --corpus-dir "C:\path\to\つくよみちゃんコーパス Vol.1 声優統計コーパス（JVSコーパス準拠）" \
        --output-tsv data/raw/tsukuyomi_train.tsv
"""

import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare TSV file from Tsukuyomi-chan Corpus"
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        required=True,
        help="Path to the Tsukuyomi-chan corpus directory",
    )
    parser.add_argument(
        "--output-tsv",
        type=str,
        default="data/raw/tsukuyomi_train.tsv",
        help="Output TSV file path",
    )
    parser.add_argument(
        "--wav-subdir",
        type=str,
        default="02 WAV（+12dB増幅）",
        help="Subdirectory containing WAV files",
    )
    parser.add_argument(
        "--transcript-file",
        type=str,
        default="04 台本と補足資料/★台本テキスト/01 補足なし台本（JSUTコーパス・JVSコーパス版）.txt",
        help="Path to transcript file relative to corpus directory",
    )
    return parser.parse_args()


def load_transcripts(transcript_path: str) -> dict:
    """Load transcripts from the corpus transcript file.

    Args:
        transcript_path: Path to the transcript file

    Returns:
        Dictionary mapping utterance ID to text
    """
    transcripts = {}
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: VOICEACTRESS100_001:テキスト...
            if ":" in line:
                utt_id, text = line.split(":", 1)
                transcripts[utt_id] = text
    return transcripts


def main():
    args = parse_args()

    corpus_dir = Path(args.corpus_dir)
    wav_dir = corpus_dir / args.wav_subdir
    transcript_path = corpus_dir / args.transcript_file

    # Verify paths exist
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    if not wav_dir.exists():
        raise FileNotFoundError(f"WAV directory not found: {wav_dir}")
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    # Load transcripts
    print(f"Loading transcripts from: {transcript_path}")
    transcripts = load_transcripts(str(transcript_path))
    print(f"Loaded {len(transcripts)} transcripts")

    # Create output directory
    output_path = Path(args.output_tsv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate TSV
    entries = []
    missing_wav = []
    missing_transcript = []

    # Get all WAV files
    wav_files = sorted(wav_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} WAV files")

    for wav_file in wav_files:
        utt_id = wav_file.stem  # e.g., VOICEACTRESS100_001

        if utt_id not in transcripts:
            missing_transcript.append(utt_id)
            continue

        text = transcripts[utt_id]
        wav_path = str(wav_file.absolute())

        # TSV format: {id}\t{text}\t{wav_path}
        entries.append(f"{utt_id}\t{text}\t{wav_path}")

    # Check for transcripts without WAV files
    wav_ids = {f.stem for f in wav_files}
    for utt_id in transcripts:
        if utt_id not in wav_ids:
            missing_wav.append(utt_id)

    # Write TSV
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry + "\n")

    print(f"\nTSV file written to: {output_path}")
    print(f"Total entries: {len(entries)}")

    if missing_wav:
        print(f"Warning: {len(missing_wav)} transcripts without WAV files")
    if missing_transcript:
        print(f"Warning: {len(missing_transcript)} WAV files without transcripts")


if __name__ == "__main__":
    main()
