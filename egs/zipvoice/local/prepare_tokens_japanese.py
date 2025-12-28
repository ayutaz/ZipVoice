#!/usr/bin/env python3
"""
Generate token file for Japanese TTS training.

This file generates a token file that maps phonemes to IDs, including:
- espeak phonemes for English support
- pyopenjtalk phonemes for Japanese support

Usage:
    uv run python egs/zipvoice/local/prepare_tokens_japanese.py \
        --output data/tokens_japanese.txt
"""

import argparse
import logging
from pathlib import Path
from typing import Set

from piper_phonemize import get_espeak_map


# Japanese phonemes from pyopenjtalk
# Reference: https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html
JAPANESE_PHONEMES = [
    # Vowels
    "a",
    "i",
    "u",
    "e",
    "o",
    # Consonants
    "k",
    "s",
    "t",
    "n",
    "h",
    "m",
    "y",
    "r",
    "w",
    "g",
    "z",
    "d",
    "b",
    "p",
    "f",
    "j",
    "v",
    # Special
    "N",  # ん (moraic nasal)
    "cl",  # っ (geminate/glottal stop)
    "pau",  # pause
    "sil",  # silence
    # Affricates and fricatives
    "ch",
    "sh",
    "ts",
    # Palatalized consonants
    "ky",
    "gy",
    "ny",
    "hy",
    "my",
    "ry",
    "py",
    "by",
    # Additional
    "dy",
    "ty",
    "kw",
    "gw",
    "ts",
    "dz",
    # Long vowels (sometimes output)
    "A",
    "I",
    "U",
    "E",
    "O",
]


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate token file for Japanese TTS"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tokens_japanese.txt"),
        help="Output token file path",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional: Extract additional tokens from manifest file",
    )
    return parser.parse_args()


def get_tokens_from_manifest(manifest_path: Path) -> Set[str]:
    """Extract unique tokens from a manifest file using JapaneseTokenizer."""
    from lhotse import load_manifest_lazy

    from zipvoice.tokenizer.tokenizer import JapaneseTokenizer

    logging.info(f"Loading manifest: {manifest_path}")
    cut_set = load_manifest_lazy(manifest_path)

    tokenizer = JapaneseTokenizer()
    all_tokens = set()

    for cut in cut_set:
        text = cut.supervisions[0].text
        tokens = tokenizer.texts_to_tokens([text])[0]
        all_tokens.update(tokens)

    return all_tokens


def main():
    args = get_args()

    # Get espeak tokens (for English support in mixed text)
    espeak_map = get_espeak_map()  # token: [token_id]
    espeak_tokens = {token: token_id[0] for token, token_id in espeak_map.items()}
    # Sort by token_id
    espeak_tokens = sorted(espeak_tokens.items(), key=lambda x: x[1])

    # Get Japanese phonemes
    japanese_tokens = set(JAPANESE_PHONEMES)

    # Optionally extract additional tokens from manifest
    if args.manifest and args.manifest.exists():
        manifest_tokens = get_tokens_from_manifest(args.manifest)
        logging.info(f"Found {len(manifest_tokens)} unique tokens in manifest")
        japanese_tokens.update(manifest_tokens)

    # Remove tokens already in espeak
    existing_tokens = {t[0] for t in espeak_tokens}
    new_japanese_tokens = sorted(japanese_tokens - existing_tokens)

    logging.info(f"espeak tokens: {len(espeak_tokens)}")
    logging.info(f"New Japanese tokens: {len(new_japanese_tokens)}")

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write token file
    with open(args.output, "w", encoding="utf-8") as f:
        # Write espeak tokens first
        for token, token_id in espeak_tokens:
            f.write(f"{token}\t{token_id}\n")

        # Write Japanese tokens
        num_espeak = len(espeak_tokens)
        for i, token in enumerate(new_japanese_tokens):
            f.write(f"{token}\t{num_espeak + i}\n")

    total_tokens = len(espeak_tokens) + len(new_japanese_tokens)
    logging.info(f"Token file written to: {args.output}")
    logging.info(f"Total tokens: {total_tokens}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)
    main()
