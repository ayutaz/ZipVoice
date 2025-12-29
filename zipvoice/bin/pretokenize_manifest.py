#!/usr/bin/env python3
"""
Pre-tokenize manifest files with JapaneseTokenizer.

This script reads a manifest file, tokenizes each text using JapaneseTokenizer,
and saves the tokens in the supervision's custom field. This avoids on-the-fly
tokenization during training, significantly speeding up data loading.

Supports multiprocessing for faster processing on multi-core systems.

Usage:
    python -m zipvoice.bin.pretokenize_manifest \
        --input-manifest data/fbank/moe_speech_cuts_train.jsonl.gz \
        --output-manifest data/fbank/moe_speech_cuts_train_tokenized.jsonl.gz \
        --token-file data/tokens_japanese_extended.txt \
        --num-workers 8
"""

import argparse
import gzip
import json
import logging
import multiprocessing as mp
import os
from functools import partial
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def tokenize_chunk(args: Tuple[List[str], str]) -> List[str]:
    """Worker function to tokenize a chunk of cuts.

    Each worker process initializes its own JapaneseTokenizer instance.

    Args:
        args: Tuple of (chunk of cut JSON strings, token file path)

    Returns:
        List of tokenized cut JSON strings
    """
    chunk, token_file = args

    # Import and initialize tokenizer in worker process
    from zipvoice.tokenizer.tokenizer import JapaneseTokenizer
    tokenizer = JapaneseTokenizer(token_file)

    results = []
    for cut_json in chunk:
        cut = json.loads(cut_json)

        # Get text from supervision
        if 'supervisions' in cut and len(cut['supervisions']) > 0:
            text = cut['supervisions'][0].get('text', '')
            if text:
                # Tokenize text to phonemes
                tokens = tokenizer.texts_to_tokens([text])[0]
                # Convert tokens to IDs
                token_ids = tokenizer.tokens_to_token_ids([tokens])[0]

                # Store in custom field
                if 'custom' not in cut['supervisions'][0]:
                    cut['supervisions'][0]['custom'] = {}
                cut['supervisions'][0]['custom']['tokens'] = tokens
                cut['supervisions'][0]['custom']['token_ids'] = token_ids

        results.append(json.dumps(cut, ensure_ascii=False))

    return results


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize manifest with JapaneseTokenizer")
    parser.add_argument("--input-manifest", type=str, required=True,
                        help="Input manifest file (jsonl.gz)")
    parser.add_argument("--output-manifest", type=str, required=True,
                        help="Output manifest file (jsonl.gz)")
    parser.add_argument("--token-file", type=str, required=True,
                        help="Token file for JapaneseTokenizer")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of worker processes (default: 70%% of CPU cores)")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Number of samples per chunk (default: 500)")
    args = parser.parse_args()

    input_path = Path(args.input_manifest)
    output_path = Path(args.output_manifest)

    # Determine number of workers (default: 70% of CPU cores)
    if args.num_workers is None:
        num_workers = max(1, int(os.cpu_count() * 0.7))
    else:
        num_workers = max(1, args.num_workers)

    logging.info(f"Using {num_workers} worker processes with chunk size {args.chunk_size}")

    # Read all cuts
    logging.info(f"Reading manifest from {input_path}...")
    cuts = []
    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        for line in f:
            cuts.append(line.strip())
    logging.info(f"Found {len(cuts)} cuts")

    # Split into chunks
    chunk_size = args.chunk_size
    chunks = [cuts[i:i+chunk_size] for i in range(0, len(cuts), chunk_size)]
    logging.info(f"Split into {len(chunks)} chunks")

    # Tokenize with multiprocessing
    logging.info("Tokenizing with multiprocessing...")

    # Prepare arguments for workers (chunk, token_file)
    chunk_args = [(chunk, args.token_file) for chunk in chunks]

    # Use spawn context for compatibility with Docker/Linux
    ctx = mp.get_context('spawn')

    tokenized_cuts = []
    with ctx.Pool(processes=num_workers) as pool:
        # Use imap for ordered results with progress bar
        for chunk_results in tqdm(
            pool.imap(tokenize_chunk, chunk_args),
            total=len(chunks),
            desc="Tokenizing chunks"
        ):
            tokenized_cuts.extend(chunk_results)

    logging.info(f"Tokenized {len(tokenized_cuts)} cuts")

    # Write output
    logging.info(f"Writing tokenized manifest to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, 'wt', encoding='utf-8') as f:
        for cut_json in tokenized_cuts:
            f.write(cut_json + '\n')

    # Verify first entry
    with gzip.open(output_path, 'rt', encoding='utf-8') as f:
        first_cut = json.loads(f.readline())
        if 'custom' in first_cut.get('supervisions', [{}])[0]:
            tokens = first_cut['supervisions'][0]['custom'].get('tokens', [])
            token_ids = first_cut['supervisions'][0]['custom'].get('token_ids', [])
            logging.info(f"Sample tokens: {tokens[:10]}...")
            logging.info(f"Sample token_ids: {token_ids[:10]}...")
        else:
            logging.warning("No tokens found in first cut!")

    logging.info("Done!")


if __name__ == "__main__":
    main()
