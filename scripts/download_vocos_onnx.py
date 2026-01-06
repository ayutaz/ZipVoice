#!/usr/bin/env python3
"""
Download Vocos ONNX model from HuggingFace for Unity Sentis integration.

The Vocos vocoder converts mel-spectrograms to audio waveforms.
Note: The ONNX model does NOT include ISTFT - this must be implemented in Unity C#.

Usage:
    python scripts/download_vocos_onnx.py
    python scripts/download_vocos_onnx.py --output-dir exp/vocos_onnx

The model is from: https://huggingface.co/wetdog/vocos-mel-24khz-onnx
"""

import argparse
import logging
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser(
        description="Download Vocos ONNX model for Unity Sentis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp/vocos_onnx",
        help="Directory to save the downloaded model",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="wetdog/vocos-mel-24khz-onnx",
        help="HuggingFace repository ID",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Downloading Vocos ONNX from {args.repo_id}")

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        logging.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return 1

    # List files in the repository
    try:
        files = list_repo_files(args.repo_id)
        logging.info(f"Available files: {files}")
    except Exception as e:
        logging.error(f"Failed to list repository files: {e}")
        return 1

    # Download ONNX files
    onnx_files = [f for f in files if f.endswith(".onnx")]
    if not onnx_files:
        logging.error("No ONNX files found in the repository")
        return 1

    for filename in onnx_files:
        logging.info(f"Downloading {filename}...")
        try:
            local_path = hf_hub_download(
                repo_id=args.repo_id,
                filename=filename,
                local_dir=output_dir,
            )
            logging.info(f"  Saved to: {local_path}")
        except Exception as e:
            logging.error(f"Failed to download {filename}: {e}")
            return 1

    # Also download any config files
    config_files = [f for f in files if f.endswith(".json") or f.endswith(".yaml")]
    for filename in config_files:
        logging.info(f"Downloading {filename}...")
        try:
            local_path = hf_hub_download(
                repo_id=args.repo_id,
                filename=filename,
                local_dir=output_dir,
            )
            logging.info(f"  Saved to: {local_path}")
        except Exception as e:
            logging.warning(f"Failed to download {filename}: {e}")

    logging.info("=" * 60)
    logging.info("Download completed!")
    logging.info(f"Output directory: {output_dir}")
    logging.info("")
    logging.info("IMPORTANT: The Vocos ONNX model does NOT include ISTFT.")
    logging.info("You must implement ISTFT in Unity C# with these parameters:")
    logging.info("  - n_fft: 1024")
    logging.info("  - hop_length: 256")
    logging.info("  - win_length: 1024")
    logging.info("  - sample_rate: 24000 Hz")
    logging.info("=" * 60)

    return 0


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    exit(main())
