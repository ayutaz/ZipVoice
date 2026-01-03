"""Merge LoRA weights and export a clean checkpoint."""

import argparse
import json
import logging
import torch

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.modules.lora_utils import (
    apply_lora_to_fm_decoder,
    create_lora_config,
    merge_lora_weights,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--checkpoint-name", type=str, default="best-valid-loss.pt")
    parser.add_argument("--output-name", type=str, default="merged.pt")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load model config
    model_config_path = f"{args.model_dir}/model.json"
    with open(model_config_path) as f:
        model_config = json.load(f)

    # Create model with LoRA structure
    model = ZipVoice(
        **model_config["model"],
        vocab_size=model_config["tokenizer"]["vocab_size"],
        pad_id=model_config["tokenizer"]["pad_id"],
    )

    # Apply LoRA to match the saved structure
    lora_config = create_lora_config(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        use_dora=True,
    )
    model = apply_lora_to_fm_decoder(model, lora_config)

    # Load checkpoint
    ckpt_path = f"{args.model_dir}/{args.checkpoint_name}"
    logging.info(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)

    # Merge LoRA weights
    logging.info("Merging LoRA weights...")
    model = merge_lora_weights(model)

    # Save merged checkpoint
    output_path = f"{args.model_dir}/{args.output_name}"
    torch.save({"model": model.state_dict()}, output_path)
    logging.info(f"Merged checkpoint saved to {output_path}")


if __name__ == "__main__":
    main()
