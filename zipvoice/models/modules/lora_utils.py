"""LoRA utilities for ZipVoice

This module provides utilities for applying LoRA (Low-Rank Adaptation) or DoRA
(Weight-Decomposed LoRA) to ZipVoice models, specifically targeting the FM decoder's
attention layers.
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


# Default target modules for LoRA in ZipVoice FM decoder
DEFAULT_TARGET_MODULES = [
    "self_attn_weights.in_proj",
    "self_attn1.in_proj",
    "self_attn1.out_proj",
    "self_attn2.in_proj",
    "self_attn2.out_proj",
]


def create_lora_config(
    rank: int = 32,
    alpha: int = 64,
    dropout: float = 0.05,
    use_dora: bool = True,
    target_modules: Optional[List[str]] = None,
) -> LoraConfig:
    """
    Create a LoRA configuration for ZipVoice FM decoder.

    Args:
        rank: LoRA rank. Higher values (32+) are recommended for style learning.
        alpha: Scaling factor. Typically 2x rank.
        dropout: Dropout rate for LoRA layers.
        use_dora: Whether to use DoRA (Weight-Decomposed LoRA) for better performance.
        target_modules: List of module names to apply LoRA. If None, uses default
                       attention layers.

    Returns:
        LoraConfig object configured for ZipVoice.
    """
    if target_modules is None:
        target_modules = DEFAULT_TARGET_MODULES

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        use_dora=use_dora,
        target_modules=target_modules,
        bias="none",
        task_type=None,  # Custom model, not a standard task
    )


def apply_lora_to_fm_decoder(
    model: nn.Module,
    lora_config: LoraConfig,
) -> nn.Module:
    """
    Apply LoRA to the FM decoder of a ZipVoice model.

    This function wraps the fm_decoder with PEFT's LoRA implementation,
    freezing the base model weights and adding trainable LoRA parameters.

    Args:
        model: ZipVoice model with fm_decoder attribute.
        lora_config: LoRA configuration.

    Returns:
        Model with LoRA-wrapped fm_decoder.

    Raises:
        AttributeError: If model doesn't have fm_decoder attribute.
    """
    if not hasattr(model, "fm_decoder"):
        raise AttributeError("Model must have 'fm_decoder' attribute")

    # Apply LoRA to FM decoder
    model.fm_decoder = get_peft_model(model.fm_decoder, lora_config)

    return model


def get_lora_trainable_params(model: nn.Module) -> dict:
    """
    Get statistics about trainable parameters in a LoRA-enabled model.

    Args:
        model: Model with LoRA applied.

    Returns:
        Dictionary with parameter statistics:
        - trainable_params: Number of trainable parameters
        - total_params: Total number of parameters
        - trainable_percent: Percentage of trainable parameters
    """
    trainable_params = 0
    total_params = 0

    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_percent": 100 * trainable_params / total_params if total_params > 0 else 0,
    }


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights into the base model for inference/export.

    After merging, the model no longer has separate LoRA parameters and can be
    exported to ONNX or used for inference without PEFT dependencies.

    Args:
        model: Model with LoRA-wrapped fm_decoder.

    Returns:
        Model with merged weights (no longer has LoRA structure).
    """
    if hasattr(model.fm_decoder, "merge_and_unload"):
        logging.info("Merging LoRA weights into base model...")
        model.fm_decoder = model.fm_decoder.merge_and_unload()
        logging.info("LoRA weights merged successfully")
    else:
        logging.warning("fm_decoder does not have merge_and_unload method. "
                       "Model may not have LoRA applied.")

    return model


def save_lora_weights(model: nn.Module, save_path: str) -> None:
    """
    Save only the LoRA adapter weights (not the full model).

    Args:
        model: Model with LoRA-wrapped fm_decoder.
        save_path: Path to save the adapter weights.
    """
    if hasattr(model.fm_decoder, "save_pretrained"):
        model.fm_decoder.save_pretrained(save_path)
        logging.info(f"LoRA weights saved to {save_path}")
    else:
        raise ValueError("fm_decoder does not have save_pretrained method. "
                        "LoRA may not be applied.")


def load_lora_weights(model: nn.Module, load_path: str) -> nn.Module:
    """
    Load LoRA adapter weights into a model.

    Args:
        model: Base ZipVoice model (without LoRA).
        load_path: Path to the saved adapter weights.

    Returns:
        Model with LoRA weights loaded.
    """
    from peft import PeftModel

    model.fm_decoder = PeftModel.from_pretrained(
        model.fm_decoder,
        load_path,
    )
    logging.info(f"LoRA weights loaded from {load_path}")

    return model
