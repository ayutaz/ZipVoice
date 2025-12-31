#!/usr/bin/env python3
# Copyright 2025 ZipVoice Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from typing import Optional

from zipvoice.vocoder.base import BaseVocoder
from zipvoice.vocoder.flow2gan import Flow2GANVocoder
from zipvoice.vocoder.vocos import VocosVocoder

__all__ = ["BaseVocoder", "VocosVocoder", "Flow2GANVocoder", "get_vocoder"]


def get_vocoder(
    vocoder_type: str = "vocos",
    local_path: Optional[str] = None,
    n_timesteps: int = 2,
    **kwargs,
) -> BaseVocoder:
    """
    Factory function to create a vocoder instance.

    Args:
        vocoder_type: Type of vocoder to use ("vocos" or "flow2gan").
        local_path: Optional path to local model directory (Vocos only).
        n_timesteps: Number of ODE solver steps for Flow2GAN (1, 2, or 4).
        **kwargs: Additional arguments passed to the vocoder constructor.

    Returns:
        A vocoder instance implementing BaseVocoder interface.

    Raises:
        ValueError: If vocoder_type is not recognized.

    Example:
        >>> vocoder = get_vocoder("flow2gan", n_timesteps=2)
        >>> vocoder = vocoder.to(device).eval()
        >>> wav = vocoder.decode(mel_spectrogram)
    """
    if vocoder_type == "vocos":
        return VocosVocoder(local_path=local_path)
    elif vocoder_type == "flow2gan":
        return Flow2GANVocoder(n_timesteps=n_timesteps, **kwargs)
    else:
        raise ValueError(
            f"Unknown vocoder type: {vocoder_type}. "
            "Supported types: 'vocos', 'flow2gan'"
        )
