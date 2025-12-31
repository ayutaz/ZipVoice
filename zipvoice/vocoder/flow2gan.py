#!/usr/bin/env python3
# Copyright 2025 ZipVoice Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from typing import Optional

import torch
from flow2gan import get_model

from zipvoice.vocoder.base import BaseVocoder


class Flow2GANVocoder(BaseVocoder):
    """Flow2GAN vocoder wrapper."""

    def __init__(
        self,
        n_timesteps: int = 2,
        clamp_pred: bool = True,
        model_name: str = "mel_24k_base",
        hf_model_name: Optional[str] = None,
        checkpoint: Optional[str] = None,
    ):
        """
        Initialize Flow2GAN vocoder.

        Args:
            n_timesteps: Number of ODE solver steps (1, 2, or 4).
                        Lower = faster, higher = better quality.
                        Each step count requires a corresponding pre-trained model.
            clamp_pred: Whether to clamp output to [-1, 1].
            model_name: Model configuration name.
            hf_model_name: HuggingFace model name for pre-trained weights.
                          If None, automatically selects based on n_timesteps.
            checkpoint: Optional path to local checkpoint file.
        """
        if n_timesteps not in (1, 2, 4):
            raise ValueError(f"n_timesteps must be 1, 2, or 4, got {n_timesteps}")

        self.n_timesteps = n_timesteps
        self.clamp_pred = clamp_pred

        # Auto-select model based on n_timesteps if not specified
        if hf_model_name is None:
            hf_model_name = f"libritts-mel-{n_timesteps}-step"

        self.model, self.config = get_model(
            model_name=model_name,
            hf_model_name=hf_model_name,
            checkpoint=checkpoint,
        )
        self._device = torch.device("cpu")

    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Convert mel-spectrogram to waveform.

        Args:
            mel: (B, 100, T) mel-spectrogram

        Returns:
            (B, 1, samples) waveform
        """
        # Flow2GAN outputs (B, samples), we add channel dim for compatibility
        wav = self.model.infer(
            cond=mel,
            n_timesteps=self.n_timesteps,
            clamp_pred=self.clamp_pred,
        )
        return wav.unsqueeze(1)

    def to(self, device: torch.device) -> "Flow2GANVocoder":
        self.model = self.model.to(device)
        self._device = device
        return self

    def eval(self) -> "Flow2GANVocoder":
        self.model.eval()
        return self
