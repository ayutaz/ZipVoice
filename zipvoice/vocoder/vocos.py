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
from vocos import Vocos

from zipvoice.vocoder.base import BaseVocoder


class VocosVocoder(BaseVocoder):
    """Vocos vocoder wrapper."""

    def __init__(self, local_path: Optional[str] = None):
        """
        Initialize Vocos vocoder.

        Args:
            local_path: Optional path to local Vocos model directory.
                       If None, downloads from HuggingFace Hub.
        """
        if local_path:
            self.model = Vocos.from_hparams(f"{local_path}/config.yaml")
            state_dict = torch.load(
                f"{local_path}/pytorch_model.bin",
                weights_only=True,
                map_location="cpu",
            )
            self.model.load_state_dict(state_dict)
        else:
            self.model = Vocos.from_pretrained("charactr/vocos-mel-24khz")

    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Convert mel-spectrogram to waveform.

        Args:
            mel: (B, 100, T) mel-spectrogram

        Returns:
            (B, 1, samples) waveform
        """
        return self.model.decode(mel)

    def to(self, device: torch.device) -> "VocosVocoder":
        self.model = self.model.to(device)
        return self

    def eval(self) -> "VocosVocoder":
        self.model.eval()
        return self
