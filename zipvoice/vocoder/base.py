#!/usr/bin/env python3
# Copyright 2025 ZipVoice Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from abc import ABC, abstractmethod

import torch


class BaseVocoder(ABC):
    """Abstract base class for vocoder implementations."""

    @abstractmethod
    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Convert mel-spectrogram to waveform.

        Args:
            mel: Mel-spectrogram tensor with shape (B, n_mels, T)
                 where B is batch size, n_mels is number of mel bins (100),
                 and T is number of time frames.

        Returns:
            Waveform tensor with shape (B, 1, samples)
        """
        pass

    @abstractmethod
    def to(self, device: torch.device) -> "BaseVocoder":
        """Move the vocoder to the specified device."""
        pass

    @abstractmethod
    def eval(self) -> "BaseVocoder":
        """Set the vocoder to evaluation mode."""
        pass
