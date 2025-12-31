# Flow2GAN Vocoder Integration

## Overview

This document describes the integration of Flow2GAN vocoder into ZipVoice to replace the existing Vocos vocoder for improved inference speed.

## Background

### Current Vocoder: Vocos

ZipVoice currently uses Vocos (`charactr/vocos-mel-24khz`) as its vocoder to convert mel-spectrograms to waveforms.

**Current implementation** (`zipvoice/bin/infer_zipvoice.py`):
```python
from vocos import Vocos
vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)
```

### New Vocoder: Flow2GAN

Flow2GAN is a flow-matching based vocoder that enables high-quality waveform generation in as few as 1-4 steps.

**Key features:**
- 1-4 step inference (configurable)
- Flow Matching + GAN fine-tuning for quality
- Multi-branch ConvNeXt architecture
- Compatible mel-spectrogram specifications

## Compatibility Analysis

Both vocoders share identical mel-spectrogram specifications:

| Parameter | ZipVoice (Vocos) | Flow2GAN |
|-----------|------------------|----------|
| Sampling Rate | 24kHz | 24kHz |
| Mel Bins | 100 | 100 |
| FFT Size | 1024 | 1024 |
| Hop Length | 256 | 256 |
| Power | 1 | 1 |

This means no changes are needed to the feature extraction pipeline.

## Flow2GAN Architecture

```
Mel-Spectrogram (B, 100, T)
        |
        v
[CondEncoder] - 4 ConvNeXt layers -> (B, 512, T)
        |
        v
[Multi-Branch AudioConvNeXt]
   |-- Branch 1: FFT=512, hop=256, 768ch, 8 layers
   |-- Branch 2: FFT=256, hop=128, 512ch, 8 layers
   |-- Branch 3: FFT=128, hop=64,  384ch, 8 layers
        |
        v (mean fusion)
Waveform (B, samples)
```

### Inference API

```python
from flow2gan import get_model

model, cfg = get_model(
    model_name="mel_24k_base",
    hf_model_name="libritts-mel-4-step"  # Options: 1-step, 2-step, 4-step
)

# Inference
pred_audio = model.infer(
    cond=mel_spec,      # (B, 100, T)
    n_timesteps=2,      # 1, 2, or 4
    clamp_pred=True
)  # -> (B, samples)
```

### Available Pre-trained Models

| Model Name | Steps | Use Case |
|------------|-------|----------|
| `libritts-mel-1-step` | 1 | Ultra-fast, real-time |
| `libritts-mel-2-step` | 2 | Balanced (recommended) |
| `libritts-mel-4-step` | 4 | Highest quality |

## Implementation Plan

### 1. Vocoder Package Structure

Create a unified vocoder abstraction:

```
zipvoice/vocoder/
    __init__.py     # Factory function get_vocoder()
    base.py         # BaseVocoder abstract class
    vocos.py        # VocosVocoder wrapper
    flow2gan.py     # Flow2GANVocoder wrapper
```

### 2. Interface Design

```python
# Base class
class BaseVocoder(ABC):
    @abstractmethod
    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        """(B, 100, T) -> (B, 1, samples)"""
        pass

# Factory function
def get_vocoder(vocoder_type: str = "vocos", **kwargs) -> BaseVocoder:
    if vocoder_type == "vocos":
        return VocosVocoder(**kwargs)
    elif vocoder_type == "flow2gan":
        return Flow2GANVocoder(**kwargs)
```

### 3. CLI Arguments

New arguments for inference scripts:

```
--vocoder-type    : "vocos" (default) or "flow2gan"
--vocoder-n-steps : 1, 2, or 4 (Flow2GAN only, default: 2)
```

### 4. Files to Modify

| File | Changes |
|------|---------|
| `zipvoice/bin/infer_zipvoice.py` | Add vocoder arguments, use factory |
| `zipvoice/bin/infer_zipvoice_dialog.py` | Same as above |
| `zipvoice/bin/infer_zipvoice_onnx.py` | Same as above |
| `runtime/nvidia_triton/model_repo/zipvoice/1/model.py` | Add vocoder config |

## Installation

```bash
# Add Flow2GAN as a dependency
uv add flow2gan --editable "C:\Users\yuta\Desktop\Private\Flow2GAN"
```

## Usage

```bash
# Using Flow2GAN (2-step, recommended)
uv run python -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --vocoder-type flow2gan \
    --vocoder-n-steps 2 \
    --prompt-wav prompt.wav \
    --text "Text to synthesize" \
    --res-wav-path result.wav

# Using Vocos (default, unchanged behavior)
uv run python -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --prompt-wav prompt.wav \
    --text "Text to synthesize" \
    --res-wav-path result.wav
```

## Expected Performance

| Vocoder | Steps | Relative Speed |
|---------|-------|---------------|
| Vocos | 1 | 1.0x (baseline) |
| Flow2GAN | 1 | ~1.2x faster |
| Flow2GAN | 2 | ~1.0x |
| Flow2GAN | 4 | ~0.8x |

Note: Actual performance depends on hardware. Flow2GAN with 1-2 steps offers competitive speed with potentially better quality for certain audio characteristics.

## References

- [Flow2GAN Repository](https://github.com/k2-fsa/Flow2GAN)
- [Vocos](https://github.com/gemelo-ai/vocos)
