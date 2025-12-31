# Flow2GAN Vocoder Integration

## Overview

This document describes the investigation and integration of Flow2GAN vocoder into ZipVoice as an alternative to the existing Vocos vocoder.

## Executive Summary

**Conclusion: Flow2GAN does not provide significant benefits over Vocos for ZipVoice.**

| Aspect | Flow2GAN | Vocos | Winner |
|--------|----------|-------|--------|
| Speed | ~0.003s | ~0.003s | Tie |
| Quality | Good (with fixes) | Good | Tie |
| Stability | Requires careful configuration | Simple | Vocos |
| Bottleneck Impact | <5% of total time | <5% of total time | N/A |

The vocoder accounts for only 0.2-5% of total inference time. The bottleneck is the ZipVoice model itself (~95%), so vocoder optimization has minimal impact on overall performance.

## Background

### Current Vocoder: Vocos

ZipVoice uses Vocos (`charactr/vocos-mel-24khz`) to convert mel-spectrograms to waveforms.

### New Vocoder: Flow2GAN

Flow2GAN is a flow-matching based vocoder with GAN fine-tuning, enabling 1-4 step inference.

**Key features:**
- 1-4 step inference (configurable)
- Flow Matching + GAN fine-tuning
- Multi-branch ConvNeXt architecture

## Compatibility Analysis

Both vocoders share identical mel-spectrogram specifications:

| Parameter | ZipVoice | Flow2GAN |
|-----------|----------|----------|
| Sampling Rate | 24kHz | 24kHz |
| Mel Bins | 100 | 100 |
| FFT Size | 1024 | 1024 |
| Hop Length | 256 | 256 |

No changes needed to the feature extraction pipeline.

## Benchmark Results

### Test Environment
- GPU: CUDA-enabled device
- Text: "The quick brown fox jumps over the lazy dog."
- Prompt: Same reference audio for all tests

### Speed Comparison

| Vocoder | Vocoder Time | Total Time | Vocoder % |
|---------|--------------|------------|-----------|
| Vocos | ~0.003s | ~2.8s | 0.1% |
| Flow2GAN 1-step | ~0.003s | ~2.8s | 0.1% |
| Flow2GAN 2-step | ~0.003s | ~2.8s | 0.1% |
| Flow2GAN 4-step | ~0.004s | ~2.8s | 0.1% |

**Key Finding:** The vocoder is not the bottleneck. ZipVoice model inference dominates processing time.

### Quality Comparison

| Vocoder | Audio Quality |
|---------|---------------|
| Vocos | Good, stable |
| Flow2GAN 1-step | Good (after fixes) |
| Flow2GAN 2-step | Good (after fixes) |
| Flow2GAN 4-step | Good (after fixes) |

No significant quality difference between vocoders after applying necessary fixes.

## Issues Discovered and Fixed

### Issue 1: Model Selection Bug

**Problem:** Flow2GAN requires step-specific pre-trained models:
- `libritts-mel-1-step` for 1-step inference
- `libritts-mel-2-step` for 2-step inference
- `libritts-mel-4-step` for 4-step inference

Initial implementation used a fixed model regardless of `n_timesteps`, causing audio corruption for 1-step and 2-step.

**Fix:** Auto-select model based on `n_timesteps`:
```python
if hf_model_name is None:
    hf_model_name = f"libritts-mel-{n_timesteps}-step"
```

### Issue 2: Audio Distortion (Hard Clipping)

**Problem:** Audio distortion ("音割れ") in all step configurations.

**Root Cause:** Training-inference mismatch:
1. During GAN training: `clamp_pred=False` (generator outputs can exceed [-1, 1])
2. During inference: `clamp_pred=True` causes hard clipping -> distortion

The discriminator internally normalizes audio (DC removal + peak normalization to 0.8), so the generator learns to produce values outside [-1, 1].

**Fix:** Apply peak normalization matching the discriminator:
```python
def decode(self, mel: torch.Tensor) -> torch.Tensor:
    wav = self.model.infer(
        cond=mel,
        n_timesteps=self.n_timesteps,
        clamp_pred=False,  # Disable hard clipping
    )
    # Peak normalization (0.8) - matches discriminator training
    wav = wav - wav.mean(dim=-1, keepdim=True)
    wav = 0.8 * wav / (wav.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
    return wav.unsqueeze(1)
```

## Implementation Details

### New Files Created

| File | Description |
|------|-------------|
| `zipvoice/vocoder/__init__.py` | Factory function `get_vocoder()` |
| `zipvoice/vocoder/base.py` | Abstract base class `BaseVocoder` |
| `zipvoice/vocoder/vocos.py` | Vocos wrapper `VocosVocoder` |
| `zipvoice/vocoder/flow2gan.py` | Flow2GAN wrapper `Flow2GANVocoder` |

### Files Modified

| File | Changes |
|------|---------|
| `zipvoice/bin/infer_zipvoice.py` | Added `--vocoder-type` and `--vocoder-n-steps` arguments |
| `pyproject.toml` | Added Flow2GAN dependency |

### Vocoder Package Structure

```
zipvoice/vocoder/
    __init__.py     # Factory function get_vocoder()
    base.py         # BaseVocoder abstract class
    vocos.py        # VocosVocoder wrapper
    flow2gan.py     # Flow2GANVocoder wrapper
```

### Interface Design

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

## Usage

```bash
# Using Vocos (default, recommended)
uv run python -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --prompt-wav prompt.wav \
    --prompt-text "Prompt text" \
    --text "Text to synthesize" \
    --res-wav-path result.wav

# Using Flow2GAN (2-step)
uv run python -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --vocoder-type flow2gan \
    --vocoder-n-steps 2 \
    --prompt-wav prompt.wav \
    --prompt-text "Prompt text" \
    --text "Text to synthesize" \
    --res-wav-path result.wav
```

## Flow2GAN Pre-trained Models

| Model Name | Steps | Use Case |
|------------|-------|----------|
| `libritts-mel-1-step` | 1 | Ultra-fast |
| `libritts-mel-2-step` | 2 | Balanced |
| `libritts-mel-4-step` | 4 | Highest quality |

**Important:** Each step count requires its corresponding pre-trained model.

## Recommendations

1. **Use Vocos as default** - Simpler, stable, and equivalent performance
2. **Keep Flow2GAN as option** - May be useful for future experiments or specific use cases
3. **Focus optimization efforts on ZipVoice model** - The actual bottleneck (95%+ of inference time)

## Future Considerations

- Flow2GAN may be beneficial if future ZipVoice optimizations significantly reduce model inference time
- Custom training of Flow2GAN on ZipVoice-specific data might improve quality
- ONNX/TensorRT export of Flow2GAN for deployment scenarios

## References

- [Flow2GAN Repository](https://github.com/k2-fsa/Flow2GAN)
- [Vocos](https://github.com/gemelo-ai/vocos)
