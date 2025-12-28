
<div align="right">
  <details>
    <summary >🌐 Language</summary>
    <div>
      <div align="center">
        <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=en">English</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=zh-CN">简体中文</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=zh-TW">繁體中文</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=ja">日本語</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=ko">한국어</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=hi">हिन्दी</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=th">ไทย</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=fr">Français</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=de">Deutsch</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=es">Español</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=it">Itapano</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=ru">Русский</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=pt">Português</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=nl">Nederlands</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=pl">Polski</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=ar">العربية</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=fa">فارسی</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=tr">Türkçe</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=vi">Tiếng Việt</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=id">Bahasa Indonesia</a>
      </div>
    </div>
  </details>
</div>

<div align="center">

# ZipVoice⚡

## Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching
</div>

## Overview

ZipVoice is a series of fast and high-quality zero-shot TTS models based on flow matching.

### 1. Key features

- Small and fast: only 123M parameters.

- High-quality voice cloning: state-of-the-art performance in speaker similarity, intelligibility, and naturalness.

- Multi-lingual: support Chinese, English, and Japanese.

- Multi-mode: support both single-speaker and dialogue speech generation.

### 2. Model variants

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Description</th>
      <th>Paper</th>
      <th>Demo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ZipVoice</td>
      <td>The basic model supporting zero-shot single-speaker TTS in both Chinese and English.</td>
      <td rowspan="2"><a href="https://arxiv.org/abs/2506.13053"><img src="https://img.shields.io/badge/arXiv-Paper-COLOR.svg"></a></td>
      <td rowspan="2"><a href="https://zipvoice.github.io"><img src="https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=Github&style=flat-square"></a></td>
    </tr>
    <tr>
      <td>ZipVoice-Distill</td>
      <td>The distilled version of ZipVoice, featuring improved speed with minimal performance degradation.</td>
    </tr>
    <tr>
      <td>ZipVoice-Dialog</td>
      <td>A dialogue generation model built on ZipVoice, capable of generating single-channel two-party spoken dialogues.</td>
      <td rowspan="2"><a href="https://arxiv.org/abs/2507.09318"><img src="https://img.shields.io/badge/arXiv-Paper-COLOR.svg"></a></td>
      <td rowspan="2"><a href="https://zipvoice-dialog.github.io"><img src="https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=Github&style=flat-square"></a></td>
    </tr>
    <tr>
      <td>ZipVoice-Dialog-Stereo</td>
      <td>The stereo variant of ZipVoice-Dialog, enabling two-channel dialogue generation with each speaker assigned to a distinct channel.</td>
    </tr>
  </tbody>
</table>

## News

**2025/07/14**: **ZipVoice-Dialog** and **ZipVoice-Dialog-Stereo**, two spoken dialogue generation models, are released. [![arXiv](https://img.shields.io/badge/arXiv-Paper-COLOR.svg)](https://arxiv.org/abs/2507.09318) [![demo page](https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=Github&style=flat-square)](https://zipvoice-dialog.github.io)

**2025/07/14**: **OpenDialog** dataset, a 6.8k-hour spoken dialogue dataset, is released. Download at [![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/k2-fsa/OpenDialog), [![ms](https://img.shields.io/badge/ModelScope-Dataset-blue?logo=data)](https://www.modelscope.cn/datasets/k2-fsa/OpenDialog). Check details at [![arXiv](https://img.shields.io/badge/arXiv-Paper-COLOR.svg)](https://arxiv.org/abs/2507.09318).

**2025/06/16**: **ZipVoice** and **ZipVoice-Distill** are released. [![arXiv](https://img.shields.io/badge/arXiv-Paper-COLOR.svg)](https://arxiv.org/abs/2506.13053) [![demo page](https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=Github&style=flat-square)](https://zipvoice.github.io)

## Installation

### 1. Clone the ZipVoice repository

```bash
git clone https://github.com/k2-fsa/ZipVoice.git
cd ZipVoice
```

### 2. Install uv (if not already installed)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Install the required packages

```bash
uv sync
uv pip install piper-phonemize --find-links https://k2-fsa.github.io/icefall/piper_phonemize.html
```

> **For Japanese TTS:** The `pyopenjtalk-plus` package is already included in the dependencies for Japanese G2P (grapheme-to-phoneme) conversion.

### 4. (Optional) Install k2 for training or efficient inference

k2 provides CUDA-optimized activation functions that speed up both training and inference. While k2 is **recommended for best performance**, ZipVoice will fall back to PyTorch implementations if k2 is not installed.

> **Note:**  Make sure to install the k2 version that matches your PyTorch and CUDA version. For example, if you are using pytorch 2.5.1 and CUDA 12.1, you can install k2 as follows:

```bash
uv pip install k2==1.24.4.dev20250208+cuda12.1.torch2.5.1 -f https://k2-fsa.github.io/k2/cuda.html
```

Please refer to https://k2-fsa.org/get-started/k2/ for details.
Users in China mainland can refer to https://k2-fsa.org/zh-CN/get-started/k2/.

- To check the k2 installation:

```bash
uv run python -c "import k2; print(k2.__file__)"
```

## Usage

### 1. Single-speaker speech generation

To generate single-speaker speech with our pre-trained ZipVoice or ZipVoice-Distill models, use the following commands (Required models will be downloaded from HuggingFace):

#### 1.1 Inference of a single sentence

```bash
uv run python -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --prompt-wav prompt.wav \
    --prompt-text "I am the transcription of the prompt wav." \
    --text "I am the text to be synthesized." \
    --res-wav-path result.wav
```
- `--model-name` can be `zipvoice` or `zipvoice_distill`, which are models before and after distillation, respectively.
- If `<>` or `[]` appear in the text, strings enclosed by them will be treated as special tokens. `<>` denotes Chinese pinyin and `[]` denotes other special tags.

#### 1.2 Inference of a list of sentences

```bash
uv run python -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --test-list test.tsv \
    --res-dir results
```

- Each line of `test.tsv` is in the format of `{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}`.

#### 1.3 Japanese TTS

For Japanese text-to-speech, use the `--tokenizer japanese` option:

```bash
uv run python -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --tokenizer japanese \
    --token-file data/tokens_japanese.txt \
    --prompt-wav prompt.wav \
    --prompt-text "プロンプト音声の書き起こし" \
    --text "合成したいテキスト" \
    --res-wav-path result.wav
```

> **Note:** Japanese TTS requires a fine-tuned model. See the [Japanese Training](#japanese-training) section for details on how to train a Japanese model.

### 2. Dialogue speech generation

#### 2.1 Inference command

To generate two-party spoken dialogues with our pre-trained ZipVoice-Dialogue or ZipVoice-Dialogue-Stereo models, use the following commands (Required models will be downloaded from HuggingFace):

```bash
uv run python -m zipvoice.bin.infer_zipvoice_dialog \
    --model-name "zipvoice_dialog" \
    --test-list test.tsv \
    --res-dir results
```

- `--model-name` can be `zipvoice_dialog` or `zipvoice_dialog_stereo`,
    which generate mono and stereo dialogues, respectively.

#### 2.2 Input formats

Each line of `test.tsv` is in one of the following formats:

(1) **Merged prompt format** where the audios and transcriptions of two speakers prompts are merged into one prompt wav file:
```
{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}
```

- `wav_name` is the name of the output wav file.
- `prompt_transcription` is the transcription of the conversational prompt wav, e.g, "[S1] Hello. [S2] How are you?"
- `prompt_wav` is the path to the prompt wav.
- `text` is the text to be synthesized, e.g. "[S1] I'm fine. [S2] What's your name? [S1] I'm Eric. [S2] Hi Eric."

(2) **Splitted prompt format** where the audios and transciptions of two speakers exist in separate files:

```
{wav_name}\t{spk1_prompt_transcription}\t{spk2_prompt_transcription}\t{spk1_prompt_wav}\t{spk2_prompt_wav}\t{text}
```

- `wav_name` is the name of the output wav file.
- `spk1_prompt_transcription` is the transcription of the first speaker's prompt wav, e.g, "Hello"
- `spk2_prompt_transcription` is the transcription of the second speaker's prompt wav, e.g, "How are you?"
- `spk1_prompt_wav` is the path to the first speaker's prompt wav file.
- `spk2_prompt_wav` is the path to the second speaker's prompt wav file.
- `text` is the text to be synthesized, e.g. "[S1] I'm fine. [S2] What's your name? [S1] I'm Eric. [S2] Hi Eric."

### 3 Guidance for better usage:

#### 3.1 Prompt length

We recommand a short prompt wav file (e.g., less than 3 seconds for single-speaker speech generation, less than 10 seconds for dialogue speech generation) for faster inference speed. A very long prompt will slow down the inference and degenerate the speech quality.

#### 3.2 Speed optimization

If the inference speed is unsatisfactory, you can speed it up as follows:

- **Distill model and less steps**: For the single-speaker speech generation model, we use the `zipvoice` model by default for better speech quality. If faster speed is a priority, you can switch to the `zipvoice_distill` and can reduce the `--num-steps` to as low as `4` (8 by default).

- **CPU speedup with multi-threading**: When running on CPU, you can pass the `--num-thread` parameter (e.g., `--num-thread 4`) to increase the number of threads for faster speed. We use 1 thread by default.

- **CPU speedup with ONNX**: When running on CPU, you can use ONNX models with `zipvoice.bin.infer_zipvoice_onnx` for faster speed (haven't supported ONNX for dialogue generation models yet). For even faster speed, you can further set `--onnx-int8 True` to use an INT8-quantized ONNX model. Note that the quantized model will result in a certain degree of speech quality degradation. **Don't use ONNX on GPU**, as it is slower than PyTorch on GPU.

- **GPU Acceleration with NVIDIA TensorRT**: For a significant performance boost on NVIDIA GPUs, first export the model to a TensorRT engine using zipvoice.bin.tensorrt_export. Then, run inference on your dataset (e.g., a Hugging Face dataset) with zipvoice.bin.infer_zipvoice. This can achieve approximately 2x the throughput compared to the standard PyTorch implementation on a GPU.

#### 3.3 Memory control

The given text will be splitted into chunks based on punctuation (for single-speaker speech generation) or speaker-turn symbol (for dialogue speech generation). Then, the chunked texts will be processed in batches. Therefore, the model can process arbitrarily long text with almost constant memory usage. You can control memory usage by adjusting the `--max-duration` parameter.

#### 3.4 "Raw" evaluation

By default, we preprocess inputs (prompt wav, prompt transcription, and text) for efficient inference and better performance. If you want to evaluate the model’s "raw" performance using exact provided inputs (e.g., to reproduce the results in our paper), you can pass `--raw-evaluation True`.

#### 3.5 Short text

When generating speech for very short texts (e.g., one or two words), the generated speech may sometimes omit certain pronunciations. To resolve this issue, you can pass `--speed 0.3` (where 0.3 is a tunable value) to extend the duration of the generated speech.

#### 3.6 Correcting mispronounced chinese polyphone characters

We use [pypinyin](https://github.com/mozillazg/python-pinyin) to convert Chinese characters to pinyin. However, it can occasionally mispronounce **polyphone characters** (多音字).

To manually correct these mispronunciations, enclose the **corrected pinyin** in angle brackets `< >` and include the **tone mark**.

**Example:**

- Original text: `这把剑长三十公分`
- Correct the pinyin of `长`:  `这把剑<chang2>三十公分`

> **Note:** If you want to manually assign multiple pinyins, enclose each pinyin with `<>`, e.g., `这把<jian4><chang2><san1>十公分`

#### 3.7 Remove long silences from the generated speech

Model will automatically determine the positions and lengths of silences in the generated speech. It occasionally has long silence in the middle of the speech. If you don't want this, you can pass `--remove-long-sil` to remove long silences in the middle of the generated speech (edge silences will be removed by default).

#### 3.8 Model downloading

If you have trouble connecting to HuggingFace when downloading the pre-trained models, try switching endpoint to the mirror site: `export HF_ENDPOINT=https://hf-mirror.com`.

## Train Your Own Model

See the [egs](egs) directory for training, fine-tuning and evaluation examples.

### Japanese Training

To fine-tune ZipVoice for Japanese TTS, follow these steps:

#### 1. Prepare your dataset

Create a TSV file with the format: `{id}\t{text}\t{wav_path}`

For the Tsukuyomi-chan Corpus, use the provided script:

```bash
uv run python egs/zipvoice/local/prepare_tsv_tsukuyomi.py \
    --corpus-dir "path/to/tsukuyomi-corpus" \
    --output data/raw/tsukuyomi_train.tsv
```

#### 2. Prepare manifests and features

```bash
# Create manifests
uv run python -m zipvoice.bin.prepare_dataset \
    --tsv-path data/raw/tsukuyomi_train.tsv \
    --prefix tsukuyomi \
    --subset raw_train \
    --num-jobs 4 \
    --output-dir data/manifests

# Add tokens
uv run python -m zipvoice.bin.prepare_tokens \
    --input-file data/manifests/tsukuyomi_cuts_raw_train.jsonl.gz \
    --output-file data/manifests/tsukuyomi_cuts_train.jsonl.gz \
    --tokenizer japanese

# Compute Fbank features
uv run python -m zipvoice.bin.compute_fbank \
    --source-dir data/manifests \
    --dest-dir data/fbank \
    --dataset tsukuyomi \
    --subset train \
    --num-jobs 4
```

#### 3. Generate Japanese token file

```bash
uv run python egs/zipvoice/local/prepare_tokens_japanese.py \
    --output data/tokens_japanese.txt
```

#### 4. Fine-tune the model

```bash
uv run python -m zipvoice.bin.train_zipvoice \
    --world-size 1 \
    --use-fp16 1 \
    --finetune 1 \
    --base-lr 0.0001 \
    --num-iters 10000 \
    --max-duration 500 \
    --model-config download/zipvoice/model.json \
    --checkpoint download/zipvoice/model.pt \
    --tokenizer japanese \
    --token-file data/tokens_japanese.txt \
    --train-manifest data/fbank/tsukuyomi_cuts_train.jsonl.gz \
    --exp-dir exp/zipvoice_japanese \
    --wandb-project zipvoice-japanese
```

> **Note:** Training logs are sent to [Weights & Biases](https://wandb.ai) by default. Use `--no-wandb` to disable.

For a complete training recipe, see `egs/zipvoice/run_japanese.sh`.

## Production Deployment

### NVIDIA Triton GPU Runtime

For production-ready deployment with high performance and scalability, check out the [Triton Inference Server integration](runtime/nvidia_triton/) that provides optimized TensorRT engines, concurrent request handling, and both gRPC/HTTP APIs for enterprise use.

### CPU Deployment

Check [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/pull/2487#issuecomment-3227884498) for the C++ deployment solution on CPU.

## Performance Optimizations

This section describes performance optimizations implemented in the `feature/performance-optimization` branch.

### Optimization Summary

| Category | Optimization | Effect | Target |
|----------|-------------|--------|--------|
| **CUDA** | `cudnn.benchmark=True` | 5-10% | Inference |
| **CUDA** | TF32 enabled | 10-20% | Inference (Ampere+ GPUs) |
| **CUDA** | `torch.compile()` | 10-30% | Inference (PyTorch 2.0+) |
| **Memory** | `expand()` instead of `repeat()` | 30% memory reduction | Inference |
| **Tokenizer** | pypinyin LRU cache | 60-80% tokenization speedup | Chinese text |
| **Tokenizer** | Pre-compiled regex | 20-30% | All text |
| **Training** | DataLoader optimization | 15-25% | Training |
| **Training** | DDP settings | 3-7% | Distributed training |
| **ONNX** | Graph optimization level | 15-25% | CPU inference |

### Benchmark Results

Tested on CUDA GPU with English text ("Hello, this is a test of the English speech synthesis system."):

| Version | Average RTF | Realtime Factor |
|---------|-------------|-----------------|
| Baseline | 0.7982 | 1.25x |
| Optimized | 0.7884 | 1.27x |
| **Improvement** | - | **~1.2%** |

> **Note:** RTF (Real-Time Factor) = processing_time / audio_duration. Lower is better.

### k2 Library for Maximum Performance

For maximum inference performance, install the k2 library which provides CUDA-optimized Swoosh activation functions:

```bash
# Install k2 matching your PyTorch/CUDA version
uv pip install k2==1.24.4.dev20250208+cuda12.1.torch2.5.1 \
  -f https://k2-fsa.github.io/k2/cuda.html
```

Without k2, the following warning appears and PyTorch fallback is used:
```
WARNING: Failed import k2. Swoosh functions will fallback to PyTorch implementation,
leading to slower speed and higher memory consumption.
```

**Expected improvement with k2:**
- Inference: 10-20% speedup (CUDA-optimized activation functions)
- Training: 20-30% speedup (optimized gradient computation)
- Memory: Reduced consumption

> **Note:** k2 is only available on Linux/WSL2. It does not work on native Windows.

### Implementation Details

**Inference optimizations** (`zipvoice/bin/infer_zipvoice.py`):
```python
# CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# torch.compile for PyTorch 2.0+
model = torch.compile(model, mode="reduce-overhead")

# Memory-efficient tensor expansion
batch_prompt_features = prompt_features.expand(len(batch_tokens), -1, -1)
```

**Training optimizations** (`zipvoice/bin/train_zipvoice.py`):
```python
# DDP optimization
model = DDP(
    model,
    device_ids=[rank],
    find_unused_parameters=False,
    gradient_as_bucket_view=True,
)

# DataLoader optimization
DataLoader(
    ...,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
)
```

**Tokenizer optimizations** (`zipvoice/tokenizer/tokenizer.py`):
```python
# LRU cache for pypinyin conversion
@lru_cache(maxsize=10000)
def _cached_lazy_pinyin_word(word: str) -> tuple:
    return tuple(lazy_pinyin([word], style=Style.TONE3, ...))

# Pre-compiled regex patterns
class EmiliaTokenizer:
    _PART_PATTERN = re.compile(r"[<[].*?[>\]]|.")
    _SPLIT_PATTERN = re.compile(r"([<[].*?[>\]])")
```

## Discussion & Communication

You can directly discuss on [Github Issues](https://github.com/k2-fsa/ZipVoice/issues).

You can also scan the QR code to join our wechat group or follow our wechat official account.

| Wechat Group | Wechat Official Account |
| ------------ | ----------------------- |
|![wechat](https://k2-fsa.org/zh-CN/assets/pic/wechat_group.jpg) |![wechat](https://k2-fsa.org/zh-CN/assets/pic/wechat_account.jpg) |

## Citation

```bibtex
@article{zhu2025zipvoice,
      title={ZipVoice: Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching},
      author={Zhu, Han and Kang, Wei and Yao, Zengwei and Guo, Liyong and Kuang, Fangjun and Li, Zhaoqing and Zhuang, Weiji and Lin, Long and Povey, Daniel},
      journal={arXiv preprint arXiv:2506.13053},
      year={2025}
}

@article{zhu2025zipvoicedialog,
      title={ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching},
      author={Zhu, Han and Kang, Wei and Guo, Liyong and Yao, Zengwei and Kuang, Fangjun and Zhuang, Weiji and Li, Zhaoqing and Han, Zhifeng and Zhang, Dong and Zhang, Xin and Song, Xingchen and Lin, Long and Povey, Daniel},
      journal={arXiv preprint arXiv:2507.09318},
      year={2025}
}
```
