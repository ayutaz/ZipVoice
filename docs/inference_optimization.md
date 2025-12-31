# ZipVoice推論最適化ガイド

## 概要

本ドキュメントでは、ZipVoiceの推論速度を最適化するための各種アプローチとベンチマーク結果を説明します。

## ベンチマーク環境

| 項目 | 内容 |
|-----|------|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| OS | Windows |
| Python | 3.11 |
| PyTorch | 2.x (CUDA対応) |

テスト条件：
- プロンプト音声: 約3秒
- 生成テキスト: "The quick brown fox jumps over the lazy dog."
- 生成音声: 約16秒
- 計測回数: 3回の平均

## ベンチマーク結果

### 速度比較サマリー

| 設定 | モデル推論 | 合計時間 | RTF | 高速化 |
|-----|-----------|---------|-----|-------|
| zipvoice 16-step (ベースライン) | 48395 ms | 48579 ms | 3.03 | 1.00x |
| zipvoice 8-step | 4498 ms | 4563 ms | 0.29 | **10.65x** |
| zipvoice 4-step | 1919 ms | 2475 ms | 0.16 | **19.63x** |
| zipvoice_distill 8-step | 1937 ms | 2494 ms | 0.16 | **19.48x** |
| zipvoice_distill 4-step | 488 ms | 1030 ms | 0.06 | **47.15x** |
| zipvoice_distill 2-step | 293 ms | 896 ms | 0.06 | **54.24x** |

**RTF (Real-Time Factor)**: 1秒の音声を生成するのに必要な時間。RTF < 1.0 でリアルタイムより高速。

### 重要な発見

1. **ODE Steps削減が最も効果的**: 16→8ステップで約10倍、16→4ステップで約20倍高速化
2. **蒸留モデルでさらに高速化**: zipvoice_distill + 2-stepで最大54倍高速化
3. **CFG (Classifier-Free Guidance) の影響**: 通常モデルはCFG計算でバッチが2倍化されるため遅い
4. **蒸留モデルの利点**: CFGを単一推論で計算できるため、同じステップ数でも高速

## 最適化アプローチ

### 1. ODE Steps削減（推奨）

最も簡単で効果的な最適化方法です。

```bash
# デフォルト（16ステップ）
uv run python -m zipvoice.bin.infer_zipvoice --num-step 16 ...

# 高速化（8ステップ）- 約10倍高速、品質良好
uv run python -m zipvoice.bin.infer_zipvoice --num-step 8 ...

# 最高速（4ステップ）- 約20倍高速、品質許容範囲
uv run python -m zipvoice.bin.infer_zipvoice --num-step 4 ...
```

### 2. 蒸留モデル使用（推奨）

蒸留モデルはCFG計算が効率化されており、同じステップ数でも高速です。

```bash
# 蒸留モデル（デフォルト設定）
uv run python -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice_distill \
    --num-step 8 \
    --guidance-scale 3.0 \
    ...

# 蒸留モデル（高速設定）- 約47倍高速
uv run python -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice_distill \
    --num-step 4 \
    --guidance-scale 3.0 \
    ...

# 蒸留モデル（最高速設定）- 約54倍高速
uv run python -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice_distill \
    --num-step 2 \
    --guidance-scale 3.0 \
    ...
```

### 3. TensorRT（GPUでさらに高速化）

TensorRTエンジンを使用すると、さらに約2倍の高速化が期待できます。

```bash
# ONNXエクスポート
uv run python -m zipvoice.bin.onnx_export \
    --model-name zipvoice_distill \
    --model-dir models/zipvoice_distill \
    --onnx-model-dir models/zipvoice_distill_onnx

# TensorRTエンジン作成
uv run python -m zipvoice.bin.tensorrt_export \
    --model-name zipvoice_distill \
    --model-dir models/zipvoice_distill \
    --tensorrt-model-dir models/zipvoice_distill_trt

# TensorRTで推論
uv run python -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice_distill \
    --trt-engine-path models/zipvoice_distill_trt/fm_decoder.fp16.plan \
    ...
```

## 推奨設定

### 高品質優先

```bash
--model-name zipvoice --num-step 16 --guidance-scale 1.0
```
- RTF: 約3.0（リアルタイムの3倍の時間が必要）

### バランス（推奨）

```bash
--model-name zipvoice_distill --num-step 8 --guidance-scale 3.0
```
- RTF: 約0.16（リアルタイムの約6倍高速）
- 高速化: 約20倍

### 高速優先

```bash
--model-name zipvoice_distill --num-step 4 --guidance-scale 3.0
```
- RTF: 約0.06（リアルタイムの約16倍高速）
- 高速化: 約47倍

### 最高速

```bash
--model-name zipvoice_distill --num-step 2 --guidance-scale 3.0
```
- RTF: 約0.06（リアルタイムの約17倍高速）
- 高速化: 約54倍
- 注意: 品質低下の可能性あり

## 技術的背景

### なぜ通常モデルが遅いのか

ZipVoiceはClassifier-Free Guidance (CFG)を使用しています。通常モデルでは：

1. 条件付き推論と無条件推論を別々に計算
2. バッチサイズが2倍になる
3. 各ステップで2回の順伝播が必要

```python
# solver.py での実装
if (guidance_scale != 0.0):
    x = torch.cat([x] * 2, dim=0)  # バッチ2倍化
    # ... 条件付き・無条件の両方を計算
    res = (1 + guidance_scale) * data_cond - guidance_scale * data_uncond
```

### なぜ蒸留モデルが速いのか

蒸留モデルは：

1. CFGを単一推論で計算できるように訓練されている
2. guidance_scaleをモデルの入力として渡す
3. バッチサイズの2倍化が不要

```python
# DistillDiffusionModel での実装
return self.model_func(
    t=t, xt=x,
    guidance_scale=guidance_scale,  # 直接渡す
    ...
)
```

## ベンチマークスクリプト

ベンチマークスクリプトは `scripts/benchmark_optimization.py` にあります：

```bash
uv run python scripts/benchmark_optimization.py \
    --prompt-wav data/prompt.wav \
    --prompt-text "Hello, this is a test." \
    --text "The quick brown fox jumps over the lazy dog." \
    --num-runs 3 \
    --output-json results/benchmark.json
```

## 結論

1. **最も効果的な最適化**: 蒸留モデル + ステップ数削減（最大54倍高速化）
2. **品質とのトレードオフ**: ステップ数を減らすほど品質は低下するが、4ステップ程度まではほぼ問題なし
3. **リアルタイム推論**: zipvoice_distill + 4-step以下でRTF < 0.1を達成
4. **さらなる高速化**: TensorRT FP16でさらに約2倍の高速化が期待できる
