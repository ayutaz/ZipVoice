# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

ZipVoiceは、Flow Matchingを用いた高速・高品質なゼロショットText-to-Speech（TTS）システムです。123Mパラメータの軽量モデルで、中国語と英語に対応しています。

### モデルバリアント
- **ZipVoice**: 基本モデル（単一話者TTS）
- **ZipVoice-Distill**: 蒸留版（高速化、品質維持）
- **ZipVoice-Dialog**: 対話生成モデル（モノラル）
- **ZipVoice-Dialog-Stereo**: 対話生成モデル（ステレオ）

## 開発コマンド

### 依存関係のインストール
```bash
uv sync
uv pip install piper-phonemize --find-links https://k2-fsa.github.io/icefall/piper_phonemize.html
uv pip install -r requirements_eval.txt  # 評価用
```

### k2のインストール（訓練に必須）
```bash
# PyTorch・CUDAバージョンに合わせてインストール
uv pip install k2==1.24.4.dev20250208+cuda12.1.torch2.5.1 -f https://k2-fsa.github.io/k2/cuda.html
```

### 推論
```bash
# 単一文の生成
uv run python -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --prompt-wav prompt.wav \
    --prompt-text "プロンプト音声の書き起こし" \
    --text "合成するテキスト" \
    --res-wav-path result.wav

# バッチ推論
uv run python -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --test-list test.tsv \
    --res-dir results

# 対話生成
uv run python -m zipvoice.bin.infer_zipvoice_dialog \
    --model-name zipvoice_dialog \
    --test-list test.tsv \
    --res-dir results
```

### 訓練
```bash
# 単一話者TTS訓練
uv run python -m zipvoice.bin.train_zipvoice \
    --world-size 8 --use-fp16 1 --num-epochs 11 \
    --model-config conf/zipvoice_base.json \
    --tokenizer emilia --token-file data/tokens_emilia.txt \
    --dataset emilia --manifest-dir data/fbank \
    --exp-dir exp/zipvoice

# 蒸留訓練
uv run python -m zipvoice.bin.train_zipvoice_distill \
    --world-size 8 --use-fp16 1 --num-iters 60000 \
    --teacher-model path/to/teacher.pt \
    --exp-dir exp/zipvoice_distill
```

### モデルエクスポート
```bash
# 標準ONNXエクスポート
uv run python -m zipvoice.bin.onnx_export \
    --model-name zipvoice \
    --model-dir exp/zipvoice \
    --onnx-model-dir exp/zipvoice_onnx

# Unity Sentis用ONNXエクスポート（推奨）
uv run python -m zipvoice.bin.onnx_export_sentis \
    --model-name zipvoice \
    --model-dir exp/zipvoice \
    --onnx-model-dir exp/zipvoice_sentis

# TensorRTエクスポート
uv run python -m zipvoice.bin.tensorrt_export [args...]

# Vocos Vocoder ONNXダウンロード（Unity用）
uv run python scripts/download_vocos_onnx.py

# Sentis互換性検証
uv run python -m zipvoice.bin.verify_sentis_onnx \
    --onnx-dir exp/zipvoice_sentis
```

### テスト
```bash
uv run pytest              # 全テスト実行
uv run pytest -v           # 詳細出力
uv run pytest --cov        # カバレッジ付き
```

### コードフォーマット
```bash
black .
isort .
```

## アーキテクチャ

### ディレクトリ構成
```
zipvoice/
├── bin/           # 実行スクリプト（訓練・推論・エクスポート）
├── models/        # モデル定義（ZipVoice, Flow Matchingデコーダ）
├── tokenizer/     # テキストトークナイザ（Emilia/LibriTTS/Espeak）
├── dataset/       # データセット処理（Lhotse CutSetベース）
├── eval/          # 評価メトリクス
└── utils/         # ユーティリティ

egs/               # 訓練レシピ
runtime/           # 本番デプロイメント（NVIDIA Triton）
```

### モデル構造

| コンポーネント | 役割 | 次元 |
|--------------|------|-----|
| Text Embedding | トークン→ベクトル | → 192 |
| Text Encoder | テキスト条件生成 | → 100 |
| FM Decoder | 速度場予測 | → 100 |
| Euler Solver | ODE積分 | ノイズ→音声 |

### Flow Matching

- **訓練**: `xt = features*t + noise*(1-t)`、MSE損失で速度場を学習
- **推論**: Euler法でnum_step回積分（デフォルト16ステップ）
- **CFG**: `v = (1+s)*v_cond - s*v_uncond`

### 処理フロー

```
テキスト → トークナイザ → Text Encoder → FM Decoder → Vocoder → 波形(24kHz)
```

### 主要コンポーネント

**モデル**: `ZipVoice`, `ZipVoiceDistill`, `ZipVoiceDialog`, `ZipVoiceDialogStereo`

**トークナイザ**: `EmiliaTokenizer`(中国語), `LibriTTSTokenizer`(英語), `EspeakTokenizer`(多言語)

**推論バックエンド**: PyTorch / ONNX / TensorRT / Triton / Unity Sentis

## 重要な実装詳細

### テキスト内の特殊記法
- `<pinyin>`: 中国語ピンインを直接指定（例：`<chang2>`）
- `[tag]`: 特殊タグ
- `[S1]`/`[S2]`: 対話モードでの話者識別

### 設定ファイル
- モデル設定: `egs/zipvoice/conf/zipvoice_base.json`
- サンプルレート: 24kHz
- 特徴量: Vocos fbank（100次元）

### 訓練レシピ
完全な訓練パイプラインは `egs/zipvoice/run_emilia.sh` を参照：
- Stage 1: データ準備
- Stage 2-3: ZipVoice訓練 + チェックポイント平均化
- Stage 4-6: 蒸留（2段階）
- Stage 7-8: ONNXエクスポート
- Stage 9-12: 推論と評価

## Unity Sentis対応

### 概要
ZipVoiceはUnity Sentis（AIランタイム）でリアルタイム推論が可能です。

### エクスポートファイル
| ファイル | 内容 |
|----------|------|
| `text_encoder.onnx` | テキスト→条件ベクトル |
| `fm_decoder.onnx` | Flow Matchingデコーダ |
| `vocos.onnx` | Vocoder（HuggingFaceから取得） |

### 制約事項
- **Opset version**: 7-15（推奨: 15）
- **テンソル次元**: 最大8次元
- **未サポート演算子**: `log1p`（`log(1+x)`で代替済み）

### Unity側で必要な実装
- **EulerSolver**: ODE積分（5-16ステップ）
- **ISTFT**: 逆フーリエ変換（n_fft=1024, hop_length=256）
- **Tokenizer**: テキスト→トークン変換

### 関連スクリプト
- `zipvoice/bin/onnx_export_sentis.py`: Sentis用エクスポート
- `zipvoice/bin/verify_sentis_onnx.py`: 互換性検証
- `scripts/download_vocos_onnx.py`: Vocos ONNXダウンロード
