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
uv run python -m zipvoice.bin.onnx_export [args...]
uv run python -m zipvoice.bin.tensorrt_export [args...]
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
├── models/        # モデル定義（ZipVoice, ZipformerベースのFlow Matchingデコーダ）
├── tokenizer/     # テキストトークナイザ（言語別：Emilia/LibriTTS/Espeak）
├── dataset/       # データセット処理（Lhotse CutSetベース）
├── eval/          # 評価メトリクス（話者類似度、MOS、WER）
└── utils/         # ユーティリティ（学習率スケジューラ、チェックポイント管理等）

egs/               # 訓練レシピとサンプルスクリプト
runtime/           # 本番デプロイメント（NVIDIA Triton）
```

### 主要コンポーネント

**モデル (`zipvoice/models/`)**
- `ZipVoice`: テキストエンコーダ（4層Zipformer）+ Flow Matchingデコーダ
- `EulerSolver`: ODE求解器（Flow Matching用）
- `Zipformer`/`ZipformerTwoStream`: 効率的なTransformerアーキテクチャ

**トークナイザ (`zipvoice/tokenizer/`)**
- `EmiliaTokenizer`: 中国語（ピンイン変換）
- `LibriTTSTokenizer`: 英語（音素ベース）
- `EspeakTokenizer`: 多言語対応（Espeak音素）
- テキスト正規化：数字変換、句読点処理

**データ処理**
- Lhotseライブラリによるマニフェスト管理
- Vocos fbankによる音声特徴量抽出（24kHz、100次元）

**訓練インフラ**
- 分散訓練（DDP）対応
- FP16混合精度
- ScaledAdamオプティマイザ
- Eden/FixedLRScheduler

### 推論バックエンド
1. **PyTorch**: ネイティブ推論
2. **ONNX**: CPU向け（`infer_zipvoice_onnx`）、INT8量子化対応
3. **TensorRT**: GPU高速化（約2倍のスループット）
4. **Triton**: 本番サーバー（gRPC/HTTP API）

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
