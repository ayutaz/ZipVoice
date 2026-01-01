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
# 通常のONNXエクスポート
uv run python -m zipvoice.bin.onnx_export \
    --model-name zipvoice \
    --model-dir exp/zipvoice_moe_90h \
    --checkpoint-name best-valid-loss.pt \
    --onnx-model-dir exp/zipvoice_onnx

# Unity Sentis互換ONNXエクスポート（Ifオペレーター除去）
uv run python -m zipvoice.bin.onnx_export \
    --model-name zipvoice \
    --model-dir exp/zipvoice_moe_90h \
    --checkpoint-name best-valid-loss.pt \
    --onnx-model-dir exp/zipvoice_onnx_unity \
    --unity-sentis 1 \
    --fm-seq-len 512

# TensorRTエクスポート
uv run python -m zipvoice.bin.tensorrt_export [args...]
```

**Unity Sentisモードのオプション**:
- `--unity-sentis 1`: Ifオペレーターを除去（Unity Sentis必須）
- `--fm-seq-len N`: FM Decoderの固定シーケンス長（デフォルト: 200）

詳細は [docs/unity_sentis_onnx.md](docs/unity_sentis_onnx.md) を参照。

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

**トークナイザ**: `EmiliaTokenizer`(中国語), `LibriTTSTokenizer`(英語), `EspeakTokenizer`(多言語), `JapaneseTokenizer`(日本語)

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

## 日本語対応プロジェクト

### 目標
ZipVoiceの軽量・高速という利点を維持しながら、日本語に対応する。最終的にONNXエクスポートしてUnityで動作させる。

### データセット
- **MoeSpeech-20speakers-ljspeech**: 20話者の日本語音声データセット（28GB）
- **つくよみちゃん**: ゼロショットテスト用（未学習話者）

### 日本語トークナイザ
- `pyopenjtalk-plus`を使用
- アクセントマーカー対応: `[H]`(高), `[L]`(低), `|`(アクセント境界), `[Q]`(促音)
- トークンファイル: `data/tokens_japanese_extended.txt`

### 学習の経緯と課題

#### 試行1: 通常のファインチューニング（凍結なし）
```bash
uv run python -m zipvoice.bin.train_zipvoice \
    --finetune 1 --checkpoint download/zipvoice/model.pt \
    --tokenizer japanese --num-iters 50000 ...
```
**結果**:
- 日本語発音・アクセント: 良好
- ゼロショット話者類似度: 悪い（Catastrophic Forgetting）

**原因**: オリジナルモデルは数千話者で訓練されていたが、20話者のみでファインチューニングしたためfm_decoderが過適合し、話者汎化能力が失われた。

#### 試行2: FM Decoder完全凍結
```bash
uv run python -m zipvoice.bin.train_zipvoice \
    --finetune 1 --checkpoint download/zipvoice/model.pt \
    --freeze-fm-decoder 1 --tokenizer japanese --num-iters 30000 ...
```
**結果**:
- 日本語発音・アクセント: 悪い（海外の人が話したような不自然さ）
- ゼロショット話者類似度: 改善（予想）

**原因**: fm_decoderを完全に凍結したため、日本語のアクセントパターンを学習できなかった。

### FM Decoder構造分析
```
FM Decoder (118M params, 99.7%)
├── Stack 0: Zipformer2Encoder (15M) - 前半
├── Stack 1: DownsampledZipformer2Encoder (15M) - 前半
├── Stack 2: DownsampledZipformer2Encoder (29M) - 後半
├── Stack 3: DownsampledZipformer2Encoder (29M) - 後半
└── Stack 4: Zipformer2Encoder (30M) - 後半
```

### 今後のアプローチ: 部分凍結

**戦略**: 後半スタック（2,3,4）のみ凍結、前半スタック（0,1）は学習可能に

| レイヤー | 凍結 | 役割 |
|---------|------|------|
| Stack 0, 1 | 学習 | 日本語アクセント・韻律を学習 |
| Stack 2, 3, 4 | 凍結 | 話者特徴の使い方を保持 |

**実装予定**:
```python
# train_zipvoice.pyに追加
parser.add_argument("--freeze-fm-decoder-stacks", type=str, default="",
    help="Comma-separated stack indices to freeze (e.g., '2,3,4')")

# 凍結ロジック
if params.freeze_fm_decoder_stacks:
    stacks_to_freeze = [int(x) for x in params.freeze_fm_decoder_stacks.split(",")]
    for i in stacks_to_freeze:
        for param in model.fm_decoder.encoders[i].parameters():
            param.requires_grad = False
```

**訓練コマンド**:
```bash
uv run python -m zipvoice.bin.train_zipvoice \
    --finetune 1 --checkpoint download/zipvoice/model.pt \
    --freeze-fm-decoder-stacks "2,3,4" \
    --tokenizer japanese --num-iters 30000 \
    --base-lr 0.01 ...
```

### チェックポイント
- `exp/zipvoice_japanese_frozen/`: FM Decoder完全凍結モデル
- `exp/zipvoice_japanese_partial/`: 部分凍結モデル（予定）

### 代替案（部分凍結で不十分な場合）
1. **LoRA実装**: fm_decoderにLoRAアダプターを追加
2. **混合データ訓練**: 英語/中国語データと混ぜて訓練（話者数を増やす）
3. **段階的解凍**: 凍結→低学習率で解凍して追加訓練
