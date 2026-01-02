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

#### 問題の背景
ZipVoiceをMoeSpeech-20speakers（20話者の日本語データセット）でファインチューニングする際、以下のトレードオフが発生：
- **日本語アクセント**: FM Decoderを学習させると改善
- **話者類似度**: FM Decoderを学習させると悪化（Catastrophic Forgetting）

オリジナルモデルは数千話者で訓練されており、20話者のみでファインチューニングするとfm_decoderが過適合し、話者汎化能力が失われる。

### FM Decoder構造分析
```
FM Decoder (118M params, 99.7%)
├── Stack 0: Zipformer2Encoder (15M) - 前半
├── Stack 1: DownsampledZipformer2Encoder (15M) - 前半
├── Stack 2: DownsampledZipformer2Encoder (29M) - 後半
├── Stack 3: DownsampledZipformer2Encoder (29M) - 後半
└── Stack 4: Zipformer2Encoder (30M) - 後半
```

### 実験結果一覧

| 試行 | 設定 | 凍結量 | Val Loss | 日本語アクセント | 話者類似度 |
|------|------|--------|----------|-----------------|-----------|
| 1 | 凍結なし | 0% | - | ○ 良い | × 悪い |
| 2 | FM Decoder完全凍結 | 100% | - | × 悪い | ○ 良い |
| 3 | Stack 2,3,4凍結 | 72% | 0.0630 | × 悪い | ? |
| 4 | Stack 4のみ凍結 | 25% | 0.0628 | ? | × 悪い |

#### 試行1: 通常のファインチューニング（凍結なし）
```bash
uv run python -m zipvoice.bin.train_zipvoice \
    --finetune 1 --checkpoint download/zipvoice/model.pt \
    --tokenizer japanese --num-iters 50000 \
    --exp-dir exp/zipvoice_moe_90h
```
**結果**:
- 日本語発音・アクセント: **良好** - 自然な日本語として聞こえる
- ゼロショット話者類似度: **悪い** - プロンプト話者の声色が反映されない

**原因**: FM Decoderが20話者に過適合し、話者汎化能力が失われた（Catastrophic Forgetting）

#### 試行2: FM Decoder完全凍結
```bash
uv run python -m zipvoice.bin.train_zipvoice \
    --finetune 1 --checkpoint download/zipvoice/model.pt \
    --freeze-fm-decoder 1 \
    --tokenizer japanese --num-iters 30000 \
    --exp-dir exp/zipvoice_japanese_frozen
```
**結果**:
- 日本語発音・アクセント: **悪い** - 海外の人が話したような不自然なアクセント
- ゼロショット話者類似度: **良い**（推定）

**原因**: FM Decoderを完全に凍結したため、日本語のアクセントパターンを学習できなかった。Text Encoder + Embeddingだけでは日本語の韻律を表現しきれない。

#### 試行3: 部分凍結（Stack 2,3,4凍結）
```bash
uv run python -m zipvoice.bin.train_zipvoice \
    --finetune 1 --checkpoint download/zipvoice/model.pt \
    --freeze-fm-decoder-stacks "2,3,4" \
    --tokenizer japanese --num-iters 15000 \
    --base-lr 0.01 \
    --exp-dir exp/zipvoice_japanese_partial
```
**設定**:
- 凍結: Stack 2,3,4 (88M params, 72%)
- 学習可能: Stack 0,1 + embed + text_encoder (34M params, 28%)

**結果**:
- 日本語発音・アクセント: **悪い** - まだ不自然
- ゼロショット話者類似度: 未評価

**原因**: 凍結量が多すぎて日本語アクセントを学習できなかった。

#### 試行4: Stack 4のみ凍結
```bash
uv run python -m zipvoice.bin.train_zipvoice \
    --finetune 1 --checkpoint download/zipvoice/model.pt \
    --freeze-fm-decoder-stacks "4" \
    --tokenizer japanese --num-iters 15000 \
    --base-lr 0.01 \
    --exp-dir exp/zipvoice_japanese_stack4only
```
**設定**:
- 凍結: Stack 4のみ (30M params, 25%)
- 学習可能: Stack 0,1,2,3 + embed + text_encoder (92M params, 75%)

**結果**:
- 日本語発音・アクセント: 未評価
- ゼロショット話者類似度: **悪い** - プロンプト話者の声色が反映されない

**原因**: 凍結量が少なすぎて話者汎化能力が失われた。

### 実装した機能

#### --freeze-fm-decoder オプション
FM Decoder全体を凍結するオプション。
```python
parser.add_argument(
    "--freeze-fm-decoder",
    type=str2bool,
    default=False,
    help="Freeze the FM decoder to preserve speaker similarity.",
)
```

#### --freeze-fm-decoder-stacks オプション
特定のスタックのみを凍結するオプション。
```python
parser.add_argument(
    "--freeze-fm-decoder-stacks",
    type=str,
    default="",
    help="Comma-separated stack indices to freeze (e.g., '2,3,4'). "
         "Use this for partial freezing to balance Japanese accent learning and speaker similarity.",
)
```

### 結論と今後の方針

**凍結アプローチの限界**:
凍結量を調整するだけでは、日本語アクセントと話者類似度の両立が困難。
- 凍結を増やす → 日本語アクセントが悪化
- 凍結を減らす → 話者類似度が悪化

**次のアプローチ: LoRA（Low-Rank Adaptation）**

LoRAを使用することで、以下が期待できる：
1. 元のFM Decoderの重みを**完全に保持**（話者類似度維持）
2. 小さな追加パラメータ（LoRAアダプター）のみ学習（日本語アクセント学習）
3. 推論時にLoRA重みをマージ可能（追加コストなし）

**実装予定**:
```python
# LoRAレイヤーをFM Decoderに追加
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,  # ランク
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # 注意機構のみ
    lora_dropout=0.1,
)
model.fm_decoder = get_peft_model(model.fm_decoder, lora_config)
```

### チェックポイント
- `exp/zipvoice_japanese_stack4only/`: Stack 4のみ凍結モデル（現在のベスト Val Loss: 0.0628）
