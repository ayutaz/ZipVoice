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
- **moe-speech-plus**: 473話者の日本語音声データセット（680GB、623時間）- 現在使用中
- **MoeSpeech-20speakers-ljspeech**: 20話者の日本語音声データセット（28GB）- 過適合のため非推奨
- **つくよみちゃん**: ゼロショットテスト用（未学習話者）

### 前処理スクリプト
- `scripts/preprocess_moe_speech_plus.py`: moe-speech-plus用TSV生成スクリプト
- `scripts/preprocess_moe_speech_plus.sh`: 完全な前処理パイプライン

### 日本語トークナイザ
- `pyopenjtalk-plus`を使用
- アクセントマーカー対応: `[H]`(高), `[L]`(低), `|`(アクセント境界), `[Q]`(疑問文)
- トークンファイル: `data/tokens_japanese_extended.txt`

#### G2P修正（2026-01-06）
アクセント核の判定条件を修正：
```python
# 旧: a1_val < 0  → 新: a1_val <= 0
level = "H" if a1_val <= 0 else "L"
```
- **変更理由**: アクセント核（A1=0）も高ピッチ領域に含めるべき
- **効果**: より自然な日本語アクセントパターンの表現
- **テスト**: 24/24 PASSED（`tests/test_japanese_tokenizer.py`）

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
| 5 | DoRA (LoRA+重み分解) | 4.66% | 0.0631 | △ 中途半端 | △ 中途半端 |

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

#### 試行5: DoRA（Weight-Decomposed LoRA）
```bash
uv run python -m zipvoice.bin.train_zipvoice \
    --finetune 1 --checkpoint download/zipvoice/model.pt \
    --use-lora 1 --lora-rank 32 --lora-alpha 64 \
    --use-dora 1 --lora-dropout 0.05 \
    --base-lr 0.001 --num-iters 10000 \
    --tokenizer japanese \
    --exp-dir exp/zipvoice_japanese_dora
```
**設定**:
- 方式: DoRA (Weight-Decomposed LoRA) - LoRAより高精度
- ターゲット: FM Decoder Attention layers (`self_attn_weights.in_proj`, `self_attn1`, `self_attn2`)
- 訓練パラメータ: 5,783,656 / 124,240,932 (4.66%)
- rank=32, alpha=64, dropout=0.05

**結果**:
- 日本語発音・アクセント: **中途半端** - 自然でも不自然でもない
- ゼロショット話者類似度: **中途半端** - プロンプト話者に似ているとは言えない

**原因**: LoRA/DoRAでは十分な表現力がなく、ゼロショット性能と日本語アクセントの両立は困難。

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

**ゼロショットアプローチの限界（20話者での実験）**:
凍結・部分凍結・LoRA/DoRAのいずれの手法でも、ゼロショット話者類似度と日本語アクセントの両立は困難であった。

| アプローチ | 日本語アクセント | 話者類似度 |
|-----------|-----------------|-----------|
| 凍結なし | ○ | × |
| 完全凍結 | × | ○ |
| 部分凍結 | × | × |
| DoRA | △ | △ |

**原因分析**: 20話者では話者の多様性が不足し、FM Decoderが過適合してしまう。

---

## 試行6: 473話者での大規模訓練（moe-speech-plus）

### 仮説
20話者での過適合問題は、**話者数の不足**が原因。473話者のmoe-speech-plusで訓練すれば、ゼロショット性能を維持しながら日本語対応が可能になるはず。

### データセット: moe-speech-plus

| 項目 | 値 |
|------|-----|
| 話者数 | 473 |
| サンプル数 | 395,170（Train: 375,412 / Dev: 19,758） |
| 総時間 | 約623時間 |
| サイズ | 約680GB |
| 音声長 | 2〜15秒（平均5.66秒） |
| サンプルレート | 44.1kHz → 24kHzにリサンプル |

### 前処理パイプライン

```bash
# 1. TSV生成
python scripts/preprocess_moe_speech_plus.py \
  --input-dir /data/moe-speech-plus \
  --output-dir data/raw \
  --dev-ratio 0.05

# 2. Manifest作成（24kHzリサンプル含む）
uv run python -m zipvoice.bin.prepare_dataset \
  --tsv-path data/raw/moe_speech_plus_train.tsv \
  --prefix moe_speech_plus --subset train --num-jobs 16 --output-dir data/manifests

# 3. Fbank特徴量計算
uv run python -m zipvoice.bin.compute_fbank \
  --source-dir data/manifests --dest-dir data/fbank \
  --dataset moe_speech_plus --subset train --num-jobs 16

# 4. トークン化
uv run python -m zipvoice.bin.pretokenize_manifest \
  --input-manifest data/fbank/moe_speech_plus_cuts_train.jsonl.gz \
  --output-manifest data/fbank/moe_speech_plus_cuts_train_tokenized.jsonl.gz \
  --tokenizer japanese --token-file data/tokens_japanese_extended.txt --num-jobs 16
```

### 訓練コマンド

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python -m zipvoice.bin.train_zipvoice \
  --world-size 1 \
  --use-fp16 1 \
  --num-epochs 20 \
  --finetune 1 \
  --checkpoint download/zipvoice/model.pt \
  --model-config egs/zipvoice/conf/zipvoice_base.json \
  --tokenizer japanese \
  --token-file data/tokens_japanese_extended.txt \
  --dataset custom \
  --train-manifest data/fbank/moe_speech_plus_cuts_train_tokenized.jsonl.gz \
  --dev-manifest data/fbank/moe_speech_plus_cuts_dev_tokenized.jsonl.gz \
  --base-lr 0.01 \
  --max-duration 100 \
  --num-workers 4 \
  --exp-dir exp/zipvoice_japanese_473speakers
```

### 重要: OOMエラーとmax-durationの関係

#### 問題
`max_duration`を上げるとOOM（Out of Memory）エラーが発生する。

#### 原因
`max_duration`は**バッチ内の合計秒数**を制限するが、**メモリ使用量は系列長の二乗に比例**する。

```
Attention機構のメモリ計算量: O(T² × d)
- 5秒のサンプル: 469 frames → 0.2M演算
- 15秒のサンプル: 1406 frames → 2.0M演算（9倍）
```

DynamicBucketingSamplerは似た長さのサンプルをまとめるため、長いサンプルが集中したバッチでOOMが発生する。

| 同じmax_duration=100でも | バッチサイズ | メモリ係数 |
|------------------------|------------|----------|
| 短いサンプル（5秒×20個） | 20 | 500 |
| 長いサンプル（15秒×6.7個） | 6-7 | 1500（3倍） |

#### 推奨設定（NVIDIA L4 24GB）

| max_duration | 安定性 | 訓練速度 |
|--------------|-------|---------|
| 100 | ◎ 安定 | 遅い |
| 150 | △ 時々OOM | 中程度 |
| 200+ | × 高確率でOOM | - |

**結論**: NVIDIA L4 24GBでは`max-duration=100`が安全。

#### 代替OOM対策

1. **max_lenを制限**: 長いサンプルを除外（例: `--max-len 10`で10秒以上を除外）
2. **gradient checkpointing**: メモリ削減（訓練速度20-30%低下）
3. **勾配累積**: 実効バッチサイズを維持しつつメモリ削減

### 現在の訓練状況（2026-01-06）

**注意**: G2P修正（A1<=0判定）により、旧トークンで訓練されたv1は廃止。新G2Pでv2を開始。

| 項目 | 値 |
|------|-----|
| 実験ディレクトリ | `exp/zipvoice_japanese_473speakers_v2/` |
| 状態 | 新規開始（G2P修正後の再訓練） |
| 設定 | max-duration=100, base-lr=0.01, FP16 |
| 推定訓練時間 | 約5.5日（130時間） |

**廃止**: `exp/zipvoice_japanese_473speakers/`（旧G2Pのトークンで訓練、整合性なし）

### 訓練開始/再開コマンド

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python -m zipvoice.bin.train_zipvoice \
  --world-size 1 \
  --use-fp16 1 \
  --num-epochs 20 \
  --finetune 1 \
  --checkpoint download/zipvoice/model.pt \
  --model-config egs/zipvoice/conf/zipvoice_base.json \
  --tokenizer japanese \
  --token-file data/tokens_japanese_extended.txt \
  --dataset custom \
  --train-manifest data/fbank/moe_speech_plus_cuts_train_tokenized.jsonl.gz \
  --dev-manifest data/fbank/moe_speech_plus_cuts_dev_tokenized.jsonl.gz \
  --base-lr 0.01 \
  --max-duration 100 \
  --num-workers 4 \
  --exp-dir exp/zipvoice_japanese_473speakers_v2
```

**訓練再開時**: `--checkpoint`を最新チェックポイント（例: `exp/zipvoice_japanese_473speakers_v2/checkpoint-5000.pt`）に変更

### 評価予定
訓練完了後、つくよみちゃん（訓練に含まれない話者）でゼロショット評価を実施。

---

## 過去の方針（参考）

**方針転換案: 話者特化ファインチューニング（2段階）**

ゼロショットを諦め、特定話者への適応を行う2段階アプローチ：

```
[オリジナルモデル] → [Stage 1: 日本語継続学習] → [Stage 2: 話者特化学習]
   (中国語/英語)         (MoeSpeech 60K)           (つくよみちゃん 100)
```

**Stage 1: 日本語継続事前学習**
- データ: MoeSpeech 20話者（約60,000サンプル）
- 目的: 日本語の発音・アクセントパターンを学習
- 方式: 全パラメータのファインチューニング

**Stage 2: 話者特化ファインチューニング**
- データ: つくよみちゃん（100サンプル）
- 目的: 特定話者の声色に適応
- 方式: LoRA/DoRAで日本語能力を保持しつつ話者適応

※ 473話者での訓練結果次第で、この方針は不要になる可能性あり。

### 実装済み機能

#### LoRA/DoRAサポート
`zipvoice/models/modules/lora_utils.py` を新規作成：
```python
from zipvoice.models.modules.lora_utils import (
    create_lora_config,
    apply_lora_to_fm_decoder,
    merge_lora_weights,
    save_lora_weights,
    load_lora_weights,
)
```

訓練スクリプトのLoRAオプション:
- `--use-lora`: LoRA有効化
- `--lora-rank`: ランク（デフォルト32）
- `--lora-alpha`: スケーリング係数（デフォルト64）
- `--use-dora`: DoRA有効化（デフォルトTrue）
- `--lora-dropout`: ドロップアウト率（デフォルト0.05）

### チェックポイント
- `exp/zipvoice_japanese_473speakers_v2/`: **473話者訓練（新G2P）** - 訓練予定
- `exp/zipvoice_japanese_473speakers/`: 473話者訓練（旧G2P）- **廃止**（G2P修正前のトークン）
- `exp/zipvoice_japanese_dora/`: DoRAモデル（Val Loss: 0.0631）
- `exp/zipvoice_japanese_stack4only/`: Stack 4のみ凍結モデル（Val Loss: 0.0628）
- `exp/zipvoice_moe_90h/`: 20話者訓練モデル（ゼロショット性能なし）
