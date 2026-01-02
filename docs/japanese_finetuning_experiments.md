# 日本語ファインチューニング実験記録

このドキュメントは、ZipVoiceの日本語対応における全ての実験と結果を詳細に記録したものです。

## 目次

1. [背景と目標](#背景と目標)
2. [データセット](#データセット)
3. [問題の発見](#問題の発見)
4. [実験一覧](#実験一覧)
5. [詳細な実験記録](#詳細な実験記録)
6. [実装した機能](#実装した機能)
7. [結論と今後の方針](#結論と今後の方針)

---

## 背景と目標

### 目標
ZipVoiceの軽量・高速という利点を維持しながら、日本語に対応する。最終的にONNXエクスポートしてUnityで動作させる。

### オリジナルモデルの特性
- パラメータ数: 123M
- 対応言語: 中国語、英語
- 訓練データ: 数千話者の大規模データセット
- ゼロショット話者クローン: 優れた性能

---

## データセット

### 訓練データ: MoeSpeech-20speakers-ljspeech
- 話者数: 20話者
- データ量: 約28GB
- フォーマット: LJSpeech形式
- パス: `/data/moe-speech-20speakers-ljspeech/`

### テストデータ: つくよみちゃん
- 用途: ゼロショットテスト（未学習話者）
- パス: `/data/tsukuyomi-chan-ljspeech/`
- 100発話のサンプル音声

---

## 問題の発見

### Catastrophic Forgetting（破滅的忘却）

ZipVoiceを20話者のみでファインチューニングすると、以下のトレードオフが発生：

| 要素 | 説明 |
|------|------|
| 日本語アクセント | FM Decoderを学習させると改善 |
| 話者類似度 | FM Decoderを学習させると悪化 |

**原因分析**:
- オリジナルモデルは数千話者で訓練され、話者汎化能力が高い
- 20話者のみでファインチューニングすると、FM Decoderが少数話者に過適合
- 結果として、ゼロショット時に話者の声色を正しく再現できなくなる

---

## 実験一覧

| 試行 | 日付 | 設定 | 凍結量 | Val Loss | 日本語アクセント | 話者類似度 | 結論 |
|------|------|------|--------|----------|-----------------|-----------|------|
| 1 | 2024-12 | 凍結なし | 0% | - | ○ 良い | × 悪い | 話者類似度に問題 |
| 2 | 2025-01-01 | FM Decoder完全凍結 | 100% | - | × 悪い | ○ 良い | アクセントに問題 |
| 3 | 2025-01-01 | Stack 2,3,4凍結 | 72% | 0.0630 | × 悪い | ? | アクセントに問題 |
| 4 | 2025-01-02 | Stack 4のみ凍結 | 25% | 0.0628 | ? | × 悪い | 話者類似度に問題 |

---

## 詳細な実験記録

### 試行1: 通常のファインチューニング（凍結なし）

#### 設定
```bash
uv run python -m zipvoice.bin.train_zipvoice \
    --world-size 1 \
    --use-fp16 1 \
    --finetune 1 \
    --checkpoint download/zipvoice/model.pt \
    --base-lr 0.005 \
    --num-iters 50000 \
    --max-duration 280 \
    --lr-hours 3800 \
    --model-config download/zipvoice/model.json \
    --tokenizer japanese \
    --token-file data/tokens_japanese_extended.txt \
    --dataset custom \
    --train-manifest data/fbank/moe_speech_cuts_train_tokenized.jsonl.gz \
    --dev-manifest data/fbank/moe_speech_cuts_dev_tokenized.jsonl.gz \
    --exp-dir exp/zipvoice_moe_90h \
    --wandb-project zipvoice-japanese
```

#### パラメータ
- 凍結: なし（全パラメータ学習可能）
- 学習率: 0.005
- イテレーション数: 50,000

#### 結果
- **日本語発音・アクセント**: 良好
  - 自然な日本語として聞こえる
  - アクセントの高低が正確
- **ゼロショット話者類似度**: 悪い
  - プロンプト音声の話者の声色が反映されない
  - 20話者の混合のような声になる

#### 考察
FM Decoderが20話者に過適合し、話者汎化能力が失われた。日本語の学習自体は成功しているが、ゼロショットTTSとしては使えない。

---

### 試行2: FM Decoder完全凍結

#### 設定
```bash
uv run python -m zipvoice.bin.train_zipvoice \
    --world-size 1 \
    --use-fp16 1 \
    --finetune 1 \
    --checkpoint download/zipvoice/model.pt \
    --freeze-fm-decoder 1 \
    --base-lr 0.01 \
    --num-iters 30000 \
    --max-duration 280 \
    --lr-hours 3800 \
    --model-config download/zipvoice/model.json \
    --tokenizer japanese \
    --token-file data/tokens_japanese_extended.txt \
    --dataset custom \
    --train-manifest data/fbank/moe_speech_cuts_train_tokenized.jsonl.gz \
    --dev-manifest data/fbank/moe_speech_cuts_dev_tokenized.jsonl.gz \
    --exp-dir exp/zipvoice_japanese_frozen \
    --wandb-project zipvoice-japanese
```

#### パラメータ
- 凍結: FM Decoder全体（118M params, 99.7%）
- 学習可能: embed + text_encoder（約4M params, 0.3%）
- 学習率: 0.01（凍結部分が多いため上げた）
- イテレーション数: 30,000

#### 結果
- **日本語発音・アクセント**: 悪い
  - 海外の人が日本語を話したような不自然なアクセント
  - 高低のパターンが不正確
- **ゼロショット話者類似度**: 良い（推定）
  - FM Decoderが保持されているため、話者特徴は維持されるはず

#### 考察
FM Decoderを完全に凍結したため、日本語のアクセントパターンを学習できなかった。Text Encoder + Embeddingだけでは日本語の韻律を表現しきれないことが判明。

---

### 試行3: 部分凍結（Stack 2,3,4凍結）

#### FM Decoder構造
```
FM Decoder (118M params)
├── Stack 0: Zipformer2Encoder (15M params) - 学習
├── Stack 1: DownsampledZipformer2Encoder (15M params) - 学習
├── Stack 2: DownsampledZipformer2Encoder (29M params) - 凍結
├── Stack 3: DownsampledZipformer2Encoder (29M params) - 凍結
└── Stack 4: Zipformer2Encoder (30M params) - 凍結
```

#### 設定
```bash
uv run python -m zipvoice.bin.train_zipvoice \
    --world-size 1 \
    --use-fp16 1 \
    --finetune 1 \
    --checkpoint download/zipvoice/model.pt \
    --freeze-fm-decoder-stacks "2,3,4" \
    --base-lr 0.01 \
    --num-iters 15000 \
    --valid-interval 500 \
    --save-every-n 2500 \
    --max-duration 280 \
    --lr-hours 3800 \
    --model-config download/zipvoice/model.json \
    --tokenizer japanese \
    --token-file data/tokens_japanese_extended.txt \
    --dataset custom \
    --train-manifest data/fbank/moe_speech_cuts_train_tokenized.jsonl.gz \
    --dev-manifest data/fbank/moe_speech_cuts_dev_tokenized.jsonl.gz \
    --exp-dir exp/zipvoice_japanese_partial \
    --wandb-project zipvoice-japanese
```

#### パラメータ
- 凍結: Stack 2,3,4 (88M params, 72%)
- 学習可能: Stack 0,1 + embed + text_encoder (34M params, 28%)
- 学習率: 0.01
- イテレーション数: 15,000（収束が早いと予想）

#### 訓練ログ（Val Loss推移）
| イテレーション | Val Loss |
|--------------|----------|
| 0 | 0.0698 |
| 500 | 0.0689 |
| 3000 | 0.0637 |
| 7000 | 0.0634 |
| 10000 | 0.0631 |
| 15000 | 0.0630 |

#### 結果
- **日本語発音・アクセント**: 悪い
  - まだ海外の人が話したような不自然なアクセント
- **ゼロショット話者類似度**: 未評価

#### 考察
Stack 0,1のみを学習させても、日本語のアクセントパターンを十分に学習できなかった。凍結量が多すぎる可能性。

---

### 試行4: Stack 4のみ凍結

#### FM Decoder構造
```
FM Decoder (118M params)
├── Stack 0: Zipformer2Encoder (15M params) - 学習
├── Stack 1: DownsampledZipformer2Encoder (15M params) - 学習
├── Stack 2: DownsampledZipformer2Encoder (29M params) - 学習
├── Stack 3: DownsampledZipformer2Encoder (29M params) - 学習
└── Stack 4: Zipformer2Encoder (30M params) - 凍結
```

#### 設定
```bash
uv run python -m zipvoice.bin.train_zipvoice \
    --world-size 1 \
    --use-fp16 1 \
    --finetune 1 \
    --checkpoint download/zipvoice/model.pt \
    --freeze-fm-decoder-stacks "4" \
    --base-lr 0.01 \
    --num-iters 15000 \
    --valid-interval 500 \
    --save-every-n 2500 \
    --max-duration 280 \
    --lr-hours 3800 \
    --model-config download/zipvoice/model.json \
    --tokenizer japanese \
    --token-file data/tokens_japanese_extended.txt \
    --dataset custom \
    --train-manifest data/fbank/moe_speech_cuts_train_tokenized.jsonl.gz \
    --dev-manifest data/fbank/moe_speech_cuts_dev_tokenized.jsonl.gz \
    --exp-dir exp/zipvoice_japanese_stack4only \
    --wandb-project zipvoice-japanese
```

#### パラメータ
- 凍結: Stack 4のみ (30M params, 25%)
- 学習可能: Stack 0,1,2,3 + embed + text_encoder (92M params, 75%)
- 学習率: 0.01
- イテレーション数: 15,000

#### 訓練ログ（Val Loss推移）
| イテレーション | Val Loss |
|--------------|----------|
| 0 | 0.0698 |
| 500 | 0.0647 |
| 3000 | 0.0637 |
| 10000 | 0.0631 |
| 14000 | 0.0639 |
| 15000 | 0.0628 |

#### 結果
- **日本語発音・アクセント**: 未評価
- **ゼロショット話者類似度**: 悪い
  - プロンプト音声の話者の声色が反映されない

#### ゼロショットテストコマンド
```bash
uv run python -m zipvoice.bin.infer_zipvoice \
    --model-dir exp/zipvoice_japanese_stack4only \
    --checkpoint-name best-valid-loss.pt \
    --tokenizer japanese \
    --prompt-wav /data/tsukuyomi-chan-ljspeech/wavs/VOICEACTRESS100_001.wav \
    --prompt-text "また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。" \
    --text "こんにちは、私はつくよみちゃんです。今日も元気に頑張ります。" \
    --res-wav-path /home/jovyan/tsukuyomi_stack4only_test.wav
```

#### 考察
凍結量を減らしたことで、試行1（凍結なし）と同様の問題が発生。FM Decoderの大部分が学習可能になったため、話者汎化能力が失われた。

---

## 実装した機能

### 1. --freeze-fm-decoder オプション

**ファイル**: `zipvoice/bin/train_zipvoice.py`

```python
parser.add_argument(
    "--freeze-fm-decoder",
    type=str2bool,
    default=False,
    help="Freeze the FM decoder to preserve speaker similarity.",
)

# 凍結ロジック
if params.freeze_fm_decoder:
    for param in model.fm_decoder.parameters():
        param.requires_grad = False
    logging.info("FM decoder is frozen")
```

### 2. --freeze-fm-decoder-stacks オプション

**ファイル**: `zipvoice/bin/train_zipvoice.py`

```python
parser.add_argument(
    "--freeze-fm-decoder-stacks",
    type=str,
    default="",
    help="Comma-separated stack indices to freeze (e.g., '2,3,4'). "
         "Use this for partial freezing to balance Japanese accent learning and speaker similarity.",
)

# 凍結ロジック
if params.freeze_fm_decoder_stacks:
    stacks_to_freeze = [int(x.strip()) for x in params.freeze_fm_decoder_stacks.split(",")]
    frozen_params = 0
    for i in stacks_to_freeze:
        if i < len(model.fm_decoder.encoders):
            for param in model.fm_decoder.encoders[i].parameters():
                param.requires_grad = False
            stack_params = sum([p.numel() for p in model.fm_decoder.encoders[i].parameters()])
            frozen_params += stack_params
            logging.info(f"FM decoder stack {i} frozen: {stack_params} params")
        else:
            logging.warning(f"Stack index {i} out of range (max: {len(model.fm_decoder.encoders)-1})")
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logging.info(f"Partial freeze complete: {frozen_params} params frozen, {trainable_params} params trainable")
```

---

## 結論と今後の方針

### 凍結アプローチの限界

凍結量を調整するだけでは、日本語アクセントと話者類似度の両立が困難であることが判明。

| 凍結量 | 日本語アクセント | 話者類似度 |
|--------|-----------------|-----------|
| 0% | ○ | × |
| 25% | ? | × |
| 72% | × | ? |
| 100% | × | ○ |

### 次のアプローチ: LoRA（Low-Rank Adaptation）

LoRAを使用することで、以下が期待できる：

1. **元のFM Decoderの重みを完全に保持**
   - 話者汎化能力を維持
   - 推論時はLoRA重みをマージして追加コストなし

2. **小さな追加パラメータのみ学習**
   - 日本語アクセントパターンを学習
   - 元の重みは変更しない

3. **理論的な利点**
   - 破滅的忘却を回避
   - 日本語と話者類似度の両立が可能

#### 実装予定
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,  # ランク
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # 注意機構のみ
    lora_dropout=0.1,
)
model.fm_decoder = get_peft_model(model.fm_decoder, lora_config)
```

---

## 関連ファイル

- `zipvoice/bin/train_zipvoice.py` - 訓練スクリプト（凍結オプション実装）
- `zipvoice/bin/infer_zipvoice.py` - 推論スクリプト
- `data/tokens_japanese_extended.txt` - 日本語トークンファイル
- `CLAUDE.md` - プロジェクト概要

## チェックポイント

| モデル | パス | 説明 |
|--------|------|------|
| Stack 4凍結 | `exp/zipvoice_japanese_stack4only/` | 現在のベスト Val Loss: 0.0628 |

---

*最終更新: 2025-01-02*
