# ZipFlow-100 開発計画

作成日: 2026-03-23

## 全体方針

- ZipVoice (master) を出発点に `feature/zipflow-100` ブランチ上で段階的に構築
- 既存 ZipVoice コードを**直接変更せず**、新規ファイル中心で実装（リグレッション回避）
- `feature/japanese-support` ブランチからトークナイザ・データ処理のみ取り込み（LoRA/freeze 関連は不要：スクラッチ学習のため）

---

## 既存ブランチ調査結果サマリー

| ブランチ | 活用度 | 要点 |
|---------|--------|------|
| **japanese-support** | **高** | JapaneseTokenizer (pyopenjtalk v3 G2P, accent marker), トークンファイル, データ前処理, LoRA/DoRA実装 |
| inference-optimization | 中 | distill 2-step で54x高速化, 4-step が品質/速度のスイートスポット |
| distill-onnx-sentis | 中 | Unity Sentis向けONNX export, log1p→log(1+x)置換, PE事前計算化 |
| flow2gan-vocoder | 低 | **Flow2GAN不採用** (Vocosが6-22x高速, 品質同等), Vocoder抽象化は参考になる |
| performance-optimization | 低 | torch.compile, TF32有効化 |

**japanese-support の重要な知見:**
- Catastrophic Forgetting: 凍結アプローチでは日本語アクセントと話者類似度の両立が困難
- ただし ZipFlow-100 はスクラッチ学習のためこの問題は回避される

---

## Phase 0: 基盤準備 (1-2日)

### 0-1. japanese-support ブランチからの取り込み

ファイル単位で取り込む（コミット履歴が複雑なため cherry-pick ではなく差分マージ）。

**取り込む:**

| ファイル | 内容 |
|---------|------|
| `zipvoice/tokenizer/tokenizer.py` | JapaneseTokenizer クラスを差分マージ (pyopenjtalk G2P + accent marker `[H]`/`[L]`/`|`/`[Q]`) |
| `zipvoice/tokenizer/normalizer.py` | JapaneseTextNormalizer クラスを末尾に追加 |
| `data/tokens_japanese.txt` (184トークン) | 語彙定義 |
| `data/tokens_japanese_extended.txt` (389トークン) | 拡張語彙定義 |
| `tests/test_japanese_tokenizer.py` | テストスイート |
| `scripts/convert_moe_speech_to_tsv.py` | MOE-Speech データ変換 |
| `scripts/preprocess_moe_speech.sh` | MOE-Speech 前処理パイプライン |
| `egs/zipvoice/local/prepare_tsv_tsukuyomi.py` | つくよみちゃんデータ前処理 |

**取り込まない:**

| ファイル | 理由 |
|---------|------|
| `zipvoice/models/modules/lora_utils.py` | スクラッチ学習のため不要 |
| `tests/test_lora.py`, `scripts/merge_lora_checkpoint.py` | 同上 |
| `--freeze-fm-decoder` 関連の train_zipvoice.py 変更 | 同上 |
| Docker 環境ファイル (Dockerfile, docker-compose.yml) | 後で必要に応じて別途作成 |

### 0-2. 依存関係の更新

`pyproject.toml` に追加:
- `pyopenjtalk-plus` (日本語 G2P)
- `jaconv` (全角半角変換)
- `wandb` (学習ログ)

### 0-3. 設定ファイルの作成

`egs/zipvoice/conf/zipflow100_base.json` を新規作成。既存 `zipvoice_base.json` との差分:

| パラメータ | ZipVoice (既存) | ZipFlow-100 |
|-----------|----------------|-------------|
| text_encoder_dim | 192 | 384 |
| text_encoder_num_layers | 4 | 6 |
| text_encoder_feedforward_dim | 512 | 1536 |
| text_encoder_num_heads | 4 | 6 |
| fm_decoder_num_heads | 4 | 8 |
| fm_decoder_num_layers | [2, 2, 4, 4, 4] (16層) | [2, 3, 5, 3, 2] (15層) |
| temporal_compression_factor | — | 4 (新規) |
| duration_predictor_moe_experts | — | 7 (新規) |

### 検証ポイント
- `test_japanese_tokenizer.py` が全パス
- 既存 `EmiliaTokenizer` / `EspeakTokenizer` が壊れていないこと

---

## Phase 1: モデル定義 (3-5日)

### 1-1. ZipFlow100 モデルクラス

**新規:** `zipvoice/models/zipflow100.py`

既存 `ZipVoice` クラス (534行) を参照するが、以下の根本的な違いがあるため**新規クラスとして作成**:

| メソッド | ZipVoice | ZipFlow-100 |
|---------|----------|-------------|
| `__init__` | fm_decoder (in_dim=feat_dim*3) | fm_decoder (in_dim=feat_dim*2) + duration_predictor + temporal_compressor/decompressor |
| `forward_fm_decoder` | `cat([xt, text_cond, speech_cond], dim=2)` | `cat([xt, reference_mel], dim=2)` + Cross-Attention for text |
| `forward_text_condition` | 均一フレーム配分 | MoE Duration Predictor で予測 |
| `forward` (train) | speech_condition_mask | F5-TTS 式: [reference \| noisy_target] concat + binary mask |
| `sample` | prompt concat → denoise | reference compress → concat → denoise → decompress |

### 1-2. AdaLN-Zero 対応 Zipformer

**新規:** `zipvoice/models/modules/zipformer_adaln.py`

既存の `Zipformer2EncoderLayer` は time_emb を直接加算 (`src = src + time_emb`)。ZipFlow-100 では AdaLN-Zero に変更。Cross-Attention (text KV) も追加。

既存 `zipformer.py` を直接変更しない理由: `ZipVoiceDistill`, `ZipVoiceDialog` 等が依存しておりリグレッションリスクがある。

### 1-3. Temporal Compressor / Decompressor

**新規:** `zipvoice/models/modules/temporal.py`

```
TemporalCompressor:  ConvNeXt 2L, stride=4, (B, T, 100) → (B, T//4, 100)  [1.5M]
TemporalDecompressor: transposed conv + ConvNeXt 1L, (B, T//4, 100) → (B, T, 100)  [0.5M]
```

参考: 既存 `zipformer.py` の `SimpleDownsample`/`SimpleUpsample` (attention-based)。ConvNeXt ベースのストライド畳み込みは新規実装。

### 1-4. MoE Duration Predictor

**新規:** `zipvoice/models/modules/duration_predictor.py`

```
MoEDurationPredictor: 1 shared + 7 routed experts, top-1 routing  [1.5M]
入力: text_encoder出力 (B, S, 384) → 出力: duration per token (B, S)
```

既存 ZipVoice には Duration Predictor がない（`prepare_avg_tokens_durations` で均一配分）。

### 検証ポイント
- パラメータ数カウント: FM Decoder ~80M, Text Encoder ~6M, Duration Predictor ~1.5M, Compressor ~1.5M, Decompressor ~0.5M = **~89M**
- ダミー入力での forward pass (train/inference 両方)
- Temporal Compressor/Decompressor の入出力形状
- MoE routing が正しく動作すること

---

## Phase 2: データパイプライン (2-3日)

### 2-1. 学習レシピ

**新規:** `egs/zipvoice/run_zipflow100.sh`

japanese-support の `run_japanese.sh` を参考に、多話者マルチデータセット対応。

### 2-2. データ前処理スクリプト

**新規:** `scripts/prepare_emilia_ja.py`, `scripts/prepare_galgame.py`

既存の `convert_moe_speech_to_tsv.py` と同じ TSV 形式に統一。

### 2-3. DataModule 拡張

**変更:** `zipvoice/dataset/datamodule.py`

追加機能:
- 複数データセットの混合サンプリング (Emilia 50% + Galgame 30% + MOE-Speech 15% + JVS 5%)
- Reference audio ペアリング (同一話者の別発話)
- Curriculum learning 用 duration-based sorting

### 2-4. Dataset 拡張

**変更:** `zipvoice/dataset/dataset.py`

バッチに追加: `reference_features`, `reference_features_lens`, `reference_text`

### 検証ポイント
- 小規模サブセット (JVS 22.9h) でデータローダーが正しく動作
- バッチ内で reference と target が同一話者であること
- Curriculum learning のソート順が正しいこと

---

## Phase 3: 学習スクリプト (3-4日)

### 3-1. Stage 1 学習

**新規:** `zipvoice/bin/train_zipflow100.py`

既存 `train_zipvoice.py` (1130行) をベースに変更:

| 項目 | ZipVoice | ZipFlow-100 |
|------|----------|-------------|
| モデル | ZipVoice | ZipFlow100 |
| 精度 | FP16 | **bf16** (flow matching の勾配レンジに最適) |
| Optimizer | ScaledAdam | **AdamW** (lr=7.5e-5, warmup 20K) |
| Scheduler | Eden | **cosine with warmup** |
| 損失 | speech_condition_mask + MSE | **F5-TTS式 reference concat + binary mask + MSE** |
| Grad ckpt | なし | **selective (2ブロックごと)** |
| Reference dropout | — | **10%** (CFG対応) |

### 3-2. Stage 2 学習 (RapFlow)

**新規:** `zipvoice/bin/train_zipflow100_rapflow.py`

- Phase A: Velocity Consistency (L_sf + alpha * L_vc)
- Phase B: Adversarial (2-4 step Euler 固定)

### 3-3. Discriminator

**新規:** `zipvoice/models/discriminators/`

| Discriminator | パラメータ | 備考 |
|---|---|---|
| DistilHuBERT SLM + CNN head | 23M (frozen) | 学習時のみ |
| MPD | 15M | 学習時のみ |
| MS-SB-CQT-D | 5M | 学習時のみ |
| MS-STFT-D | 1M | 学習時のみ |

Loss比: L_adv : L_fm : L_mel : L_slm = 3 : 1 : 2 : 1

### 3-4. Stage 3 蒸留

**新規:** `zipvoice/bin/train_zipflow100_distill.py`

既存 `train_zipvoice_distill.py` の `DistillEulerSolver` を再利用。IntMeanFlow 独自ロジック (O3S, CFG吸収) は新規実装。

### 検証ポイント
- JVS (22.9h) で 1000 step が完了し loss が減少すること
- bf16 で gradient が NaN/Inf にならないこと
- gradient checkpointing で VRAM 18GB 以下
- Reference dropout 10% が正しく機能すること

---

## Phase 4: 推論と評価 (2-3日)

### 4-1. 推論スクリプト

**新規:** `zipvoice/bin/infer_zipflow100.py`

処理フロー:
```
Reference audio → Vocos encoder → mel → temporal compress → compressed_ref
Target text → JapaneseTokenizer → Text Encoder → Duration Predictor → frame展開
[compressed_ref | noisy_target] → Flow Decoder (2-4 step) → temporal decompress → Vocos → waveform
```

### 4-2. 評価

既存 `zipvoice/eval/` を再利用 + 以下を追加:
- UTMOS (MOS 推定)
- Speaker Similarity (reference vs generated)
- CER (日本語文字誤り率)

---

## Phase 5: 最適化とデプロイ (3-5日)

### 5-1. Q8F16 量子化

**新規:** `zipvoice/bin/quantize_zipflow100.py`

Kokoro-82M 方式: 層ごとに INT8 量子化 → mel-loss で感度評価 → 高感度層は FP16 維持

### 5-2. ONNX Export

既存 `onnx_export.py` に `zipflow100` モデルタイプを追加。
distill-onnx-sentis の知見 (log1p→log(1+x), PE事前計算化) を適用。

出力: `text_encoder.onnx` (~6MB) + `fm_decoder.onnx` (~83MB)

### 5-3. sherpa-onnx 統合

C++ 日本語 tokenizer フロントエンド → sherpa-onnx 側リポジトリで実装。

---

## リスクと対策

| リスク | 深刻度 | 対策 |
|--------|--------|------|
| AdaLN-Zero + Cross-Attention の Zipformer 統合 | **高** | 新規ファイル `zipformer_adaln.py` で実装、既存コードに触らない |
| Temporal Compressor の品質 | 中 | Kc=4→Kc=2 フォールバック可、stride を設定パラメータ化 |
| Stage 2 の VRAM (RTX 4090 24GB) | 中 | grad checkpointing 必須、最悪 MPD のみに絞る |
| F5-TTS 式 in-context の実装精度 | 中 | F5-TTS オリジナルコードと突き合わせ検証 |
| MoE Duration Predictor の学習安定性 | 低 | load balancing loss 追加、安定しなければ通常 FFN にフォールバック |
| 既存 ZipVoice コードとの共存 | 低 | 共有モジュール (scaling.py, solver.py) は変更しない |

---

## ファイル一覧

### 新規作成 (14ファイル)

```
zipvoice/models/zipflow100.py                     # メインモデル
zipvoice/models/modules/zipformer_adaln.py         # AdaLN-Zero + Cross-Attention Zipformer
zipvoice/models/modules/temporal.py                # Temporal Compressor / Decompressor
zipvoice/models/modules/duration_predictor.py      # MoE Duration Predictor
zipvoice/models/discriminators/__init__.py          # Discriminator パッケージ
zipvoice/models/discriminators/distilhubert_slm.py  # DistilHuBERT SLM
zipvoice/models/discriminators/mpd.py               # Multi-Period Discriminator
zipvoice/models/discriminators/ms_sb_cqt_d.py       # Multi-Scale Sub-Band CQT Discriminator
zipvoice/models/discriminators/ms_stft_d.py         # Multi-Scale STFT Discriminator
zipvoice/bin/train_zipflow100.py                    # Stage 1 学習
zipvoice/bin/train_zipflow100_rapflow.py            # Stage 2 学習
zipvoice/bin/train_zipflow100_distill.py            # Stage 3 蒸留
zipvoice/bin/infer_zipflow100.py                    # 推論
egs/zipvoice/conf/zipflow100_base.json              # モデル設定
```

### 変更 (5ファイル)

```
zipvoice/tokenizer/tokenizer.py      # JapaneseTokenizer 追加
zipvoice/tokenizer/normalizer.py     # JapaneseTextNormalizer 追加
zipvoice/dataset/datamodule.py       # 多データセット混合, reference ペアリング
zipvoice/dataset/dataset.py          # reference_features をバッチに追加
pyproject.toml                        # pyopenjtalk-plus, jaconv, wandb 追加
```

### japanese-support から取り込み (そのまま)

```
data/tokens_japanese.txt
data/tokens_japanese_extended.txt
tests/test_japanese_tokenizer.py
scripts/convert_moe_speech_to_tsv.py
scripts/preprocess_moe_speech.sh
egs/zipvoice/local/prepare_tsv_tsukuyomi.py
```

### 変更しない (既存のまま流用)

```
zipvoice/models/modules/zipformer.py    # TTSZipformer, Zipformer2EncoderLayer — 参照のみ
zipvoice/models/modules/scaling.py      # BiasNorm, SwooshR — そのまま使用
zipvoice/models/modules/solver.py       # EulerSolver — そのまま使用
zipvoice/utils/common.py                # make_pad_mask 等 — そのまま使用
zipvoice/utils/feature.py              # VocosFbank — そのまま使用
zipvoice/utils/checkpoint.py           # チェックポイント管理 — そのまま使用
```

---

## 参照リソース

| リソース | 場所 |
|---------|------|
| HANDOFF.md (詳細設計) | `/Users/s19447/Desktop/small-flow-tts/HANDOFF.md` |
| 調査ドキュメント | `/Users/s19447/Desktop/small-flow-tts/docs/` |
| 最終設計 "ZipFlow-100" | `/Users/s19447/Desktop/small-flow-tts/docs/18_100mb_max_quality_design.md` |
| ZipVoice モデル定義 | `zipvoice/models/zipvoice.py` (534行) |
| Zipformer 実装 | `zipvoice/models/modules/zipformer.py` (1680行) |
| 学習スクリプト | `zipvoice/bin/train_zipvoice.py` (1130行) |
| RapFlow-TTS | https://github.com/naver-ai/RapFlow-TTS |
| F5-TTS | https://github.com/SWivid/F5-TTS |
