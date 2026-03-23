# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

**ZipFlow-100**: ZipVoice (k2-fsa) をフォークし、100MB以下・MOS 4.1-4.3・Zero-Shot対応のモバイル日本語TTSを新規構築するプロジェクト。

| 項目 | 値 |
|------|-----|
| デプロイサイズ | ~89MB (acoustic, Q8F16) + ~14MB (vocoder, 外部共有) |
| パラメータ | 89M (acoustic model) |
| 品質目標 | MOS 4.1-4.3 (zero-shot) |
| Zero-Shot | F5-TTS 式 in-context (追加パラメータ 0) |
| 推論ステップ | 2-4 NFE |
| 学習環境 | RTX 4090 1台 (24GB VRAM) |
| フォーク元 | ZipVoice (k2-fsa, 123M, 中国語/英語) |

### ZipVoice から流用するもの
- Zipformer backbone (attention weight reuse, BiasNorm, SwooshR)
- Flow Matching (OT-CFM)
- Flow Distillation (32→4 step)
- ONNX export パイプライン
- sherpa-onnx 統合 (モバイルデプロイ)

### 自前実装が必要なもの
- 日本語フロントエンド (pyopenjtalk)
- In-context zero-shot (F5-TTS 式 reference mel concat)
- 時間圧縮 Kc=4 (SupertonicTTS 方式: ConvNeXt 2L, stride=4, 86Hz→21Hz)
- 時間伸張 (transposed conv + ConvNeXt 1L, 21Hz→86Hz)
- MoE Duration Predictor (1 shared + 7 routed experts, top-1)
- RapFlow adversarial fine-tuning
- DistilHuBERT SLM discriminator (学習時のみ)
- IntMeanFlow 蒸留
- Q8F16 量子化 (Kokoro-82M 方式, 層選択的)

## 開発コマンド

### 依存関係のインストール
```bash
uv sync
uv pip install piper-phonemize --find-links https://k2-fsa.github.io/icefall/piper_phonemize.html
uv pip install -r requirements_eval.txt  # 評価用
```

### k2のインストール（訓練に必須）
```bash
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

## ZipFlow-100 アーキテクチャ

### 全体構成

```
Text → pyopenjtalk (IPA) → Text Encoder (6M, Conformer 6L, dim=384)
                              ↓
                           Duration Predictor (1.5M, MoE: 1+7 experts)
                              ↓
                           Temporal Compressor (1.5M, ConvNeXt 2L, stride=4)
                              ↓                          ↓ (concat)
                           noisy latent              Reference mel (zero-shot)
                              ↓
                           Flow Decoder (80M, Zipformer U-Net 5 stacks)
                              ↓
                           Temporal Decompressor (0.5M)
                              ↓
                           Vocos Vocoder (13.5M, 外部共有) → 24kHz waveform
```

### パラメータ配分

| コンポーネント | パラメータ | Q8F16 サイズ |
|---|---|---|
| Flow Decoder | 80M (90%) | ~80MB |
| Text Encoder | 6M | ~6MB |
| Duration Predictor (MoE) | 1.5M | ~1.5MB |
| Temporal Compressor | 1.5M | ~1.5MB |
| Temporal Decompressor | 0.5M | — |
| **Acoustic Model 合計** | **~89M** | **~89MB** |

### Flow Decoder 詳細 (Zipformer U-Net)
- 5 stacks, dim=512, ff=1536, heads=8
- layers=[2, 3, 5, 3, 2] (15 layers total)
- downsampling=[1x, 2x, 4x, 2x, 1x]
- 各 Zipformer2EncoderLayer: Shared RelPositionMHSA (weight reuse), 2x SelfAttention, 1x NonlinAttention, 3x Feedforward (SwooshR), ConvolutionModule, Bypass connections, BiasNorm, AdaLN-Zero
- Cross-Attention: text KV at each stack

### Zero-Shot の仕組み (F5-TTS 式 in-context)
- Reference audio (3-10秒) → Vocos encoder → mel → temporal compression → compressed reference
- [compressed_reference | noisy_target] を concat → Flow Decoder が denoise (speech infilling)
- 学習時: reference dropout 10% で CFG 対応
- Speaker Encoder 不要 (追加パラメータ 0)

## 学習パイプライン (RTX 4090, ~14-20日)

### Stage 1: OT-CFM 基礎学習 (~7-10日)
- bf16 (FP16ではなく。Flow matchingの勾配レンジに最適、loss scaling不要)
- AdamW, lr=7.5e-5, warmup 20K steps
- gradient checkpointing (selective, 2ブロックごと)
- Flash Attention (PyTorch SDPA)
- curriculum learning (短→長)
- 500K-800K steps

### Stage 2: RapFlow Consistency + Adversarial Fine-tuning (+3-5日)
- Phase A: Velocity Consistency (L_sf + alpha * L_vc)
- Phase B: Adversarial (2-4 step Euler固定で学習)
  - DistilHuBERT (23M, frozen) + CNN head
  - MPD (15M) + MS-SB-CQT-D (5M) + MS-STFT-D (1M)
  - Loss比: L_adv : L_fm : L_mel : L_slm = 3 : 1 : 2 : 1
  - **Discriminatorは学習時のみ、推論モデルに含まない**

### Stage 3: IntMeanFlow 蒸留 (+1-2日)
- Teacher (Stage 2完了, frozen) → Student (同一アーキテクチャ)
- 32 step → 2-4 step, CFGをstudentに吸収

### Stage 4: Q8F16 量子化 (数時間)
- 層ごとにINT8量子化→mel-lossで感度評価
- 高感度層 (attention scores, BiasNorm, 出力層, AdaLN-Zero) → FP16維持
- 低感度層 (FFN, Conv, 中間層) → INT8

### Stage 5: sherpa-onnx 統合
- ONNX export: text_encoder.onnx + fm_decoder.onnx + vocos.onnx
- C++ 日本語 tokenizer フロントエンド

## 既存コードベース (ZipVoice) のアーキテクチャ

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

### 既存モデル構造 (ZipVoice 123M)

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

### 既存モデルバリアント
`ZipVoice`, `ZipVoiceDistill`, `ZipVoiceDialog`, `ZipVoiceDialogStereo`

### 既存トークナイザ
`EmiliaTokenizer`(中国語), `LibriTTSTokenizer`(英語), `EspeakTokenizer`(多言語)

### 推論バックエンド
PyTorch / ONNX / TensorRT / Triton

## 重要な実装詳細

### テキスト内の特殊記法
- `<pinyin>`: 中国語ピンイン直接指定
- `[tag]`: 特殊タグ
- `[S1]`/`[S2]`: 対話モードでの話者識別

### 設定ファイル
- モデル設定: `egs/zipvoice/conf/zipvoice_base.json`
- サンプルレート: 24kHz
- 特徴量: Vocos fbank（100次元）

### 訓練レシピ
完全な訓練パイプラインは `egs/zipvoice/run_emilia.sh` を参照

## 設計判断の根拠

| 判断 | 選択 | 理由 |
|------|------|------|
| 量子化 | Q8F16 (層選択的) | Kokoro-82M で品質維持が実証済み |
| Vocoder | 外部共有 | 100MB 予算を acoustic model に全振り |
| Zero-shot | In-context (F5-TTS式) | 追加パラメータ 0、品質もSpeaker Encoder式以上 |
| Backbone | Zipformer | Attention weight reuse で効率 3x、ONNX 実績あり |
| 生成対象 | Mel-spectrogram | Codec decoder (DAC 70M) は大きすぎる |
| 時間圧縮 | Kc=4 | 品質と速度のバランス (Kc=6は品質リスク) |
| Adversarial | RapFlow 方式 | Matcha-TTS で MOS +0.18 の実績 |
| 精度 | bf16 | Flow matching の勾配レンジに最適 |

## 参照リソース

- 詳細設計: `/Users/s19447/Desktop/small-flow-tts/HANDOFF.md`
- 調査ドキュメント: `/Users/s19447/Desktop/small-flow-tts/docs/`
- 研究メモ: `/Users/s19447/Desktop/small-flow-tts/research/`
- フォーク元: https://github.com/k2-fsa/ZipVoice
- RapFlow: https://github.com/naver-ai/RapFlow-TTS
- F5-TTS: https://github.com/SWivid/F5-TTS
- sherpa-onnx: https://github.com/k2-fsa/sherpa-onnx
