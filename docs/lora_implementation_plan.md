# ZipVoice LoRA実装計画書

このドキュメントは、ZipVoiceの日本語対応におけるLoRA実装の調査結果と実装計画をまとめたものです。

## 目次

1. [背景と目的](#背景と目的)
2. [LoRA手法の調査](#lora手法の調査)
3. [ZipVoiceへの適用分析](#zipvoiceへの適用分析)
4. [実装計画](#実装計画)
5. [期待される効果](#期待される効果)

---

## 背景と目的

### 問題の経緯

ZipVoiceの日本語ファインチューニングにおいて、以下のトレードオフが発生：

| アプローチ | 日本語アクセント | 話者類似度 |
|-----------|-----------------|-----------|
| 凍結なし | ○ 良い | × 悪い |
| 完全凍結 | × 悪い | ○ 良い |
| 部分凍結 | × 悪い | × 悪い |

**原因**: Catastrophic Forgetting（破滅的忘却）
- オリジナルモデル: 数千話者で訓練 → 話者汎化能力が高い
- ファインチューニング: 20話者のみ → FM Decoderが過適合

### LoRAの期待

LoRA（Low-Rank Adaptation）を使用することで：
1. **元の重みを完全に保持** → 話者汎化能力を維持
2. **追加パラメータのみ学習** → 日本語アクセントを習得
3. **推論時にマージ可能** → 追加コストなし

---

## LoRA手法の調査

### 1. 言語モデル（LLM）のLoRA手法

#### 主要な手法

| 手法 | 概要 | 精度 | パラメータ効率 |
|------|------|------|---------------|
| **LoRA** | 低ランク行列の注入 | 基準 | 良い |
| **DoRA** | 方向と大きさの分離 | +3.7〜4.4pt | 良い |
| **QLoRA** | 4bit量子化+LoRA | やや低い | 最高 |
| **AdaLoRA** | 動的ランク割当 | 高い | 良い |
| **SVF** | 特異値を直接チューニング | 高い | 最高 |
| **MiLoRA** | 小さい特異値のみ調整 | 高い（数学） | 良い |
| **PiSSA** | 主成分の初期化改善 | 高い | 良い |

#### DoRA（Weight-Decomposed Low-Rank Adaptation）

**論文**: ICML 2024採択（採択率1.5%）

**仕組み**:
```
通常のLoRA: W' = W + BA
DoRA: W' = m * (W + BA) / ||W + BA||

m: 大きさ（magnitude）パラメータ
方向成分と大きさ成分を分離して学習
```

**利点**:
- LoRAより+3.7〜4.4ポイント精度向上
- ハイパーパラメータに頑健
- 少ないパラメータでより良い性能

#### LLM向けベストプラクティス

| 項目 | 推奨値 | 根拠 |
|------|--------|------|
| 適用層 | 全層（Q,K,V,O） | Key/Valueだけでは不十分 |
| ランク | 16〜64 | タスク複雑度に応じて |
| 学習率 | 1e-4〜5e-4 | 通常FTより高め |
| Alpha | 2×rank | 標準設定 |

**参考文献**:
- [Practical Tips for LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
- [LoRA Variants Overview](https://towardsdatascience.com/are-you-still-using-lora-to-fine-tune-your-llm/)

---

### 2. 画像生成モデル（Stable Diffusion）のLoRA手法

#### 適用場所

| 手法 | 適用層 | 用途 |
|------|--------|------|
| Cross-Attention LoRA | UNetのクロスアテンション | テキスト-画像関係 |
| スタイルLoRA | 全層、高ランク（32〜64） | 画風学習 |
| キャラクターLoRA | 全層、低ランク（4〜16） | 人物学習 |

#### SD向けベストプラクティス

| 項目 | 推奨値 | 根拠 |
|------|--------|------|
| スタイル学習ランク | 32〜64 | 細かいニュアンスを捉える |
| キャラクター学習ランク | 4〜16 | シンプルな特徴 |
| 学習率 | 1e-4〜5e-4 | 通常FTの100倍 |
| ステップ数 | 3000〜5000 | 通常FTの1/2 |
| データ量 | 10〜50枚 | 少量で十分 |

**重要な知見**:
- **スタイル学習は高ランクが必要** → 日本語アクセント=スタイルと捉えられる
- **少ないステップで収束** → 効率的な学習が可能

**参考文献**:
- [LoRA for Stable Diffusion](https://huggingface.co/blog/lora)
- [Fine-tuning FLUX with LoRA](https://modal.com/blog/fine-tuning-flux-style-lora)

---

### 3. 音声モデル（Whisper/wav2vec）のLoRA手法

#### 主要な研究

| 手法 | 対象 | 結果 |
|------|------|------|
| LoRA-Whisper | 多言語ASR | 言語干渉を軽減 |
| Sparsely Shared LoRA | 子供音声 | 低リソース音声改善 |
| LoRA-INT8 | 広東語ASR | CER 49.5%→11.1% |

#### 音声モデル向け知見

| 項目 | 値 | 効果 |
|------|-----|------|
| パラメータ更新率 | 0.08% | 訓練時間1/5 |
| ランク | 8 | CER大幅改善 |
| メモリ削減 | 1/10 | エッジデバイスで動作可能 |

**重要な知見**:
- **言語適応にLoRAが効果的** → 日本語適応に適用可能
- **低ランク（8）でも大幅改善** → 効率的
- **WavLM/wav2vec2ではLoRAが効果的**

**参考文献**:
- [LoRA-Whisper](https://arxiv.org/html/2406.06619v1)
- [Whisper Fine-tuning Strategies](https://link.springer.com/article/10.1186/s13636-024-00349-3)

---

## ZipVoiceへの適用分析

### ZipVoice FM Decoderの構造

```
FM Decoder (118M params)
└── encoders (nn.ModuleList, 5スタック)
    └── [i] Zipformer2Encoder / DownsampledZipformer2Encoder
        └── layers (nn.ModuleList)
            └── [j] Zipformer2EncoderLayer
                ├── self_attn_weights (RelPositionMultiheadAttentionWeights)
                │   ├── in_proj ← LoRA適用候補
                │   └── linear_pos
                ├── self_attn1, self_attn2 (SelfAttention)
                │   ├── in_proj ← LoRA適用候補
                │   └── out_proj ← LoRA適用候補
                ├── feed_forward1-3 (FeedforwardModule)
                │   └── in_proj ← LoRA適用候補（オプション）
                ├── conv_module1-2 (ConvolutionModule)
                └── nonlin_attention (NonlinAttention)
```

### 類似性分析

| ドメイン | 問題 | ZipVoice類似点 |
|---------|------|---------------|
| Stable Diffusion | スタイル vs コンテンツ | アクセント vs 話者特徴 |
| LoRA-Whisper | 言語干渉 | 日本語適応 |
| LLM | ドメイン適応 | 言語ドメイン適応 |

### 推奨設定の根拠

| 設定 | 値 | 根拠 |
|------|-----|------|
| 手法 | **DoRA** | LLM/VLMで最高精度、ICML 2024採択 |
| ランク | **32** | SD知見: スタイル学習は高ランク必要 |
| 適用層 | **全Attention** | LLM知見: 全層適用が効果的 |
| 学習率 | **1e-3** | SD知見: 高学習率で効率的 |
| Alpha | **64** | 標準: 2×rank |

---

## 実装計画

### Phase 1: 環境準備

#### 依存関係の追加

```bash
uv pip install peft
```

**pyproject.toml更新**:
```toml
[project.optional-dependencies]
lora = ["peft>=0.10.0"]
```

### Phase 2: LoRAモジュールの実装

#### 2.1 ScaledLinear互換性対応

ZipVoiceは`nn.Linear`ではなく`ScaledLinear`を使用しているため、互換性レイヤーが必要：

**ファイル**: `zipvoice/models/modules/lora_utils.py`

```python
"""LoRA utilities for ZipVoice"""

from typing import List, Optional
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer

def get_lora_target_modules(model: nn.Module) -> List[str]:
    """
    ZipVoice FM Decoderからpeft用のtarget_modulesを取得

    Returns:
        List of module names suitable for LoRA adaptation
    """
    target_modules = []

    for name, module in model.fm_decoder.named_modules():
        # Attention layers
        if any(x in name for x in [
            "self_attn_weights.in_proj",
            "self_attn1.in_proj",
            "self_attn1.out_proj",
            "self_attn2.in_proj",
            "self_attn2.out_proj",
        ]):
            # Check if it's a Linear-like layer
            if hasattr(module, 'weight') and len(module.weight.shape) == 2:
                target_modules.append(f"fm_decoder.{name}")

    return target_modules


def create_lora_config(
    rank: int = 32,
    alpha: int = 64,
    dropout: float = 0.05,
    use_dora: bool = True,
    target_modules: Optional[List[str]] = None,
) -> LoraConfig:
    """
    ZipVoice用のLoRA設定を作成

    Args:
        rank: LoRAのランク（スタイル学習は32推奨）
        alpha: スケーリング係数（2×rank推奨）
        dropout: ドロップアウト率
        use_dora: DoRAを使用するか
        target_modules: 適用するモジュール名のリスト

    Returns:
        LoraConfig object
    """
    if target_modules is None:
        target_modules = [
            "self_attn_weights.in_proj",
            "self_attn1.in_proj",
            "self_attn1.out_proj",
            "self_attn2.in_proj",
            "self_attn2.out_proj",
        ]

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        use_dora=use_dora,
        target_modules=target_modules,
        bias="none",
        task_type=None,  # カスタムモデル
    )


def apply_lora_to_model(
    model: nn.Module,
    lora_config: LoraConfig,
) -> nn.Module:
    """
    ZipVoiceモデルにLoRAを適用

    Args:
        model: ZipVoiceモデル
        lora_config: LoRA設定

    Returns:
        LoRA適用済みモデル
    """
    # FM Decoderにのみ適用
    model.fm_decoder = get_peft_model(model.fm_decoder, lora_config)

    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    LoRA重みをベースモデルにマージ（推論/エクスポート用）

    Args:
        model: LoRA適用済みモデル

    Returns:
        マージ済みモデル
    """
    if hasattr(model.fm_decoder, 'merge_and_unload'):
        model.fm_decoder = model.fm_decoder.merge_and_unload()

    return model
```

### Phase 3: 訓練スクリプトの修正

#### 3.1 コマンドライン引数の追加

**ファイル**: `zipvoice/bin/train_zipvoice.py`

```python
# LoRA関連の引数を追加
parser.add_argument(
    "--use-lora",
    type=str2bool,
    default=False,
    help="Use LoRA for fine-tuning. Preserves base model weights.",
)

parser.add_argument(
    "--lora-rank",
    type=int,
    default=32,
    help="LoRA rank. Higher for style learning (32), lower for simple adaptation (8-16).",
)

parser.add_argument(
    "--lora-alpha",
    type=int,
    default=64,
    help="LoRA alpha (scaling factor). Typically 2x rank.",
)

parser.add_argument(
    "--use-dora",
    type=str2bool,
    default=True,
    help="Use DoRA (Weight-Decomposed LoRA) for better performance.",
)

parser.add_argument(
    "--lora-dropout",
    type=float,
    default=0.05,
    help="LoRA dropout rate.",
)

parser.add_argument(
    "--lora-target-modules",
    type=str,
    default="",
    help="Comma-separated list of target modules for LoRA. "
         "Empty for default attention layers.",
)
```

#### 3.2 LoRA適用ロジック

```python
# モデル初期化後に追加
if params.use_lora:
    from zipvoice.models.modules.lora_utils import (
        create_lora_config,
        apply_lora_to_model,
        get_lora_target_modules,
    )

    # ターゲットモジュールの決定
    if params.lora_target_modules:
        target_modules = [x.strip() for x in params.lora_target_modules.split(",")]
    else:
        target_modules = None  # デフォルト使用

    # LoRA設定の作成
    lora_config = create_lora_config(
        rank=params.lora_rank,
        alpha=params.lora_alpha,
        dropout=params.lora_dropout,
        use_dora=params.use_dora,
        target_modules=target_modules,
    )

    # LoRAの適用
    model = apply_lora_to_model(model, lora_config)

    # 学習可能パラメータの確認
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"LoRA applied: {trainable_params:,} trainable / {total_params:,} total "
                 f"({100 * trainable_params / total_params:.2f}%)")
```

### Phase 4: 推論スクリプトの修正

#### 4.1 LoRAアダプターの読み込み

**ファイル**: `zipvoice/bin/infer_zipvoice.py`

```python
parser.add_argument(
    "--lora-path",
    type=str,
    default="",
    help="Path to LoRA adapter weights. If provided, loads and merges LoRA.",
)

# モデル読み込み後に追加
if args.lora_path:
    from peft import PeftModel
    model.fm_decoder = PeftModel.from_pretrained(
        model.fm_decoder,
        args.lora_path,
    )
    # 推論用にマージ
    model.fm_decoder = model.fm_decoder.merge_and_unload()
    logging.info(f"Loaded and merged LoRA from {args.lora_path}")
```

### Phase 5: ONNXエクスポート対応

#### 5.1 LoRAマージ後のエクスポート

**ファイル**: `zipvoice/bin/onnx_export.py`

```python
# エクスポート前にLoRAをマージ
if hasattr(model.fm_decoder, 'merge_and_unload'):
    logging.info("Merging LoRA weights before ONNX export...")
    model.fm_decoder = model.fm_decoder.merge_and_unload()
```

---

## 訓練コマンド

### 推奨設定

```bash
uv run python -m zipvoice.bin.train_zipvoice \
    --world-size 1 \
    --use-fp16 1 \
    --finetune 1 \
    --checkpoint download/zipvoice/model.pt \
    --use-lora 1 \
    --lora-rank 32 \
    --lora-alpha 64 \
    --use-dora 1 \
    --lora-dropout 0.05 \
    --base-lr 0.001 \
    --num-iters 10000 \
    --valid-interval 500 \
    --save-every-n 2000 \
    --max-duration 280 \
    --lr-hours 3800 \
    --model-config download/zipvoice/model.json \
    --tokenizer japanese \
    --token-file data/tokens_japanese_extended.txt \
    --dataset custom \
    --train-manifest data/fbank/moe_speech_cuts_train_tokenized.jsonl.gz \
    --dev-manifest data/fbank/moe_speech_cuts_dev_tokenized.jsonl.gz \
    --exp-dir exp/zipvoice_japanese_lora \
    --wandb-project zipvoice-japanese
```

### パラメータ説明

| パラメータ | 値 | 根拠 |
|-----------|-----|------|
| `--use-lora 1` | 有効 | LoRA使用 |
| `--lora-rank 32` | 32 | スタイル学習（SD知見） |
| `--lora-alpha 64` | 2×rank | 標準設定 |
| `--use-dora 1` | 有効 | 精度向上（LLM知見） |
| `--base-lr 0.001` | 1e-3 | 高学習率（SD知見） |
| `--num-iters 10000` | 10K | 少ないステップ（SD知見） |

---

## 期待される効果

### 理論的な期待

| 指標 | 従来（凍結なし） | LoRA期待値 |
|------|----------------|-----------|
| 日本語アクセント | ○ 良い | ○ 良い（維持） |
| 話者類似度 | × 悪い | ○ 良い（改善） |
| 訓練時間 | 基準 | 1/2〜1/5（削減） |
| パラメータ更新量 | 100% | 0.5〜2%（削減） |

### 成功判定基準

1. **日本語アクセント**: 自然な日本語として聞こえる
2. **話者類似度**: つくよみちゃんの声色が反映される
3. **Val Loss**: 0.065以下

---

## 実装スケジュール

| Phase | 内容 | 想定時間 |
|-------|------|---------|
| 1 | 環境準備（peftインストール） | 5分 |
| 2 | LoRAモジュール実装 | 30分 |
| 3 | 訓練スクリプト修正 | 30分 |
| 4 | 推論スクリプト修正 | 15分 |
| 5 | ONNXエクスポート対応 | 15分 |
| 6 | 訓練実行 | 3〜5時間 |
| 7 | ゼロショットテスト | 10分 |

---

## 参考文献

### LLM
- [Practical Tips for LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
- [LoRA Variants Overview](https://towardsdatascience.com/are-you-still-using-lora-to-fine-tune-your-llm/)
- [DoRA by NVIDIA](https://developer.nvidia.com/blog/introducing-dora-a-high-performing-alternative-to-lora-for-fine-tuning/)

### 画像生成
- [LoRA for Stable Diffusion](https://huggingface.co/blog/lora)
- [Fine-tuning FLUX with LoRA](https://modal.com/blog/fine-tuning-flux-style-lora)
- [SD Fine-tuning Guide](https://blog.segmind.com/beginners-guide-lora-fine-tuning/)

### 音声
- [LoRA-Whisper](https://arxiv.org/html/2406.06619v1)
- [Whisper Fine-tuning Strategies](https://link.springer.com/article/10.1186/s13636-024-00349-3)
- [LoRA-INT8 Whisper](https://www.mdpi.com/1424-8220/25/17/5404)

---

## 関連ファイル

| ファイル | 変更内容 |
|---------|---------|
| `pyproject.toml` | peft依存関係追加 |
| `zipvoice/models/modules/lora_utils.py` | 新規作成 |
| `zipvoice/bin/train_zipvoice.py` | LoRA引数・ロジック追加 |
| `zipvoice/bin/infer_zipvoice.py` | LoRA読み込み追加 |
| `zipvoice/bin/onnx_export.py` | LoRAマージ対応 |

---

*作成日: 2025-01-02*
*最終更新: 2025-01-02*
