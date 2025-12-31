# Flow2GAN ボコーダー統合

## 概要

本ドキュメントでは、ZipVoiceへのFlow2GANボコーダー統合の調査結果と実装内容について説明します。

## 結論

**Flow2GANはVocosと比較してメリットがなく、むしろVocosの方が高速です。**

| 観点 | Flow2GAN | Vocos | 結果 |
|------|----------|-------|------|
| ボコーダー速度 | 25〜95 ms | 4 ms | **Vocos 6〜22倍高速** |
| 音質 | 良好（修正後） | 良好 | 同等 |
| 安定性 | 設定に注意が必要 | シンプル | Vocos優位 |
| ボトルネック影響 | 全体の1〜5% | 全体の0.2% | N/A |

**重要**: ボコーダーは推論時間全体の0.2〜5%程度しか占めていません。ボトルネックはZipVoiceモデル本体（約93〜98%）であり、ボコーダーの最適化は全体性能にほとんど影響しません。

## 背景

### 現行ボコーダー: Vocos

ZipVoiceは現在、Vocos (`charactr/vocos-mel-24khz`) を使用してメルスペクトログラムを波形に変換しています。

### 新ボコーダー: Flow2GAN

Flow2GANはFlow MatchingとGANファインチューニングを組み合わせたボコーダーで、1〜4ステップでの推論が可能です。

**主な特徴:**
- 1〜4ステップ推論（設定可能）
- Flow Matching + GANファインチューニング
- マルチブランチConvNeXtアーキテクチャ

## 互換性分析

両ボコーダーは同一のメルスペクトログラム仕様を使用：

| パラメータ | ZipVoice | Flow2GAN |
|-----------|----------|----------|
| サンプリングレート | 24kHz | 24kHz |
| Mel Bins | 100 | 100 |
| FFT Size | 1024 | 1024 |
| Hop Length | 256 | 256 |

特徴抽出パイプラインの変更は不要です。

## ベンチマーク結果

### テスト環境
- GPU: CUDA対応デバイス (NVIDIA)
- テキスト: "The quick brown fox jumps over the lazy dog."
- プロンプト: 全テストで同一の参照音声を使用
- 計測回数: 各条件5回の平均値

### 推論パイプライン詳細時間内訳

#### Vocos（デフォルト）

| 処理ステップ | 処理時間 | 割合 |
|-------------|---------|------|
| プロンプト読込 | 10.73 ms | 0.5% |
| プロンプト前処理 | 20.18 ms | 1.0% |
| 特徴量抽出 | 2.66 ms | 0.1% |
| トークン化 | 1.22 ms | 0.1% |
| **モデル推論 (ZipVoice)** | **1929.54 ms** | **97.8%** |
| ボコーダー | 4.29 ms | 0.2% |
| 後処理 | 4.15 ms | 0.2% |
| **合計** | **1972.77 ms** | **100%** |

#### Flow2GAN 1-step

| 処理ステップ | 処理時間 | 割合 |
|-------------|---------|------|
| プロンプト読込 | 6.21 ms | 0.3% |
| プロンプト前処理 | 20.91 ms | 1.1% |
| 特徴量抽出 | 2.98 ms | 0.2% |
| トークン化 | 1.33 ms | 0.1% |
| **モデル推論 (ZipVoice)** | **1891.24 ms** | **96.8%** |
| ボコーダー | 25.76 ms | 1.3% |
| 後処理 | 5.25 ms | 0.3% |
| **合計** | **1953.67 ms** | **100%** |

#### Flow2GAN 2-step

| 処理ステップ | 処理時間 | 割合 |
|-------------|---------|------|
| プロンプト読込 | 15.68 ms | 0.8% |
| プロンプト前処理 | 18.24 ms | 0.9% |
| 特徴量抽出 | 4.76 ms | 0.2% |
| トークン化 | 2.14 ms | 0.1% |
| **モデル推論 (ZipVoice)** | **1910.95 ms** | **95.3%** |
| ボコーダー | 48.24 ms | 2.4% |
| 後処理 | 4.31 ms | 0.2% |
| **合計** | **2004.33 ms** | **100%** |

#### Flow2GAN 4-step

| 処理ステップ | 処理時間 | 割合 |
|-------------|---------|------|
| プロンプト読込 | 8.06 ms | 0.4% |
| プロンプト前処理 | 20.23 ms | 1.0% |
| 特徴量抽出 | 3.28 ms | 0.2% |
| トークン化 | 1.85 ms | 0.1% |
| **モデル推論 (ZipVoice)** | **1894.01 ms** | **93.4%** |
| ボコーダー | 94.89 ms | 4.7% |
| 後処理 | 5.78 ms | 0.3% |
| **合計** | **2028.10 ms** | **100%** |

### ボコーダー速度比較

| ボコーダー | ボコーダー処理時間 | 全体処理時間 | ボコーダー割合 |
|-----------|------------------|-------------|--------------|
| **Vocos** | **4.29 ms** | 1972.77 ms | **0.2%** |
| Flow2GAN 1-step | 25.76 ms | 1953.67 ms | 1.3% |
| Flow2GAN 2-step | 48.24 ms | 2004.33 ms | 2.4% |
| Flow2GAN 4-step | 94.89 ms | 2028.10 ms | 4.7% |

### 重要な発見

1. **ZipVoiceモデルがボトルネック**: 推論時間の93.4〜97.8%を占める
2. **Vocosが最速**: 4.29msでFlow2GANより6〜22倍高速
3. **ボコーダー最適化の効果は限定的**: ボコーダーを0msにしても全体の2〜5%しか改善しない
4. **Flow2GANはステップ数に比例して遅くなる**: 1-step→2-step→4-stepで約2倍ずつ増加

### 音質比較

| ボコーダー | 音質 |
|-----------|------|
| Vocos | 良好、安定 |
| Flow2GAN 1-step | 良好（修正後） |
| Flow2GAN 2-step | 良好（修正後） |
| Flow2GAN 4-step | 良好（修正後） |

必要な修正を適用後、ボコーダー間で音質に大きな差はありません。

### 初回ロード時間（参考）

| 項目 | 時間 |
|-----|------|
| ZipVoiceモデルロード | ~2300 ms |
| Vocosロード | ~540 ms |
| Flow2GANロード | ~1350-1640 ms |

## 統合手順

以下は実際に行ったFlow2GAN統合の手順です。

### Step 1: ブランチ作成

```bash
git checkout master
git checkout -b feature/flow2gan-vocoder
```

### Step 2: Flow2GANリポジトリの準備

Flow2GANをローカルにclone:
```bash
git clone https://github.com/k2-fsa/Flow2GAN.git C:\Users\yuta\Desktop\Private\Flow2GAN
```

Flow2GANには`pyproject.toml`がなかったため、新規作成:
```toml
[project]
name = "flow2gan"
version = "0.1.0"
dependencies = [
    "torch>=2.0",
    "torchaudio",
    "huggingface-hub",
    "safetensors",
]

[project.optional-dependencies]
train = [
    "librosa",
    "lhotse",
]
```

**注意**: `librosa`と`lhotse`はPython 3.11でnumba依存関係の問題があるため、推論に不要なものはオプショナル依存に移動しました。

### Step 3: Flow2GANを依存関係に追加

`uv add`でローカルパスからeditableインストール:
```bash
uv add flow2gan --editable "C:\Users\yuta\Desktop\Private\Flow2GAN"
```

これにより`pyproject.toml`に以下が追加されます:
```toml
[tool.uv.sources]
flow2gan = { path = "C:\\Users\\yuta\\Desktop\\Private\\Flow2GAN", editable = true }
```

### Step 4: ボコーダーパッケージの作成

`zipvoice/vocoder/`ディレクトリを作成し、統一インターフェースを実装:

#### 4.1 基底クラス (`base.py`)
```python
from abc import ABC, abstractmethod
import torch

class BaseVocoder(ABC):
    @abstractmethod
    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        """(B, 100, T) -> (B, 1, samples)"""
        pass

    @abstractmethod
    def to(self, device: torch.device) -> "BaseVocoder":
        pass

    @abstractmethod
    def eval(self) -> "BaseVocoder":
        pass
```

#### 4.2 Vocosラッパー (`vocos.py`)
既存のVocosコードをラッパークラスに移行。

#### 4.3 Flow2GANラッパー (`flow2gan.py`)
```python
from flow2gan import get_model

class Flow2GANVocoder(BaseVocoder):
    def __init__(self, n_timesteps: int = 2, ...):
        if hf_model_name is None:
            hf_model_name = f"libritts-mel-{n_timesteps}-step"
        self.model, self.config = get_model(...)
```

#### 4.4 ファクトリ関数 (`__init__.py`)
```python
def get_vocoder(vocoder_type: str = "vocos", **kwargs) -> BaseVocoder:
    if vocoder_type == "vocos":
        return VocosVocoder(**kwargs)
    elif vocoder_type == "flow2gan":
        return Flow2GANVocoder(**kwargs)
```

### Step 5: 推論スクリプトの更新

`zipvoice/bin/infer_zipvoice.py`に以下を追加:

```python
# 新しい引数
parser.add_argument("--vocoder-type", type=str, default="vocos",
                    choices=["vocos", "flow2gan"])
parser.add_argument("--vocoder-n-steps", type=int, default=2)

# ボコーダー初期化を新しいファクトリ関数に変更
from zipvoice.vocoder import get_vocoder
vocoder = get_vocoder(
    vocoder_type=params.vocoder_type,
    n_timesteps=params.vocoder_n_steps,
)
```

### Step 6: テストと問題の発見

#### 6.1 初回テスト
```bash
uv run python -m zipvoice.bin.infer_zipvoice \
    --vocoder-type flow2gan --vocoder-n-steps 2 \
    --prompt-wav data/prompt.wav --prompt-text "Hello" \
    --text "The quick brown fox" --res-wav-path result.wav
```

#### 6.2 発見した問題1: 音声が破損（1-step, 2-step）
- **原因**: モデルが`libritts-mel-4-step`に固定されていた
- **修正**: `n_timesteps`に応じてモデルを自動選択

#### 6.3 発見した問題2: 音割れ（全ステップ）
- **原因**: `clamp_pred=True`によるハードクリッピング
- **修正**: ピーク正規化を適用（Discriminatorと同じ処理）

### Step 7: ベンチマーク実施

```bash
# Vocos
uv run python scripts/benchmark_inference.py --vocoder-type vocos

# Flow2GAN (各ステップ)
uv run python scripts/benchmark_inference.py --vocoder-type flow2gan --vocoder-n-steps 1
uv run python scripts/benchmark_inference.py --vocoder-type flow2gan --vocoder-n-steps 2
uv run python scripts/benchmark_inference.py --vocoder-type flow2gan --vocoder-n-steps 4
```

### Step 8: コミットとプッシュ

```bash
git add .
git commit -m "Add Flow2GAN vocoder integration"
git push -u origin feature/flow2gan-vocoder
```

### コミット履歴

| コミット | 内容 |
|---------|------|
| `ad2a4ab` | Add Flow2GAN vocoder integration |
| `6379e3d` | Update infer_zipvoice.py to use vocoder package |
| `86468cd` | Fix Flow2GAN model selection based on n_timesteps |
| `fef2297` | Fix audio distortion and update integration documentation |
| `55af77e` | Update documentation to Japanese |
| `0feff4f` | Add detailed inference timing benchmarks |

## 発見した問題と修正

### 問題1: モデル選択バグ

**問題:** Flow2GANはステップ数ごとに専用の事前学習モデルが必要：
- `libritts-mel-1-step` → 1ステップ推論用
- `libritts-mel-2-step` → 2ステップ推論用
- `libritts-mel-4-step` → 4ステップ推論用

初期実装では`n_timesteps`に関係なく固定のモデルを使用していたため、1ステップと2ステップで音声が破損していました。

**修正:** `n_timesteps`に基づいてモデルを自動選択：
```python
if hf_model_name is None:
    hf_model_name = f"libritts-mel-{n_timesteps}-step"
```

### 問題2: 音割れ（ハードクリッピング）

**問題:** すべてのステップ設定で音割れが発生。

**根本原因:** 訓練と推論の不一致：
1. GAN訓練時: `clamp_pred=False`（生成器の出力は[-1, 1]を超える可能性あり）
2. 推論時: `clamp_pred=True`でハードクリッピング → 音割れ

Discriminatorは内部で音声を正規化（DC除去 + 0.8へのピーク正規化）するため、Generatorは[-1, 1]を超える値を出力するように学習します。

**修正:** Discriminatorと同じピーク正規化を適用：
```python
def decode(self, mel: torch.Tensor) -> torch.Tensor:
    wav = self.model.infer(
        cond=mel,
        n_timesteps=self.n_timesteps,
        clamp_pred=False,  # ハードクリッピング無効
    )
    # ピーク正規化 (0.8) - Discriminator訓練時と同じ
    wav = wav - wav.mean(dim=-1, keepdim=True)
    wav = 0.8 * wav / (wav.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
    return wav.unsqueeze(1)
```

## 実装詳細

### 新規作成ファイル

| ファイル | 説明 |
|---------|------|
| `zipvoice/vocoder/__init__.py` | ファクトリ関数 `get_vocoder()` |
| `zipvoice/vocoder/base.py` | 抽象基底クラス `BaseVocoder` |
| `zipvoice/vocoder/vocos.py` | Vocosラッパー `VocosVocoder` |
| `zipvoice/vocoder/flow2gan.py` | Flow2GANラッパー `Flow2GANVocoder` |

### 修正ファイル

| ファイル | 変更内容 |
|---------|---------|
| `zipvoice/bin/infer_zipvoice.py` | `--vocoder-type`と`--vocoder-n-steps`引数を追加 |
| `pyproject.toml` | Flow2GAN依存関係を追加 |

### ボコーダーパッケージ構成

```
zipvoice/vocoder/
    __init__.py     # ファクトリ関数 get_vocoder()
    base.py         # BaseVocoder抽象クラス
    vocos.py        # VocosVocoderラッパー
    flow2gan.py     # Flow2GANVocoderラッパー
```

### インターフェース設計

```python
# 基底クラス
class BaseVocoder(ABC):
    @abstractmethod
    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        """(B, 100, T) -> (B, 1, samples)"""
        pass

# ファクトリ関数
def get_vocoder(vocoder_type: str = "vocos", **kwargs) -> BaseVocoder:
    if vocoder_type == "vocos":
        return VocosVocoder(**kwargs)
    elif vocoder_type == "flow2gan":
        return Flow2GANVocoder(**kwargs)
```

## 使用方法

```bash
# Vocos使用（デフォルト、推奨）
uv run python -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --prompt-wav prompt.wav \
    --prompt-text "プロンプトテキスト" \
    --text "合成するテキスト" \
    --res-wav-path result.wav

# Flow2GAN使用（2ステップ）
uv run python -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --vocoder-type flow2gan \
    --vocoder-n-steps 2 \
    --prompt-wav prompt.wav \
    --prompt-text "プロンプトテキスト" \
    --text "合成するテキスト" \
    --res-wav-path result.wav
```

## Flow2GAN事前学習モデル

| モデル名 | ステップ数 | 用途 |
|---------|-----------|------|
| `libritts-mel-1-step` | 1 | 超高速 |
| `libritts-mel-2-step` | 2 | バランス |
| `libritts-mel-4-step` | 4 | 最高品質 |

**重要:** 各ステップ数には対応する事前学習モデルが必要です。

## 推奨事項

1. **Vocosをデフォルトとして使用** - シンプルで安定しており、同等の性能
2. **Flow2GANはオプションとして保持** - 将来の実験や特定のユースケースで有用な可能性
3. **最適化の焦点はZipVoiceモデルに** - 実際のボトルネック（推論時間の95%以上）

## 今後の検討事項

- 将来ZipVoiceの最適化によりモデル推論時間が大幅に短縮された場合、Flow2GANが有益になる可能性
- ZipVoice専用データでのFlow2GANカスタム訓練による品質向上
- デプロイメントシナリオ向けのFlow2GAN ONNX/TensorRTエクスポート

## 参考資料

- [Flow2GAN リポジトリ](https://github.com/k2-fsa/Flow2GAN)
- [Vocos](https://github.com/gemelo-ai/vocos)
