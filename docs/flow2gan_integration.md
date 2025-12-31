# Flow2GAN ボコーダー統合

## 概要

本ドキュメントでは、ZipVoiceへのFlow2GANボコーダー統合の調査結果と実装内容について説明します。

## 結論

**Flow2GANはVocosと比較して明確なメリットがありません。**

| 観点 | Flow2GAN | Vocos | 結果 |
|------|----------|-------|------|
| 速度 | ~0.003秒 | ~0.003秒 | 同等 |
| 音質 | 良好（修正後） | 良好 | 同等 |
| 安定性 | 設定に注意が必要 | シンプル | Vocos優位 |
| ボトルネック影響 | 全体の5%未満 | 全体の5%未満 | N/A |

ボコーダーは推論時間全体の0.2〜5%程度しか占めていません。ボトルネックはZipVoiceモデル本体（約95%）であり、ボコーダーの最適化は全体性能にほとんど影響しません。

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
- GPU: CUDA対応デバイス
- テキスト: "The quick brown fox jumps over the lazy dog."
- プロンプト: 全テストで同一の参照音声を使用

### 速度比較

| ボコーダー | ボコーダー処理時間 | 全体処理時間 | ボコーダー割合 |
|-----------|------------------|-------------|--------------|
| Vocos | ~0.003秒 | ~2.8秒 | 0.1% |
| Flow2GAN 1-step | ~0.003秒 | ~2.8秒 | 0.1% |
| Flow2GAN 2-step | ~0.003秒 | ~2.8秒 | 0.1% |
| Flow2GAN 4-step | ~0.004秒 | ~2.8秒 | 0.1% |

**重要な発見:** ボコーダーはボトルネックではありません。ZipVoiceモデルの推論が処理時間の大部分を占めています。

### 音質比較

| ボコーダー | 音質 |
|-----------|------|
| Vocos | 良好、安定 |
| Flow2GAN 1-step | 良好（修正後） |
| Flow2GAN 2-step | 良好（修正後） |
| Flow2GAN 4-step | 良好（修正後） |

必要な修正を適用後、ボコーダー間で音質に大きな差はありません。

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
