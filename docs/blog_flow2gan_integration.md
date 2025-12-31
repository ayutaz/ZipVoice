# ZipVoiceのボコーダーをFlow2GANに置き換えて高速化できるか検証してみた

## はじめに

ZipVoiceは、Flow Matchingを使った高速・高品質なゼロショットText-to-Speech（TTS）システムです。
今回、ボコーダー部分をFlow2GANに置き換えることで推論速度を向上できないか検証しました。

**結論から言うと、Flow2GANに置き換えるメリットはありませんでした。**
むしろVocosの方が6〜22倍高速という結果になりました。

## 検証環境

| 項目 | 内容 |
|-----|------|
| OS | Windows |
| GPU | NVIDIA CUDA対応 |
| Python | 3.11 |
| パッケージ管理 | uv |

## ボコーダーとは

TTSシステムは以下のパイプラインで動作します：

```
テキスト → モデル推論 → メルスペクトログラム → ボコーダー → 波形
```

ボコーダーはメルスペクトログラムを音声波形に変換するコンポーネントです。
ZipVoiceはデフォルトでVocosを使用しています。

## Flow2GANとは

Flow2GANはFlow MatchingとGANファインチューニングを組み合わせたボコーダーで、1〜4ステップでの推論が可能です。

**主な特徴:**
- 1〜4ステップ推論（設定可能）
- Flow Matching + GANファインチューニング
- マルチブランチConvNeXtアーキテクチャ

ZipVoiceとFlow2GANは同じMel-Spectrogram仕様（24kHz、100 bins、FFT 1024、hop 256）を使用しているため、理論上は置き換え可能です。

## Flow2GANの統合

### uv addでの依存関係追加

Flow2GANをローカルから追加：

```bash
uv add flow2gan --editable "C:\path\to\Flow2GAN"
```

**注意点**: Flow2GANには`pyproject.toml`がなかったため、新規作成が必要でした。また、`librosa`はPython 3.11でnumba依存関係の問題があるため、推論に不要なものはオプショナル依存に移動しました。

### ボコーダーパッケージの作成

VocosとFlow2GANを切り替えられるように、統一インターフェースを作成しました：

```python
from abc import ABC, abstractmethod
import torch

class BaseVocoder(ABC):
    @abstractmethod
    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        """(B, 100, T) -> (B, 1, samples)"""
        pass

def get_vocoder(vocoder_type: str = "vocos", **kwargs) -> BaseVocoder:
    if vocoder_type == "vocos":
        return VocosVocoder(**kwargs)
    elif vocoder_type == "flow2gan":
        return Flow2GANVocoder(**kwargs)
```

### 推論スクリプトへの引数追加

```python
parser.add_argument("--vocoder-type", type=str, default="vocos",
                    choices=["vocos", "flow2gan"])
parser.add_argument("--vocoder-n-steps", type=int, default=2)
```

## 発生した問題

### 問題1: モデル選択バグ

Flow2GANはステップ数ごとに専用モデルが必要でした：
- 1-step → `libritts-mel-1-step`
- 2-step → `libritts-mel-2-step`
- 4-step → `libritts-mel-4-step`

最初は4-stepモデルを固定で使っていたため、1-step/2-stepで音声が完全に壊れて「ピー」という音しか出ませんでした。

**解決策**: `n_timesteps`に応じてモデルを自動選択するように修正：

```python
if hf_model_name is None:
    hf_model_name = f"libritts-mel-{n_timesteps}-step"
```

### 問題2: 音割れ（ハードクリッピング）

モデル選択を修正後も、すべてのステップ設定で音割れが発生しました。

**原因**: Flow2GANのGAN訓練時は`clamp_pred=False`で訓練されています。Discriminatorが内部で正規化するため、Generatorは[-1, 1]を超える値を出力するように学習されています。推論時に`clamp_pred=True`でハードクリッピングすると、音割れが発生します。

**解決策**: Discriminatorと同じピーク正規化を適用：

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

## ベンチマーク結果

推論パイプラインの各処理ステップの時間を計測しました。

### 推論パイプライン詳細時間内訳

#### Vocos

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

### ボコーダー速度比較

| ボコーダー | 処理時間 | 全体に占める割合 |
|-----------|---------|----------------|
| **Vocos** | **4.29 ms** | **0.2%** |
| Flow2GAN 1-step | 25.76 ms | 1.3% |
| Flow2GAN 2-step | 48.24 ms | 2.4% |
| Flow2GAN 4-step | 94.89 ms | 4.7% |

**Vocosが6〜22倍高速**という結果になりました。

### 音質比較

修正後の音質は、VocosとFlow2GAN（全ステップ）で大きな差はありませんでした。

## 結論

### 重要な発見

1. **ZipVoiceモデルがボトルネック**: 推論時間の93〜98%を占める
2. **Vocosが最速**: 4.29msでFlow2GANより6〜22倍高速
3. **ボコーダー最適化の効果は限定的**: ボコーダーを0msにしても全体の2〜5%しか改善しない
4. **Flow2GANはステップ数に比例して遅くなる**: 1-step→2-step→4-stepで約2倍ずつ増加

### 学んだこと

- **最適化する前にまずプロファイリングすべき**: どこがボトルネックかを把握せずに最適化しても効果は薄い
- **ボトルネックを特定してから対策を考える**: 今回の場合、ボコーダーではなくZipVoice本体を最適化すべき
- **全体の2%しか占めない部分を最適化しても効果は薄い**: 最適化の優先順位を正しく付ける必要がある

### 今後の方針

ボコーダーの最適化よりも、ZipVoiceモデル本体の最適化（TensorRT、ONNX等）に注力すべきという結論に至りました。

## 参考リンク

- [ZipVoice GitHub](https://github.com/k2-fsa/ZipVoice)
- [Flow2GAN GitHub](https://github.com/k2-fsa/Flow2GAN)
- [Vocos](https://github.com/gemelo-ai/vocos)
