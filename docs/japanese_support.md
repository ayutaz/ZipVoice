# ZipVoice 日本語サポート

本ドキュメントでは、ZipVoiceの日本語ファインチューニング方法について説明します。

## 概要

ZipVoiceは中国語・英語の事前学習モデルをベースに、日本語音声データでファインチューニングすることで日本語TTSを実現できます。

### 実装済み機能

| 機能 | ファイル | 説明 |
|------|---------|------|
| JapaneseTokenizer | `zipvoice/tokenizer/tokenizer.py` | pyopenjtalk-plusによる日本語G2P |
| JapaneseTextNormalizer | `zipvoice/tokenizer/normalizer.py` | 日英混合テキストの正規化 |
| 語彙拡張 | `zipvoice/bin/train_zipvoice.py` | 日本語トークン追加時のembedding拡張 |
| Docker環境 | `Dockerfile`, `docker-compose.yml` | k2ライブラリ対応のLinux環境 |

---

## クイックスタート

### 前提条件

- Docker Desktop（GPU対応）
- NVIDIA GPU（8GB VRAM以上推奨）
- 日本語音声データ（WAVファイル + 書き起こし）

### Step 1: データ準備

TSVファイルを作成します（タブ区切り）:

```
data/raw/custom_train.tsv
```

```tsv
utt_001	こんにちは、今日はいい天気ですね。	/path/to/audio/001.wav
utt_002	私の名前は田中です。	/path/to/audio/002.wav
...
```

### Step 2: 事前学習モデルのダウンロード

```bash
huggingface-cli download --local-dir download k2-fsa/ZipVoice zipvoice/model.pt zipvoice/model.json
```

### Step 3: マニフェスト生成

```bash
# Windows環境でUTF-8を強制
set PYTHONUTF8=1

uv run python -m zipvoice.bin.prepare_dataset \
    --tsv-path data/raw/custom_train.tsv \
    --prefix custom \
    --subset train \
    --num-jobs 4 \
    --output-dir data/manifests
```

### Step 4: トークン付加

```bash
uv run python -m zipvoice.bin.prepare_tokens \
    --input-file data/manifests/custom_cuts_train.jsonl.gz \
    --output-file data/manifests/custom_cuts_train_tokens.jsonl.gz \
    --tokenizer japanese
```

### Step 5: Fbank特徴量計算

```bash
uv run python -m zipvoice.bin.compute_fbank \
    --source-dir data/manifests \
    --dest-dir data/fbank \
    --dataset custom \
    --subset train_tokens \
    --num-jobs 4
```

### Step 6: パス変換（Windows→Linux）

Docker内で使用するためにパスを変換:

```bash
uv run python scripts/convert_paths.py
```

### Step 7: Dockerでファインチューニング

```bash
docker-compose build
docker-compose up zipvoice-train
```

### Step 8: 推論

```bash
# モデルディレクトリ準備
mkdir -p exp/custom/infer
cp exp/custom/checkpoint-10000.pt exp/custom/infer/model.pt
cp download/zipvoice/model.json exp/custom/infer/model.json
cp data/tokens_japanese_extended.txt exp/custom/infer/tokens.txt

# 推論実行
docker-compose run --rm zipvoice-shell python -m zipvoice.bin.infer_zipvoice \
    --model-dir exp/custom/infer \
    --tokenizer japanese \
    --prompt-wav data/prompt.wav \
    --prompt-text "プロンプト音声の書き起こし" \
    --text "合成したいテキスト" \
    --res-wav-path output.wav
```

---

## 詳細設定

### docker-compose.yml

```yaml
services:
  zipvoice-train:
    build:
      context: .
      dockerfile: Dockerfile
    image: zipvoice-japanese:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ./data:/workspace/data
      - ./download:/workspace/download
      - ./exp:/workspace/exp
    environment:
      - WANDB_API_KEY=${WANDB_API_KEY}
      - CUDA_VISIBLE_DEVICES=0
    command: >
      python -m zipvoice.bin.train_zipvoice
      --world-size 1
      --use-fp16 1
      --finetune 1
      --base-lr 0.0001
      --num-iters 10000
      --save-every-n 1000
      --max-duration 60
      --model-config download/zipvoice/model.json
      --checkpoint download/zipvoice/model.pt
      --tokenizer japanese
      --token-file data/tokens_japanese_extended.txt
      --dataset custom
      --train-manifest data/fbank/custom_cuts_train.jsonl.gz
      --dev-manifest data/fbank/custom_cuts_dev.jsonl.gz
      --exp-dir exp/custom
      --wandb-project zipvoice-japanese
```

### 訓練パラメータ

| パラメータ | 推奨値 | 説明 |
|-----------|-------|------|
| `--num-iters` | 10000-50000 | イテレーション数（データ量に応じて調整） |
| `--base-lr` | 0.0001 | ファインチューニング用学習率 |
| `--max-duration` | 60-120 | バッチあたりの音声長（秒） |
| `--finetune` | 1 | FixedLRScheduler使用 |
| `--save-every-n` | 1000 | チェックポイント保存間隔 |

### トークンファイル生成

新しいデータセットで訓練する場合、トークンファイルを生成:

```bash
uv run python -m zipvoice.bin.generate_tokens \
    --manifest data/manifests/custom_cuts_train.jsonl.gz \
    --tokenizer japanese \
    --output data/tokens_japanese_custom.txt
```

---

## 技術詳細

### JapaneseTokenizer

pyopenjtalk-plusを使用して日本語テキストを音素に変換します。

```python
from zipvoice.tokenizer import JapaneseTokenizer

tokenizer = JapaneseTokenizer(token_file="data/tokens_japanese.txt")
tokens = tokenizer.text_to_tokens("こんにちは")
# Output: ['k', 'o', 'N', 'n', 'i', 'ch', 'i', 'w', 'a']
```

### 日本語音素セット

| カテゴリ | 音素 |
|---------|------|
| 母音 | a, i, u, e, o |
| 子音 | k, s, t, n, h, m, y, r, w, g, z, d, b, p |
| 拗音 | ky, sh, ch, ny, hy, my, ry, gy, j, by, py |
| 特殊 | N（撥音）, cl（促音）, pau（ポーズ） |

### 語彙拡張

事前学習モデル（360トークン）に日本語トークン（25個）を追加し、385トークンに拡張:

```
2025-12-28 07:16:58,804 INFO [train_zipvoice.py:1008] Vocabulary size mismatch: checkpoint=360, model=385. Extending embedding layer.
```

---

## トラブルシューティング

### k2ライブラリがWindows非対応

**症状**: NaN勾配、訓練が発散

**原因**: k2ライブラリ（Swoosh活性化関数）がWindowsで利用不可

**解決策**: Docker環境（Linux）で訓練を実行

```bash
docker-compose up zipvoice-train
```

### UTF-8エンコーディングエラー

**症状**: `UnicodeDecodeError: 'cp932' codec can't decode`

**原因**: WindowsのデフォルトエンコーディングがCP932

**解決策**:
1. 環境変数設定: `set PYTHONUTF8=1`
2. `prepare_dataset.py`で明示的にUTF-8指定（修正済み）

### Docker内でファイルが見つからない

**症状**: `FileNotFoundError: data\\fbank\\...`

**原因**: Windowsパス（バックスラッシュ）がLinuxで認識されない

**解決策**: パス変換スクリプトを実行

```bash
uv run python scripts/convert_paths.py
```

---

## 訓練結果例

つくよみちゃんコーパス（100文、約30分）での訓練結果:

| 指標 | 値 |
|------|-----|
| イテレーション | 10,000 |
| エポック | 385 |
| 初期Validation Loss | 0.1237 |
| 最終Validation Loss | 0.0902 |
| 訓練時間 | 約3時間 |
| GPU使用メモリ | 約3.5GB |

---

## ファイル構成

```
ZipVoice/
├── zipvoice/
│   ├── tokenizer/
│   │   ├── tokenizer.py      # JapaneseTokenizer
│   │   └── normalizer.py     # JapaneseTextNormalizer
│   └── bin/
│       ├── train_zipvoice.py # 訓練スクリプト（語彙拡張対応）
│       ├── infer_zipvoice.py # 推論スクリプト
│       ├── prepare_dataset.py # マニフェスト生成（UTF-8対応）
│       ├── prepare_tokens.py  # トークン付加
│       └── compute_fbank.py   # Fbank特徴量計算
├── scripts/
│   └── convert_paths.py       # パス変換スクリプト
├── data/
│   ├── raw/                   # TSVファイル
│   ├── manifests/             # Lhotseマニフェスト
│   ├── fbank/                 # Fbank特徴量
│   └── tokens_japanese_extended.txt
├── download/
│   └── zipvoice/              # 事前学習モデル
├── exp/
│   └── custom/                # 訓練出力
├── Dockerfile
└── docker-compose.yml
```

---

## 参考リンク

- [ZipVoice GitHub](https://github.com/k2-fsa/ZipVoice)
- [pyopenjtalk-plus](https://github.com/sarulab-speech/pyopenjtalk-plus)
- [Lhotse](https://github.com/lhotse-speech/lhotse)
- [k2](https://github.com/k2-fsa/k2)
