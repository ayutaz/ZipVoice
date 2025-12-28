# ZipVoice 日本語サポート

本ドキュメントでは、ZipVoiceの日本語対応について説明します。

## 目次

1. [概要](#概要)
2. [事前学習の詳細](#事前学習の詳細)
3. [日本語対応の選択肢](#日本語対応の選択肢)
4. [大規模学習ガイド](#大規模学習ガイド)
5. [ファインチューニングガイド](#ファインチューニングガイド)
6. [技術詳細](#技術詳細)
7. [トラブルシューティング](#トラブルシューティング)

---

## 概要

ZipVoiceは中国語・英語の事前学習モデルをベースに、日本語音声データで訓練することで日本語TTSを実現できます。

### 論文情報

| 項目 | 内容 |
|------|------|
| タイトル | ZipVoice: Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching |
| arXiv | [2506.13053](https://arxiv.org/abs/2506.13053) |
| PDF | https://arxiv.org/pdf/2506.13053 |
| デモ | https://zipvoice.github.io |

### モデル仕様

| 項目 | 値 |
|------|-----|
| パラメータ数 | 123M |
| サンプルレート | 24kHz |
| 特徴量 | Vocos fbank (100次元) |
| アーキテクチャ | Zipformer-based Flow Matching |
| 推論ステップ | 16（デフォルト） |

### 実装済み機能

| 機能 | ファイル | 説明 |
|------|---------|------|
| JapaneseTokenizer | `zipvoice/tokenizer/tokenizer.py` | pyopenjtalk-plusによる日本語G2P |
| JapaneseTextNormalizer | `zipvoice/tokenizer/normalizer.py` | 日英混合テキストの正規化 |
| 語彙拡張 | `zipvoice/bin/train_zipvoice.py` | 日本語トークン追加時のembedding拡張 |
| Docker環境 | `Dockerfile`, `docker-compose.yml` | k2ライブラリ対応のLinux環境 |

---

## 事前学習の詳細

### 訓練データ（Emilia Dataset）

ZipVoiceはEmiliaデータセットで事前学習されています。

| 言語 | Emilia | Emilia-YODAS | 合計 |
|------|--------|--------------|------|
| 英語 | 46,800時間 | 92,200時間 | **139,000時間** |
| 中国語 | 49,900時間 | 300時間 | **50,300時間** |
| **日本語** | 1,700時間 | 1,100時間 | **2,800時間** |
| 韓国語 | 200時間 | 7,300時間 | **7,500時間** |
| ドイツ語 | 1,600時間 | 5,600時間 | **7,200時間** |
| フランス語 | 1,400時間 | 7,400時間 | **8,800時間** |
| **合計** | 101,700時間 | 113,900時間 | **215,600時間** |

> **重要**: 現在公開されているZipVoiceモデルは**英語と中国語のみ**で訓練されています。日本語（2,800時間）はEmiliaに含まれていますが、公式モデルには使用されていません。

### 公式訓練設定

`egs/zipvoice/run_emilia.sh` より:

```bash
python3 -m zipvoice.bin.train_zipvoice \
    --world-size 8 \
    --use-fp16 1 \
    --num-epochs 11 \
    --max-duration 500 \
    --lr-hours 30000 \
    --model-config conf/zipvoice_base.json \
    --tokenizer emilia \
    --token-file data/tokens_emilia.txt \
    --dataset emilia \
    --manifest-dir data/fbank \
    --exp-dir exp/zipvoice
```

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| world-size | 8 | GPU数 |
| use-fp16 | 1 | Mixed Precision |
| num-epochs | 11 | エポック数 |
| max-duration | 500 | バッチあたりの音声長（秒） |
| lr-hours | 30000 | 学習率スケジュール |

### 学習率の計算式

```python
lr_hours = 1000 * (train_hours ** 0.3)
```

| データ量 | lr-hours |
|---------|----------|
| 80時間 | 3,780 |
| 500時間 | 7,040 |
| 1,000時間 | 7,940 |
| 10,000時間 | 15,850 |
| 100,000時間 | 31,620 |

---

## 日本語対応の選択肢

### 選択肢の比較

| アプローチ | データ量 | 訓練時間 | 期待品質 | 推奨度 |
|-----------|---------|---------|---------|--------|
| ファインチューニング（小規模） | 0.5-10時間 | 数時間 | 基本 | 検証用 |
| ファインチューニング（中規模） | 50-100時間 | 1日 | 良好 | 実用 |
| 事前学習から（日本語特化） | 1,000-10,000時間 | 数日〜1週間 | 高品質 | 最適 |
| 多言語事前学習 | EN+ZH+JA | 1-2週間 | 高品質+多言語 | 汎用 |

### 推奨アプローチ: 段階的訓練

1. **フェーズ1: 検証（50-100時間）**
   - 小規模データでファインチューニング
   - 日本語発音の基本品質を確認

2. **フェーズ2: 改善（300-500時間）**
   - より多様な話者・テキスト
   - 発音の安定性向上

3. **フェーズ3: 本格訓練（1,000-10,000時間）**
   - 事前学習からやり直し
   - 製品品質のモデル

---

## 大規模学習ガイド

### データ量別の訓練設定

#### 80時間の場合

```bash
python3 -m zipvoice.bin.train_zipvoice \
    --world-size 1 \
    --use-fp16 1 \
    --num-iters 50000 \
    --max-duration 60 \
    --base-lr 0.0001 \
    --lr-hours 3780 \
    --model-config conf/zipvoice_base.json \
    --tokenizer japanese \
    --token-file data/tokens_japanese.txt \
    --dataset custom \
    --train-manifest data/fbank/custom_cuts_train.jsonl.gz \
    --dev-manifest data/fbank/custom_cuts_dev.jsonl.gz \
    --exp-dir exp/zipvoice_japanese_80h
```

| パラメータ | 値 |
|-----------|-----|
| イテレーション | 50,000 |
| エポック数 | ~10 |
| 推定訓練時間（1GPU） | 15-17時間 |

#### 1,000時間の場合（事前学習）

```bash
python3 -m zipvoice.bin.train_zipvoice \
    --world-size 4 \
    --use-fp16 1 \
    --num-iters 100000 \
    --max-duration 200 \
    --lr-hours 7940 \
    --model-config conf/zipvoice_base.json \
    --tokenizer japanese \
    --token-file data/tokens_japanese.txt \
    --dataset custom \
    --train-manifest data/fbank/custom_cuts_train.jsonl.gz \
    --dev-manifest data/fbank/custom_cuts_dev.jsonl.gz \
    --exp-dir exp/zipvoice_japanese_1000h
```

| パラメータ | 値 |
|-----------|-----|
| イテレーション | 100,000 |
| 推定訓練時間（4GPU） | 2-3日 |

#### 10,000時間の場合（日本語特化モデル）

```bash
python3 -m zipvoice.bin.train_zipvoice \
    --world-size 8 \
    --use-fp16 1 \
    --num-epochs 11 \
    --max-duration 500 \
    --lr-hours 15850 \
    --model-config conf/zipvoice_base.json \
    --tokenizer japanese \
    --token-file data/tokens_japanese.txt \
    --dataset custom \
    --train-manifest data/fbank/custom_cuts_train.jsonl.gz \
    --dev-manifest data/fbank/custom_cuts_dev.jsonl.gz \
    --exp-dir exp/zipvoice_japanese_10000h
```

| パラメータ | 値 |
|-----------|-----|
| エポック | 11 |
| 推定訓練時間（8GPU） | 5-7日 |

### 訓練時間の見積もり

#### 1 GPUの場合

| データ量 | イテレーション | 推定時間 |
|---------|--------------|---------|
| 0.5時間（100文） | 10,000 | 3時間 |
| 10時間 | 20,000 | 6時間 |
| 50時間 | 30,000 | 9時間 |
| 80時間 | 50,000 | 16時間 |
| 100時間 | 60,000 | 20時間 |

#### GPU数によるスケーリング

| GPU数 | 相対速度 |
|-------|---------|
| 1 | 1x |
| 2 | ~1.9x |
| 4 | ~3.6x |
| 8 | ~7x |

### データ準備

#### TSVファイル形式

```tsv
utt_001	こんにちは、今日はいい天気ですね。	/path/to/audio/001.wav
utt_002	私の名前は田中です。	/path/to/audio/002.wav
```

または時間指定付き:

```tsv
utt_001	こんにちは	/path/to/audio/001.wav	0.0	2.5
utt_002	今日はいい天気ですね	/path/to/audio/001.wav	2.5	5.0
```

#### 大規模データの前処理

```bash
# Stage 1: マニフェスト生成
python3 -m zipvoice.bin.prepare_dataset \
    --tsv-path data/raw/japanese_train.tsv \
    --prefix japanese \
    --subset train \
    --num-jobs 32 \
    --output-dir data/manifests

# Stage 2: トークン付加
python3 -m zipvoice.bin.prepare_tokens \
    --input-file data/manifests/japanese_cuts_train.jsonl.gz \
    --output-file data/manifests/japanese_cuts_train_tokens.jsonl.gz \
    --tokenizer japanese \
    --num-jobs 32

# Stage 3: Fbank特徴量計算
python3 -m zipvoice.bin.compute_fbank \
    --source-dir data/manifests \
    --dest-dir data/fbank \
    --dataset japanese \
    --subset train_tokens \
    --num-jobs 32

# Stage 4: トークンファイル生成
python3 -m zipvoice.bin.generate_tokens \
    --manifest data/manifests/japanese_cuts_train.jsonl.gz \
    --tokenizer japanese \
    --output data/tokens_japanese.txt
```

### 利用可能な日本語コーパス

| コーパス | 時間数 | 話者数 | 特徴 |
|---------|-------|--------|------|
| JSUT | 10時間 | 1名（女性） | 無料、高品質 |
| JVS | 30時間 | 100名 | 多話者、感情 |
| Common Voice JP | 50時間+ | 多数 | オープン |
| ReazonSpeech | 35,000時間 | 多数 | 大規模、ASR向け |
| Emilia JA | 2,800時間 | 多数 | TTS向け |

---

## ファインチューニングガイド

### 小規模ファインチューニング（既存モデルベース）

#### Step 1: 事前学習モデルのダウンロード

```bash
huggingface-cli download --local-dir download k2-fsa/ZipVoice zipvoice/model.pt zipvoice/model.json
```

#### Step 2: データ準備

TSVファイルを作成:

```
data/raw/custom_train.tsv
data/raw/custom_dev.tsv
```

#### Step 3: マニフェスト生成

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

#### Step 4: トークン付加

```bash
uv run python -m zipvoice.bin.prepare_tokens \
    --input-file data/manifests/custom_cuts_train.jsonl.gz \
    --output-file data/manifests/custom_cuts_train_tokens.jsonl.gz \
    --tokenizer japanese
```

#### Step 5: Fbank特徴量計算

```bash
uv run python -m zipvoice.bin.compute_fbank \
    --source-dir data/manifests \
    --dest-dir data/fbank \
    --dataset custom \
    --subset train_tokens \
    --num-jobs 4
```

#### Step 6: Dockerでファインチューニング

```bash
docker-compose build
docker-compose up zipvoice-train
```

#### Step 7: 推論

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
    --prompt-text "プロンプト音声の書き起こしテキスト" \
    --text "合成したいテキスト" \
    --res-wav-path output.wav
```

### docker-compose.yml設定

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

### 訓練パラメータガイド

| パラメータ | ファインチューニング | 事前学習 |
|-----------|-------------------|---------|
| `--finetune` | 1 | 0 |
| `--base-lr` | 0.0001 | 自動計算 |
| `--num-iters` | 10000-50000 | - |
| `--num-epochs` | - | 11 |
| `--max-duration` | 60-120 | 500 |
| `--checkpoint` | 指定 | なし |

---

## 技術詳細

### JapaneseTokenizer

pyopenjtalk-plusを使用して日本語テキストを音素に変換:

```python
from zipvoice.tokenizer import JapaneseTokenizer

tokenizer = JapaneseTokenizer(token_file="data/tokens_japanese.txt")
tokens = tokenizer.texts_to_tokens(["こんにちは"])
# Output: [['k', 'o', 'N', 'n', 'i', 'ch', 'i', 'w', 'a']]
```

### 日本語音素セット

| カテゴリ | 音素 |
|---------|------|
| 母音 | a, i, u, e, o |
| 子音 | k, s, t, n, h, m, y, r, w, g, z, d, b, p |
| 拗音 | ky, sh, ch, ny, hy, my, ry, gy, j, by, py |
| 特殊 | N（撥音）, cl（促音）, pau（ポーズ） |
| 無声化 | I, U（無声母音） |

### 語彙拡張の仕組み

事前学習モデル（360トークン）に日本語トークンを追加:

```python
# train_zipvoice.py より
if vocab_size != new_vocab_size:
    # Embedding層を拡張
    old_embedding = model.text_encoder.embedding.weight.data
    new_embedding = torch.nn.Embedding(new_vocab_size, embed_dim)
    new_embedding.weight.data[:vocab_size] = old_embedding
    # 新規トークンは既存の平均で初期化
    new_embedding.weight.data[vocab_size:] = old_embedding.mean(dim=0)
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

### プロンプト音声と出力が一致しない

**症状**: 生成音声が日本語に聞こえない

**原因**: `--prompt-text`がプロンプト音声の内容と一致していない

**解決策**: プロンプト音声の実際の書き起こしを使用

```bash
# 正しい例
--prompt-wav audio.wav \
--prompt-text "音声ファイルの実際の書き起こしテキスト"
```

### メモリ不足（OOM）

**症状**: `CUDA out of memory`

**原因**: バッチサイズが大きすぎる

**解決策**: `--max-duration`を下げる

```bash
--max-duration 30  # 60から30に下げる
```

---

## 訓練結果例

### つくよみちゃんコーパス（100文、約30分）

| 指標 | 値 |
|------|-----|
| イテレーション | 10,000 |
| エポック | 385 |
| 初期Validation Loss | 0.1237 |
| 最終Validation Loss | 0.0902 |
| 訓練時間 | 約3時間 |
| GPU使用メモリ | 約3.5GB |

### 期待される改善（データ量別）

| データ量 | 期待品質 |
|---------|---------|
| 0.5時間 | 基本的な発音、不安定 |
| 10時間 | 発音改善、一部不自然 |
| 50-100時間 | 良好な品質、実用レベル |
| 500時間以上 | 高品質、自然な発話 |
| 1,000時間以上 | 製品品質 |

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
├── egs/zipvoice/
│   ├── run_emilia.sh         # Emilia訓練レシピ
│   ├── run_custom.sh         # カスタム訓練レシピ
│   ├── run_finetune.sh       # ファインチューニングレシピ
│   └── run_japanese.sh       # 日本語訓練レシピ
├── scripts/
│   └── convert_paths.py      # パス変換スクリプト
├── data/
│   ├── raw/                  # TSVファイル
│   ├── manifests/            # Lhotseマニフェスト
│   ├── fbank/                # Fbank特徴量
│   └── tokens_japanese_extended.txt
├── download/
│   └── zipvoice/             # 事前学習モデル
├── exp/
│   └── zipvoice_japanese/    # 訓練出力
├── docs/
│   └── japanese_support.md   # 本ドキュメント
├── Dockerfile
└── docker-compose.yml
```

---

## 参考リンク

- [ZipVoice GitHub](https://github.com/k2-fsa/ZipVoice)
- [ZipVoice Paper (arXiv)](https://arxiv.org/abs/2506.13053)
- [ZipVoice Demo](https://zipvoice.github.io)
- [Emilia Dataset (HuggingFace)](https://huggingface.co/datasets/amphion/Emilia-Dataset)
- [pyopenjtalk-plus](https://github.com/sarulab-speech/pyopenjtalk-plus)
- [Lhotse](https://github.com/lhotse-speech/lhotse)
- [k2](https://github.com/k2-fsa/k2)
