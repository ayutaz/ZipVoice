# 日本語サポート調査結果

## 現状

ZipVoiceは現在、中国語（Mandarin）と英語をサポートしています。日本語は公式にはサポートされておらず、データ準備段階で日本語を含む発話は明示的に除外されています（`egs/zipvoice/local/preprocess_emilia.py`）。

## 日本語対応の技術的可能性

日本語対応は技術的に実現可能です。以下のアプローチが考えられます。

### アプローチ1: EspeakTokenizer使用（簡易）

espeak-ngは日本語（`ja`）をサポートしているため、既存の`EspeakTokenizer`をそのまま使用できます。

```bash
uv run python -m zipvoice.bin.train_zipvoice \
    --tokenizer espeak \
    --lang ja \
    --token-file data/tokens_ja.txt \
    ...
```

**メリット**: 実装不要、すぐに試せる
**デメリット**: 日本語の発音品質がespeak-ng依存

### アプローチ2: pyopenjtalk使用（推奨）

日本語G2P（Grapheme-to-Phoneme）にpyopenjtalkを使用する専用トークナイザを実装します。

**必要な変更**:

1. `JapaneseTokenizer`クラスの実装（`zipvoice/tokenizer/tokenizer.py`）
2. `JapaneseTextNormalizer`の追加（`zipvoice/tokenizer/normalizer.py`）
3. 訓練/推論スクリプトへの`--tokenizer japanese`オプション追加
4. 日本語トークンファイル生成スクリプト
5. ファインチューニングレシピ

### 日英混合テキスト対応

既存の`EmiliaTokenizer`パターンに従い、日本語と英語を自動判別して処理することが可能です：

- 日本語セグメント → pyopenjtalk
- 英語セグメント → espeak (en-us)

## pyopenjtalkの音素セット

pyopenjtalkは以下の音素を出力します：

| カテゴリ | 音素 |
|---------|------|
| 母音 | a, i, u, e, o |
| 子音 | k, s, t, n, h, m, y, r, w, g, z, d, b, p |
| 拗音 | ky, sh, ch, ny, hy, my, ry, gy, j, by, py |
| その他 | ts, f, N（撥音）, cl（促音）, q（声門閉鎖） |
| 韻律 | pau, sil |

### トークン設計

既存のespeakトークンとの衝突を避けるため、日本語音素には`_JP`サフィックスを付与：

```
a_JP    256
i_JP    257
k_JP    258
N_JP    259  # 撥音
cl_JP   260  # 促音
...
```

## ファインチューニングワークフロー

1. **データ準備**: TSVファイル作成（`utt_id\tテキスト\twav_path`）
2. **マニフェスト生成**: `uv run python -m zipvoice.bin.prepare_dataset`
3. **トークン追加**: `uv run python -m zipvoice.bin.prepare_tokens --tokenizer japanese`
4. **Fbank計算**: `uv run python -m zipvoice.bin.compute_fbank`
5. **ファインチューニング**: `uv run python -m zipvoice.bin.train_zipvoice --finetune 1 --tokenizer japanese`
6. **推論**: `uv run python -m zipvoice.bin.infer_zipvoice --tokenizer japanese`

## 語彙サイズの問題

事前学習済みモデルは固定の語彙サイズ（espeak + 中国語ピンイン）を持っています。日本語トークンを追加すると語彙が拡張されるため、ファインチューニング開始時にembedding層の拡張が必要です。

**対策**:
- 既存トークンの重みは保持
- 新しい日本語トークンはランダム初期化

この処理は`zipvoice/bin/train_zipvoice.py`のチェックポイント読み込み部分に追加が必要です。

## 必要な依存関係

```bash
uv add pyopenjtalk
```

## 日本語文字の検出（Unicode範囲）

```python
def is_japanese(char: str) -> bool:
    # ひらがな
    if '\u3040' <= char <= '\u309f':
        return True
    # カタカナ
    if '\u30a0' <= char <= '\u30ff':
        return True
    # 漢字（CJKと共通、文脈で判断）
    if '\u4e00' <= char <= '\u9faf':
        return True
    return False
```

## 参考ファイル

- `zipvoice/tokenizer/tokenizer.py` - トークナイザ実装
- `zipvoice/tokenizer/normalizer.py` - テキスト正規化
- `egs/zipvoice/run_finetune.sh` - ファインチューニングレシピ
- `egs/zipvoice/local/preprocess_emilia.py` - 日本語除外ロジック（L97-149）

## 実装の優先順位

1. **Phase 1**: JapaneseTokenizer + JapaneseTextNormalizer（日英混合対応含む）
2. **Phase 2**: 訓練/推論スクリプト修正
3. **Phase 3**: トークンファイル生成 + ファインチューニングレシピ
