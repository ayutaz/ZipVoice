# Unity Sentis向けONNXエクスポート ドキュメント

## 概要

ZipVoiceモデルをUnity Sentis（AIインファレンスエンジン）で実行するためのONNX変換ガイドです。
Unity Sentisには特定のONNXオペレーター制限があるため、標準のONNXエクスポートでは動作しません。

**対応バージョン**: Unity Sentis 2.1+

---

## 目次

1. [背景と課題](#背景と課題)
2. [技術的な問題と解決策](#技術的な問題と解決策)
3. [使用方法](#使用方法)
4. [制限事項](#制限事項)
5. [Unity側の実装ガイド](#unity側の実装ガイド)
6. [トラブルシューティング](#トラブルシューティング)

---

## 背景と課題

### Unity Sentisのオペレーター制限

Unity Sentisは以下の制限があります：

| 項目 | 要件 | ZipVoice標準ONNX |
|------|------|------------------|
| Opset version | 7-15 | 13 ✓ |
| テンソル次元 | 8未満 | 3 ✓ |
| `If` オペレーター | **非対応** | 6個 ✗ |
| `Loop` オペレーター | 非対応 | 未使用 ✓ |

**問題**: 標準のONNXエクスポートでは`If`オペレーターが6個含まれており、Unity Sentisで実行できません。

### `If`オペレーターの発生源

`If`オペレーターは`CompactRelPositionalEncoding`モジュールの`extend_pe()`メソッドから発生します。

```python
# zipvoice/models/modules/zipformer.py:983-992
def extend_pe(self, x: Tensor, left_context_len: int = 0) -> None:
    T = x.size(0) + left_context_len
    if self.pe is not None:                    # ← If オペレーター発生
        if self.pe.size(0) >= T * 2 - 1:       # ← If オペレーター発生
            self.pe = self.pe.to(dtype=x.dtype, device=x.device)
            return
    # ... PE計算処理
```

この条件分岐は、位置エンコーディング（PE）テンソルが十分なサイズかどうかをチェックし、
必要に応じて再計算するためのものです。動的なシーケンス長をサポートするために必要でした。

### `If`オペレーターの分布

| モデル | `If`数 | 発生箇所 |
|--------|--------|----------|
| Text Encoder | 1個 | `text_encoder.encoders.0.encoder_pos` |
| FM Decoder | 5個 | `fm_decoder.encoders.{0-4}.encoder_pos` |

---

## 技術的な問題と解決策

### 解決策の概要

`If`オペレーターを除去するため、以下の3段階のアプローチを実装しました：

1. **PE事前計算（Warmup）**: 最大シーケンス長に対応するPEを事前に計算
2. **`extend_pe`のno-op化**: `extend_pe`メソッドをラムダ関数で置換
3. **スクリプト化のスキップ**: `torch.jit.script()`によるPEモジュールのスクリプト化を回避

### 詳細な実装

#### ステップ1: PE事前計算

```python
# zipvoice/bin/onnx_export.py:406-425
MAX_SEQ_LEN = 2048

def warmup_and_disable_extend_pe(module, max_len):
    for name, child in module.named_modules():
        if hasattr(child, "extend_pe") and callable(child.extend_pe):
            # 十分なサイズでPEを事前計算
            child.extend_pe(torch.zeros(max_len))
            # extend_peをno-opラムダで置換
            child.extend_pe = lambda *args, **kwargs: None
```

**効果**:
- PE形状が`[1999, 48]`から`[4095, 48]`に拡張される
- `extend_pe`呼び出し時に条件分岐が実行されなくなる

#### ステップ2: スクリプト化のスキップ

元々の`convert_scaled_to_non_scaled`関数は、ONNXエクスポート時にPEモジュールを
`torch.jit.script()`でスクリプト化していました。これが`If`オペレーターを再導入する原因でした。

```python
# zipvoice/utils/scaling_converter.py:97-102
# 修正前
elif is_onnx and isinstance(m, CompactRelPositionalEncoding):
    d[name] = torch.jit.script(m)  # ← If が再導入される

# 修正後
elif is_onnx and isinstance(m, CompactRelPositionalEncoding) and not skip_pe_script:
    d[name] = torch.jit.script(m)
```

`skip_pe_script=True`を指定することで、スクリプト化をスキップし、
warmup済みのモジュールをそのままトレースします。

#### ステップ3: 順序の重要性

warmupは`convert_scaled_to_non_scaled`の**前**に実行する必要があります。

```python
# 正しい順序
warmup_and_disable_extend_pe(model, MAX_SEQ_LEN)  # 1. warmup
convert_scaled_to_non_scaled(model, is_onnx=True, skip_pe_script=True)  # 2. 変換
```

理由: `convert_scaled_to_non_scaled`後にwarmupを行うと、既にスクリプト化された
モジュールに対して操作することになり、効果がありません。

### 変更されたファイル

| ファイル | 変更内容 |
|----------|----------|
| `zipvoice/bin/onnx_export.py` | `--unity-sentis`、`--fm-seq-len`引数追加、warmup関数実装 |
| `zipvoice/utils/scaling_converter.py` | `skip_pe_script`パラメータ追加 |

---

## 使用方法

### 基本コマンド

```bash
# Unity Sentis互換ONNXをエクスポート
uv run python -m zipvoice.bin.onnx_export \
    --model-name zipvoice \
    --model-dir exp/zipvoice_moe_90h \
    --checkpoint-name best-valid-loss.pt \
    --onnx-model-dir exp/zipvoice_moe_90h_onnx_unity \
    --unity-sentis 1 \
    --fm-seq-len 512
```

### パラメータ説明

| パラメータ | デフォルト | 説明 |
|------------|-----------|------|
| `--unity-sentis` | 0 | 1でUnity Sentis互換モードを有効化 |
| `--fm-seq-len` | 200 | FM Decoderのトレース時シーケンス長 |
| `--onnx-model-dir` | exp | 出力ディレクトリ |

### 推奨シーケンス長

| 用途 | 推奨`--fm-seq-len` | 最大出力時間（概算） |
|------|-------------------|---------------------|
| 短いセリフ（ゲーム） | 256-512 | 2-5秒 |
| 通常の文章 | 512-1024 | 5-10秒 |
| 長いナレーション | 1024-2048 | 10-20秒 |

**計算式**: 出力時間 ≒ `seq_len × hop_length / sample_rate` = `seq_len × 256 / 24000`秒

### 出力ファイル

```
exp/zipvoice_moe_90h_onnx_unity/
├── text_encoder.onnx      # テキストエンコーダー (約17MB)
├── text_encoder_int8.onnx # INT8量子化版
├── fm_decoder.onnx        # Flow Matchingデコーダー (約477MB)
├── fm_decoder_int8.onnx   # INT8量子化版
├── model.json             # モデル設定
└── tokens.txt             # トークンファイル
```

### 検証コマンド

```bash
# If オペレーターが0個であることを確認
uv run python -c "
import onnx
from collections import Counter

for name in ['text_encoder.onnx', 'fm_decoder.onnx']:
    model = onnx.load(f'exp/zipvoice_moe_90h_onnx_unity/{name}')
    ops = Counter(node.op_type for node in model.graph.node)
    print(f'{name}: If={ops.get(\"If\", 0)}, Total={sum(ops.values())}')
"
```

期待される出力:
```
text_encoder.onnx: If=0, Total=1893
fm_decoder.onnx: If=0, Total=7324
```

---

## 制限事項

### 1. 固定シーケンス長

**問題**: Unity Sentisモードでエクスポートしたモデルは、エクスポート時に指定した
`--fm-seq-len`と異なるシーケンス長の入力を処理できません。

**エラー例**:
```
RuntimeException: The input tensor cannot be reshaped to the requested shape.
Input shape:{1,695,16}, requested shape:{-1,200,4,4}
```

**原因**: JITトレース時に内部のattention計算のreshape操作が固定サイズとして記録されるため。

**対策**:
- 推論時に入力を固定長にパディング
- 出力から余分な部分をトリミング

### 2. 計算効率の低下

固定長パディングにより、短いテキストでも長いテキストと同じ計算量が必要になります。

| 実際の長さ | 固定長 (512) | 無駄な計算 |
|-----------|-------------|-----------|
| 100 | 512 | 約80% |
| 256 | 512 | 約50% |
| 512 | 512 | 0% |

### 3. 動的シェイプ非対応

通常のONNX（`If`付き）では動的シェイプをサポートしていますが、
Unity Sentisモードでは入力シェイプが固定されます。

### 4. 追加コンポーネントが必要

完全なTTSパイプラインには以下も必要です（現時点では未実装）:

| コンポーネント | 状態 | 備考 |
|---------------|------|------|
| Text Encoder | ✓ 完了 | ONNX変換済み |
| FM Decoder | ✓ 完了 | ONNX変換済み |
| Vocoder (Vocos) | 未実装 | 別途ONNX変換が必要 |
| 日本語トークナイザー | 未実装 | Unity側で実装が必要 |
| Mel特徴量抽出 | 未実装 | プロンプト音声用 |

---

## Unity側の実装ガイド

### 必要なパッケージ

```
com.unity.sentis >= 2.1.0
```

### 基本的な推論フロー

```csharp
// 1. モデルのロード
var textEncoder = ModelLoader.Load("text_encoder.onnx");
var fmDecoder = ModelLoader.Load("fm_decoder.onnx");

// 2. ワーカーの作成
var textEncoderWorker = new Worker(textEncoder, BackendType.GPUCompute);
var fmDecoderWorker = new Worker(fmDecoder, BackendType.GPUCompute);

// 3. 入力の準備（パディング必要）
int fixedSeqLen = 512;  // エクスポート時と同じ値
var paddedInput = PadToFixedLength(input, fixedSeqLen);

// 4. 推論実行
textEncoderWorker.Schedule(paddedInput);
var textCondition = textEncoderWorker.PeekOutput();

// 5. FM Decoder（16ステップのEuler積分）
for (int step = 0; step < 16; step++) {
    float t = (float)step / 16;
    // ... Euler積分処理
}

// 6. 出力のトリミング
var output = TrimToOriginalLength(paddedOutput, originalLength);
```

### パディング処理の例

```csharp
Tensor PadToFixedLength(Tensor input, int targetLength) {
    int currentLength = input.shape[1];
    if (currentLength >= targetLength) {
        return input;  // または切り詰め
    }

    // ゼロパディング
    var padded = new Tensor(new int[] { 1, targetLength, input.shape[2] });
    // input の内容を padded にコピー
    // 残りはゼロで埋まっている
    return padded;
}
```

### 注意点

1. **メモリ管理**: `Dispose()`を適切に呼び出してテンソルを解放
2. **バックエンド選択**: GPUが利用可能なら`BackendType.GPUCompute`を使用
3. **非同期実行**: `Schedule()`は非同期、`PeekOutput()`で結果取得

---

## トラブルシューティング

### エラー: "If operator not supported"

**原因**: `--unity-sentis 1`を指定せずにエクスポートした

**解決策**:
```bash
uv run python -m zipvoice.bin.onnx_export \
    --unity-sentis 1 \
    ...
```

### エラー: "Input shape mismatch"

**原因**: 推論時のシーケンス長がエクスポート時と異なる

**解決策**:
1. 入力を固定長にパディング
2. または、適切な`--fm-seq-len`で再エクスポート

### エラー: "Out of memory"

**原因**: シーケンス長が長すぎる

**解決策**:
1. `--fm-seq-len`を小さくする
2. INT8量子化版（`*_int8.onnx`）を使用
3. バッチサイズを1に制限

### モデルサイズが大きい

**FM Decoder**: 約477MB（FP32）

**対策**:
- INT8量子化版を使用（約120MB）
- Unity側でストリーミングロード

---

## 今後の課題

1. **Vocoderの ONNX化**: Vocosモデルの変換
2. **動的シェイプ対応**: カスタムオペレーターの実装検討
3. **最適化**: モデルの軽量化、推論速度向上
4. **日本語トークナイザー**: Unity向けG2P実装

---

## 参考リンク

- [Unity Sentis サポートオペレーター一覧](https://docs.unity3d.com/Packages/com.unity.sentis@2.1/manual/supported-operators.html)
- [Unity Sentis ドキュメント](https://docs.unity3d.com/Packages/com.unity.sentis@2.1/manual/index.html)
- [ONNX Opset バージョン](https://onnx.ai/onnx/intro/converters.html)

---

## 変更履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|----------|
| 2025-12-29 | 1.0 | 初版作成。`If`オペレーター除去機能を実装 |
