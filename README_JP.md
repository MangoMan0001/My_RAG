*This project has been created as part of the 42 curriculum by ayhirose.*

# Call Me Maybe

### Description
自然言語のプロンプトを解析し、LLM（大規模言語モデル）に構造化された関数呼び出し（JSON形式）を生成させるプログラムです。
LLMの出力確率（Logits）をトークンごとに直接操作する「制約付きデコーディング（Constrained Decoding）」を用いて、自然言語ではなくJSONオブジェクトを生成することを目的としています。

ディレクトリ構成
```
.
├── Makefile
├── README.md
├── README_ja.md          # 日本語ドキュメント
├── pyproject.toml        # 依存ライブラリやリンター(flake8, mypy)の設定
│
├── src/
│   ├── call_me_maybe.py  # エントリーポイント
│   ├── model.py          # 制約付きデコーディングのコアロジックとステートマシン
│   ├── parser.py         # Pydanticを用いた入力バリデーション
│   └── utils.py          # 引数解析
│
├── llm_sdk/              # 提供されたLLM操作用ラッパー
└── data/
    ├── input/            # テスト用の関数定義とプロンプトJSON
    └── output/           # 生成された結果の出力先
```

### Instructions

このプログラムは Python 3.10以上 での実行が前提です。パッケージ管理には uv を使用しています。

1. **インストール**
```bash
make install
```
仮想環境（.venv）を構築し、pydantic や numpy などの必要な依存関係をインストールします。

2. **実行**
```bash
make run
```
デフォルトの設定でプログラムが実行され、data/output/function_calls.json に結果が出力されます。

```bash
uv run python src/call_me_maybe.py -f edge_functions.json -i edge_inputs.json
```

```bash
-f or -functions_definition : 関数定義ファイルを指定
-i or –input : 入力ファイルを指定
-o or –output : 出力ファイルを指定
```
使用可能なフラグ

```bash
sorce .venv/bin/activate
```
（プロジェクトのPDFに厳密に従って実行する場合は、事前に `source .venv/bin/activate` を実行して仮想環境を有効にしてください。仮想環境を終了するには、`deactivate` を実行してください。）

3. **他の `Makefile` コマンド**
```bash
make lint
make lint-strict
```
flake8 と mypy による静的型解析を実行します。

```bash
make debug
```
pdb を使用したデバッグモードで実行します。

```bash
make clean
```
キャッシュファイルを削除します。
仮想環境の削除も含むfcleanも同様に使用できます。

## Additional sections

### Algorithm explanation（アルゴリズムの解説）
プロンプトにモデルから選択されたトークンを繋げていく言語生成の流れの中、選択されるトークンをプログラムから制限することで出力をコントロールしています。出力したいフォーマットはプロンプトによってほぼ変化しません。なのでそのフォーマットではないものを弾くことで大まかな出力はコントロールできます。難しいのは関数名や引数の値とその型です。関数名はモデルがトークン化できる最小単位を２次元リストとして管理することで、インデックスで選ばれる関数を絞り込むように制御することができます。引数は関数が決まれば型も決まるためNUMBERであれば-や0~9と使用できるトークンを絞り込むことができます。あとはトークンの区切り方の癖に気をつけながらJSONの形が崩れないように作りました。

## Design decisions（設計の決定事項）
課題より指定もありましたが、バリデーションにおいてはPydanticを使用しています。
引数取得はargparse、計算速度向上のためのNumpyを使用しています。
しかし、このプログラムにおいてのボトルネックは事前に用意されていた__init__.pyに含まれる関数の実行速度にあったため課題全体での実行速度には寄与していません。

## Performance analysis（パフォーマンス分析）
制約付きデコーディングの制御により、指定されたスキーマに合致するJSONを生成します。
速度に関しては前述の通り__init__.pyの実行スピードを上回っています。
テスト開発環境では１関数につき1分の処理スピードです。

## Challenges faced（直面した課題と解決策）
Pythonのjson.loadsによってエスケープが消費され、LLMに「生の改行」や「素のダブルクォート」が渡されることでJSONが破壊される問題に直面しました。これに対し、特定の文字を含むトークンを絶対拒絶リストに追加することで解決しました。

## Testing strategy（テスト戦略）
提供された標準テストに加え、以下の「Edgeケース」を含む独自のテスト用JSONを使用しました。
- 空文字や極端に長い文字列
- 特殊記号や絵文字（👾）、エスケープ文字の連続
- スキーマと異なる型を要求する「不自然なプロンプト（Wrong types）」
- 曖昧で関数を特定しづらいプロンプト

## Example(使用例)
```
1. Request logits
--- Pre Limit Token ---
ID:   5209 | Score:   18.86 | Token: ' Please'
ID:   4710 | Score:   17.98 | Token: ' \n\n'
ID:    220 | Score:   17.57 | Token: ' '
ID:   3555 | Score:   17.00 | Token: ' What'
ID:   7281 | Score:   16.89 | Token: ' Also'
2. <<< Constrain logits >>>
--- Post Limit Token ---
ID:    515 | Score:    4.81 | Token: '{\n'
ID:      0 | Score:    -inf | Token: '!'
ID:      1 | Score:    -inf | Token: '"'
ID:      2 | Score:    -inf | Token: '#'
ID:      3 | Score:    -inf | Token: '$'
3. Select the token with the highest score
4. Generate the restriction token to be used next
```
制約デコーディングを1トークンごとに現在の状態に分けて出力します。

### Resources

AI
- 制約付きデコーディングにおけるLogits操作のアルゴリズム設計とデバッグの壁打ち。
- `flake8` および `mypy` のエラーログ解析と `pyproject.toml` の最適化。
- DocstringおよびREADMEの英訳、構成支援。
