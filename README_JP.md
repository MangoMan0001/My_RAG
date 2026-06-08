*This project has been created as part of the 42 curriculum by ayhirose.*

# RAG against the machine

### 前提知識
1. RAG
Retrieving Augmented Generation (RAG) \
検索 拡張(強化) 生成 \

2. BM25

### Description
LLM（大規模言語モデル）を用いた課題です。\
自然言語のプロンプトを解析し、既存の学習にはない新たな外部ソースを元とした回答の生成を目的としています。

個人目標
- CPU単独での高速な回答


使用したパッケージ
>パッケージ管理は`python uv`を使用しています
```
accelerate>=1.13.0
bm25s>=0.3.8
fire>=0.7.1
flake8>=7.3.0
flake8-bugbear>=25.11.29
flake8-pyproject
langchain>=1.2.17
langchain-community>=0.4.1
langchain-unstructured>=1.0.1
mypy>=1.19.1
pep8-naming>=0.15.1
pydantic>=2.12.5
pystemmer>=3.0.0
torch>=2.11.0
tqdm>=4.67.3
flake8-docstrings
llama-cpp-python>=0.3.25
```


ディレクトリ構成
```
.
├── Makefile
├── README.md             # 英語ドキュメント (要件指定)
├── README_JP.md          # 日本語ドキュメント
├── pyproject.toml        # 依存ライブラリやリンター(flake8, mypy)の設定
├── uv.lock               # uvの依存関係ロックファイル
├── .gitignore
├── .python-version
│
├── src/
│   └── student/
│       ├── __main__.py   # 実行時のモジュールエントリーポイント
│       ├── __init__.py
│       ├── cli.py        # CLIコマンドの定義 (Fire)
│       └── core/         # RAGシステムのコアロジック
│           ├── answer.py     # 回答生成モジュール (llama.cpp)
│           ├── evaluater.py  # 評価モジュール
│           ├── indexer.py    # チャンク化・インデックス作成モジュール
│           ├── models.py     # Pydanticを用いたデータモデル定義
│           └── searcher.py   # BM25を用いた検索モジュール
│
├── data/                 # データ格納ディレクトリ
│   ├── datasets/         # 質問データセット (Answered/Unanswered)
│   ├── model/            # ダウンロードしたLLMモデルファイル (GGUF)
│   ├── output/           # 検索結果や生成された回答JSONの出力先
│   ├── processed/        # チャンクデータやBM25のインデックス保存先
│   └── raw/              # インデックス作成元の生データ (vLLMリポジトリ等)
│
└── moulinette_pkg/       # 評価システム用パッケージ (提供ファイル)
```



### Instructions

このプログラムは Python 3.10以上 での実行が前提です。パッケージ管理には uv を使用しています。

1. **インストール**
```bash
make install
```
仮想環境（.venv）を構築し、必要な依存関係をインストールします。\
課題で必須になる、Qwen3-0.6B-Q8_0.ggufをインストールも同時に行います。(`data/model`)

2. **実行**
```bash
make run
```
メインプログラムのヘルプが表示されます。\
実行方法は多岐に渡るためrunコマンドではヘルプが表示されます。
>**注意**\
プログラムは依存関係のないグローバル環境では必ずしも正しく実行されるとは限りません。\
先に`make install`にてインストールした`.venv`を通して実行してください。\
実行方法
`uv run <実行プログラム>`

**indexの作成** \
BM25により検索可能なインデックスを生成します。\
indexerモジュールはデフォルトで"./data/raw/vllm-0.10.1"を対象データとして参照します。
```bash
uv run python -m student index --max_chunk_size 2000
```

```bash
使用可能なフラグ
--index_dir: str = "data/processed"	保存ファイルを指定
--max_chunk_size: int = 2000		最大チャンクサイズを指定
```


**検索**\
BM25によりクエリと関連性の高いソースをインデックスから検索する。\
検索結果は`MinimalSearchResults`形式でターミナルに出力されます。
>形式については`src/student/core/model.py`を参照してください。\
>**注意** この操作を行う前にindexが生成されている必要があります。
```bash
uv run python -m student search "How to configure OpenAI server?"
```

```bash
使用可能なフラグ
--k: int = 5						検索するソース数
--query: str						検索時のクエリ
--index_dir: str = "data/processed"	参照するインデックス
--question_id: str = "q1"			クエリを紐づく固有のID
```


**データセット検索**\
`RagDataset`形式のデータセットファイルを読み込み、まとめて検索を行います。\
検索結果は`StudentSearchResults`形式で指定されたフォルダに記録されます。
>形式については`src/student/core/model.py`を参照してください。\
>**注意** この操作を行う前にindexが生成されている必要があります。
```bash
uv run python -m student search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results
```

```bash
使用可能なフラグ
--k: int = 10
--index_dir: str = "data/processed"
--save_directory: str = "data/output/search_results"								検索結果の保存先
--dataset_path: str = 'data/datasets/UnansweredQuestions/dataset_docs_public.json'	使用するデータセットファイルのパス
```


**回答**\
渡せされたクエリに関連するソースをもとにLLM(`Qwen3-0.6B`)の回答生成します。\
回答結果は`StudentSearchResultsAndAnswer`形式でターミナルに出力されます。
>形式については`src/student/core/model.py`を参照してください。\
>**注意** この操作を行う前にindexが生成されている必要があります。

```bash
uv run python -m student answer "How to configure OpenAI server?" --k 10
```

```bash
使用可能なフラグ
--query: str
--k: int = 10
--index_dir: str = "data/processed"
```


**データセット回答**\
`StudentSearchResults`形式のデータセットファイルを読み込み、まとめて回答を行います。\
回答結果は`StudentSearchResultsAndAnswer`形式で指定されたフォルダに記録されます。
>形式については`src/student/core/model.py`を参照してください。\
>**注意** この操作を行う前にindexが生成されている必要があります。

```bash
uv run python -m student answer_dataset --student_search_results_path data/output/search_results/dataset_docs_public.json --save_directory data/output/search_results_and_answer
```

```bash
使用可能なフラグ
--save_directory: str = "data/output/search_results_and_answer"
--student_search_results_path: str = "data/output/search_results/dataset_docs_public.json"
```


**評価**\
検索されたソースの整合性を`Recall@k`で指標する。\
評価内容は`moulinette`と同一です。\
`StudentSearchResultsAndAnswer`形式のデータセットファイルを読み込み、まとめて回答を行います。\
結果はターミナルに出力されます。
>形式については`src/student/core/model.py`を参照してください。\
>**注意** この操作を行う前にindexが生成されている必要があります。

>性能要件(課題pdf引用)
>- インデックス作成時間：最大5分
>- コールドスタート時のレイテンシ：最大60秒（システム起動後の最初の検索。モデル読み込みを含む）
>- ウォーム状態での検索スループット：1000件の質問に対し最大90秒（コールドスタート後）
>- Recall@5：ドキュメントの質問で80%、コードの質問で50%

```bash
uv run python -m moulinette evaluate_student_search_results	--student_answer_path data/output/search_results/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000
```

```bash
使用可能なフラグ
--k: int = 10
--max_context_length: int = 2000													クエリ毎に評価されるテキストの長さ
--student_answer_path: str = "data/output/search_results/dataset_docs_public.json"	評価対象となるファイルパス
--dataset_path: str = "data/datasets/AnsweredQuestions/dataset_docs_public.json"	評価基準となるファイルパス
```


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
