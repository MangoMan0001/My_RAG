*This project has been created as part of the 42 curriculum by ayhirose.*

# RAG against the machine

### 前提知識
**1. RAG**\
Retrieving Augmented Generation (RAG) \
検索 拡張(強化) 生成 \
モデルに学習時の記憶したデータ以外の知識を持たせて回答を得る方法。 \
以下の順に構成される。

- インデックス作成
追加したいデータ（課題では.mdと.pyファイル）を辞書化する。検索のためのインデックスを作成する。

- 検索
`TF-IDF`や`bm25`、セマンティック検索などの自然言語処理技術を用いてプロンプトから重要となる単語とその関連性の高いデータを辞書から選び出す。

- 強化/拡張
選ばれたデータとすでに学習済みの記憶を組み合わせる。課題では検索時の情報のみに依存するため省略される。

- 生成
強化/拡張した記憶から回答を生成する。LLMはこの演算が非常に重たく、多くのモデルはGPUのサポートにより並列処理されることで高速化されている。

**2. TF-IDF**\
自然言語処理で使われる評価手順の一つ。\
「その文章の中に何度も出てくる単語」で、かつ「他の文章にはあまり出てこないレアな単語」ほど重要だ！と判断する古典的な計算式。\
長文であればあるほど評価が高くなってしまうという欠点がある
>単語の出現頻度(TF) * 単語の希少性(IDF)

**3. BM25**\
自然言語処理で使われる評価手順の一つ。\
`TF-IDF`が抱える長文時のデメリットを克服した手法。\
この手法はある単語の出現頻度が高くなりすぎた場合でも評価に影響しない。
>bm25アルゴリズムを動かすパッケージ`bm25s`


**4. Recall@k**\
自然言語処理における検索システムの評価項目。\
データセットの正解に対してモデルが提案した上位k件の中に含まれる正解の割合。

### Description
LLM（大規模言語モデル）を用いた課題です。\
自然言語のプロンプトを解析し、既存の学習にはない新たな外部ソースを元とした回答の生成を目的としています。

**個人目標**
- CPU単独での高速な回答


**使用したパッケージ**
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


**ディレクトリ構成**
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
uv run python -m student evaluate --student_answer_path data/output/search_results/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000
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

###	システムアーキテクチャ
RAGパイプラインの構成要素と、それらの相互作用について\
エントリーポイントととなるコマンドラインは`python fire`パッケージです。各Pythonモジュールを管理しています。\
RAGパイプラインは`langchain`フレームワークを土台にインデックス作成と検索を`bm25s`パッケージ。Qwenによる回答生成は`llama_cpp`にて実装しています。

###	チャンキング戦略
ドキュメントのセグメンテーションに対するアプローチ
- 各チャンク間のオーバラップを100文字設けたことで、前後関係のわかりにくいぶつ切りのチャンクを抑えることができました。
- `bm25s`の`stopwords='en'`を設定。意味を持たない頻出単語('a' 'is' 'the'など)を取り除き、速度向上、精度向上に寄与しました。
- 単語の接尾接頭を取り除き、語幹に還元するステミングを追加。bm25の精度向上に寄与しました。(`Stemmer`)
>ステミングとは 複数形や活用形（例："fishing", "fished", "fisher"）を同一の単語（例："fish"）として処理

###	検索手法
検索アルゴリズムとランキングメカニズムの詳細\
bm25アルゴリズムで検索したチャンクのIDをローカルで既に保存したインデックスから参照します。これを`k`回繰り返します。\
`bm25s`の`retrieve`メソッドは一度に`k`つのファイルを検索します。listのインデックスが低い順からクエリとの関連性が高いため、それを元に順にランキングを作成します。

###	性能分析
Recall@kスコアおよびシステムの性能について
>性能要件(課題pdf引用)
>- インデックス作成時間：最大5分
>- コールドスタート時のレイテンシ：最大60秒（システム起動後の最初の検索。モデル読み込みを含む）
>- ウォーム状態での検索スループット：1000件の質問に対し最大90秒（コールドスタート後）
>- Recall@5：ドキュメントの質問で80%、コードの質問で50%

インデックス作成は10秒以下で作成が完了します。主に生データの`vllm-0.10.1`から今回の課題に使用される`.py` `.md`のみを抜粋している点と使用した`bm25s`の基盤となる`Numpy`や`Scipy`はC言語で作成されている点から遅延が抑えられていると思われます。\
検索も`bm25s`を使用しています。\
Recall@kは`bm25s`標準検索のみのスコアと、チャンキング戦略の項目で挙げた`ステミング`や`stopwods`の実装後のスコアとの間に大きく差があるため精度向上に寄与していたことがわかります。
```python
<pre score> .md
Recall@1: 0.44
Recall@3: 0.46
Recall@5: 0.51
Recall@10: 0.56
```
```python
<post score> .md
Recall@1: 0.72
Recall@3: 0.79
Recall@5: 0.82
Recall@10: 0.86
```
```python
<pre score> .py
Recall@1: 0.14
Recall@3: 0.19
Recall@5: 0.25
Recall@10: 0.28
```
```python
<post score> .py
Recall@1: 0.41
Recall@3: 0.51
Recall@5: 0.55
Recall@10: 0.59
```


###	設計上の判断
実装における重要な選択について\
- 初めて使用するパッケージだったため、略さずに提供メソッドをそのまま記述することにしました。\
- `flake8`の標準設定の最大列文字数の79文字では意図しない改行が多発し、著しく可読性が下がるため上限値を120文字に設定しました。

###	直面した課題
遭遇した困難と解決策の記録\

**回答速度**
- 問題点：課題PDFで推奨されている`transformers`などのモデル操作パッケージはGPUを使用した並列回答をサポートしているものがほとんどであり、それらのパッケージをCPUのみの校舎PCで実行した場合1回答につき60秒以上かかり、熱による処理速度の低下も相まりデータセット100問の回答に2時間以上かかっていた。

- 対策：モデル操作をCPU動作前提のC++で作成された`llama_cpp`パッケージに変更。ローカルLLMをGGUFデータ形式(GPT-Generated Unified Format)で8bitのローカルファイルとして実行前にインストールを行う仕様に変更。

- 結果：1問4秒に短縮、100問が6〜7分で処理が可能になった。

**Recall@kスコアの低迷**
- 問題点：`Recall@k`が性能要件に満たなかった。

- 対策：`indexer`モジュールで実装していた`bm25s`の`ステミング`と`stopwords`が`searcher`モジュールで引き継がれていなかった。チャンキングは機能していたが、検索時の読み込み時にも設定が引き継がれるように変更した。

- 結果：`.md`で1.5倍、`.py`で2.2倍スコアが上昇した。

###	使用例：システムの実行例を明確に提示する
基本的な操作は`Instructions`の項目を参照してください。

わかりにくかった`moulinette`の操作方法を説明します。\
```bash
cd moulinette_pkg
./moulinette-ubuntu evaluate_student_search_results ../data/output/search_results/dataset_docs_public.json ../data/datasets/AnsweredQuestions/dataset_docs_public.json --threshold 0.8
```

### Resources

AI
- 制約付きデコーディングにおけるLogits操作のアルゴリズム設計とデバッグの壁打ち。
- `flake8` および `mypy` のエラーログ解析と `pyproject.toml` の最適化。
- DocstringおよびREADMEの英訳、構成支援。

`python uv`公式ドキュメント、公式AI\
`langchain`公式ドキュメント、公式AI\
`llama-cpp-python`公式ドキュメント、公式AI
