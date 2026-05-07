# ※事前に `uv add langchain langchain-community` が必要ですわ！
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    JSONLoader,
    UnstructuredMarkdownLoader, # Markdown解析に優れたローダー
    UnstructuredHTMLLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    PythonCodeTextSplitter,
    MarkdownTextSplitter
)
import bm25s
import sys
import os

# 1. 拡張子とローダーのマッピング定義
LOADER_MAPPING = {
    '.json': JSONLoader,
    '.jsonl': JSONLoader,
    '.md': UnstructuredMarkdownLoader,
    '.html': UnstructuredHTMLLoader, # HTML専用ローダー
    '.py': TextLoader,
    '.js': TextLoader,
    '.cpp': TextLoader, '.h': TextLoader, '.hpp': TextLoader, '.cu': TextLoader, '.cuh': TextLoader,
    '.cmake': TextLoader, '.in': TextLoader, '.inl': TextLoader,
    '.yaml': TextLoader, '.toml': TextLoader, '.jinja': TextLoader, '.tpl': TextLoader,
    '.sh': TextLoader, '.txt': TextLoader, '.cpu': TextLoader, '.neuron': TextLoader, '.rocm': TextLoader, '.xpu': TextLoader, '.tpu': TextLoader, '.ppc64le': TextLoader, '.s390x': TextLoader,
}
# 1. 除外リストの定義 (globパターン)
EXCLUDE_PATTERNS = [
    "**/*.png", "**/*.jpg", "**/*.ico",         # 画像
    "**/.env", "**/.gitignore",                 # 設定・機密情報
    "**/.dockerignore", "**/.helmignore",
    "**/.DS_Store", "**/__pycache__/*",         # OS・キャッシュ
    "**/.pytest_cache/*", "**/*.log", "**/*.tmp",
    "**/node_modules/*", "**/venv/*",           # 依存関係
    "**/*.pyc", "**/*.o"
]

# 拡張子とローダーの決定版マッピング

# スプリッターの振り分けも強化
def get_splitter(ext):
    if ext == '.py': return PythonCodeTextSplitter(...)
    if ext == '.md': return MarkdownTextSplitter(...)
    if ext in ['.cpp', '.h', '.hpp', '.cu', '.cuh']: return RecursiveCharacterTextSplitter(...) # C++系
    return RecursiveCharacterTextSplitter(...) # その他デフォルト

# def main() -> None:
#     print("🧹 vLLMの森から、PythonとMarkdownのファイルだけを集めていますわ...")

#     # 1. フィルタリングしながら読み込む魔法のローダー
#     # （まずはテストとして .md だけ読み込んでみるのがオススメですわ！）
#     md_loader = DirectoryLoader(
#         path="./vllm-0.10.1",
#         glob="**/*.md",
#         loader_cls=TextLoader
#     )

# 構造化データには専用ローダーを割り当てる


def load_and_process_smart(directory_path: str) --> None:
    all_docs = []

    # 2. 拡張子ごとにループして読み込む（細かい制御が可能）
    for ext, loader_cls in LOADER_MAPPING.items():
        # 特定の拡張子だけをロードするディレクトリローダーを作成
        loader = DirectoryLoader(
            directory_path,
            glob=f"**/*{ext}", # その拡張子のみ対象
            loader_cls=loader_cls,
            # JSONLoaderの場合は特別な設定が必要なケースがあるため注意
        )
        try:
            docs = loader.load()
            for doc in docs:
                doc.metadata['extension'] = ext
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error loading {ext}: {e}")

    # 3. 分類して専用スプリッターへ渡す
    # (先ほどのコードと同様の処理)
    # ...


def load_and_split_documents(directory_path: str):
    # 2. DirectoryLoaderで除外パターンを適用してロード
    loader = DirectoryLoader(
        directory_path,
        glob="**/*",
        exclude=EXCLUDE_PATTERNS,
        loader_cls=TextLoader,
        recursive=True
    )
    all_docs = loader.load()

    # 3. 拡張子ベースでスプリッターを使い分ける分類処理
    python_docs = []
    md_docs = []
    default_docs = []

    for doc in all_docs:
        # メタデータのパスから拡張子を取得
        source = doc.metadata.get('source', '')
        ext = os.path.splitext(source)[-1].lower()

        # メタデータに拡張子情報を付与 (運用上のベストプラクティス)
        doc.metadata['extension'] = ext

        if ext == '.py':
            python_docs.append(doc)
        elif ext == '.md':
            md_docs.append(doc)
        else:
            default_docs.append(doc)

    # 4. 各専用スプリッターで分割
    # 言語専用スプリッターは構文解析(例: Pythonなら関数/クラス定義)を考慮
    final_docs = (
        PythonCodeTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(python_docs) +
        MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(md_docs) +
        RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(default_docs)
    )

    return final_docs

# 使用例
# docs = load_and_split_documents("./my_project")


    print(f"📄 見つかったファイル数: {len(md_docs + py_docs)} 件")

    # python用とmd用を抜き出し、残りをデフォルト（recusive）へ
    python_docs = []
    md_docs = []
    default_docs = []

    for doc in all_docs:
        source = doc.metadata.get('source', '')
        if source.endswith('.py'):
            python_docs.append(doc)
        elif source.endswith('.md'):
            md_docs.append(doc)
        else:
            default_docs.append(doc)

    # それぞれスプリッターにかける
    final_docs = (
        PythonCodeTextSplitter().split_documents(python_docs) +
        MarkdownTextSplitter().split_documents(md_docs) +
        RecursiveCharacterTextSplitter().split_documents(default_docs)
    )




    # 2. ハサミ（スプリッター）の準備
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=0
    )

    # 3. 切り刻む！
    chunks = text_splitter.split_documents()
    print(f"✂️ 切り刻んだチャンクの総数: {len(chunks)} 個")

    # 4. 最初の1個目をターミナルにお披露目！
    print(chunks)
    sys.exit(1)
    if chunks:
        print("\n👑 本日の記念すべき最初のチャンクですわ！👇\n")
        print("="*50)
        print(f"【メタデータ（出処）】: {chunks[0].metadata}")
        print("-" * 50)
        print(chunks[0].page_content)
        print("="*50)


if __name__ == "__main__":
    main()
