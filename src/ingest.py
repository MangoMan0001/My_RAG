# ※事前に `uv add langchain langchain-community` が必要ですわ！
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main() -> None:
    print("🧹 vLLMの森から、PythonとMarkdownのファイルだけを集めていますわ...")

    # 1. フィルタリングしながら読み込む魔法のローダー
    # （まずはテストとして .md だけ読み込んでみるのがオススメですわ！）
    md_loader = DirectoryLoader(
        path="./vllm-0.10.1",
        glob="**/*.md",
        loader_cls=TextLoader
    )

    py_loader = DirectoryLoader(
        path="./vllm-0.10.1",
        glob="**/*.md",
        loader_cls=TextLoader
    )

    docs = md_loader.load() + py_loader.load()
    print(f"📄 見つかったファイル数: {len(docs)} 件")

    # 2. ハサミ（スプリッター）の準備
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=0
    )

    # 3. 切り刻む！
    chunks = text_splitter.split_documents(docs)
    print(f"✂️ 切り刻んだチャンクの総数: {len(chunks)} 個")

    # 4. 最初の1個目をターミナルにお披露目！
    if chunks:
        print("\n👑 本日の記念すべき最初のチャンクですわ！👇\n")
        print("="*50)
        print(f"【メタデータ（出処）】: {chunks[0].metadata}")
        print("-" * 50)
        print(chunks[0].page_content)
        print("="*50)


if __name__ == "__main__":
    main()
