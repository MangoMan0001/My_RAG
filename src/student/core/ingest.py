# ※事前に `uv add langchain langchain-community` が必要ですわ！
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bm25s
import Stemmer
import sys
import os

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
        glob="**/*.py",
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

    # if chunks:
    #     print("\n👑 本日の記念すべき最初のチャンクですわ！👇\n")
    #     print("="*50)
    #     print(f"【メタデータ（出処）】: {chunks[0].metadata}")
    #     print("-" * 50)
    #     print(chunks[0].page_content)
    #     print("="*50)

    print("\n🧠 検索エンジン（BM25）の辞書を作りますわ！...")

    # ステミング用ライブラリを使用 run running runsをrunに統一する
    stemmer = Stemmer.Stemmer('english')

    # 1. チャンクの中から「テキスト本体」だけを抽出したリストを作りますの
    corpus = [chunk.page_content for chunk in chunks]

    # 2. テキストをBM25が理解できる「単語のリスト（トークン）」に分解！
    corpus_tokens = bm25s.tokenize(corpus, stopwords='en', stemmer=stemmer)

    # 3. 辞書の作成（インデックス化）！
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    print("✨ 辞書が完成いたしましたわ！検索テストを開始します！")

    # 4. 試し斬り！クエリ（質問）の準備
    query = "What is api_server?"
    query_tokens = bm25s.tokenize([query])

    # 5. 検索実行！（k=5 で上位5件を取得）
    # results は見つかったテキスト、scores はその関連度スコアですの
    results, scores = retriever.retrieve(query_tokens, k=5)

    # 5. ディレクトリに保存
    os.makedirs('data/processed', exist_ok=True)
    retriever.save('data/processed', corpus=corpus)

    print("\n👑 【検索結果 トップ5】👇\n")
    for i in range(5):
        print(f"--- 🏅 第{i+1}位 (スコア: {scores[0][i]:.2f}) ---")
        print(f"【メタデータ（出処）】: {chunks[results[0][i]].metadata}")
        print(chunks[results[0][i]].page_content[:200])  # 長すぎるので最初の200文字だけ表示


if __name__ == "__main__":
    main()
