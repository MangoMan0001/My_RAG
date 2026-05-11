import bm25s

class BM25Sercher:
    """BM25のインデックスを管理し、検索を行うクラス"""

    def __init__(self, index_dir: str = "data/processed/bm25_index"):
        pass

    def search(self, query: str, k: int) -> None:
        print("✨ 辞書が完成いたしましたわ！検索テストを開始します！")

        # 4. 試し斬り！クエリ（質問）の準備
        query = "What is api_server?"
        query_tokens = bm25s.tokenize([query])

        # 5. 検索実行！（k=5 で上位5件を取得）
        # results は見つかったテキスト、scores はその関連度スコアですの
        results, scores = retriever.retrieve(query_tokens, k=5)

        # 5. ディレクトリに保存

        print("\n👑 【検索結果 トップ5】👇\n")
        for i in range(5):
            print(f"--- 🏅 第{i+1}位 (スコア: {scores[0][i]:.2f}) ---")
            print(f"【メタデータ（出処）】: {chunks[results[0][i]].metadata}")
            print(chunks[results[0][i]].page_content[:200])  # 長すぎるので最初の200文字だけ表示


if __name__ == "__main__":
    pass
