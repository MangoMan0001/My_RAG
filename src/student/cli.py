#!/usr/bin/env python3
from .core.indexer import BM25Indexer


class RAGCLI:
    """RAGシステムを操作するためのコマンドラインインターフェースですわ。"""

    def index(self,
              max_chunk_size: int = 2000,
              index_dir: str = "data/processed") -> None:
        """
        リポジトリのファイルを読み込み、インデックスを作成しますの。
        """
        print(f"Starting to create an index. chunk_size: {max_chunk_size}")
        indexer = BM25Indexer()
        indexer.indexer(max_chunk_size,
                        index_dir)
        print(f"✅ Ingestion complete! Indices saved under {index_dir}")

    def search(self,
               query: str,
               k: int = 5) -> None:
        """
        単一の質問に対して検索を行いますの。
        """
        print(f"質問: '{query}' に対して上位 {k} 件を検索しますわ...")
        # TODO: rag_core.searcher から検索処理を呼び出す

    def search_dataset(self,
                       dataset_path: str,
                       k: int = 5,
                       save_directory: str = "data/output") -> None:
        """
        データセットの複数の質問に対して一括で検索を行いますの。
        """
        print(f"データセット '{dataset_path}' の検索を開始しますわ...")
        # TODO: バッチ検索処理の実装

    def answer(self,
               query: str,
               k: int = 5) -> None:
        """
        単一の質問に対して、検索したコンテキストを用いて回答を生成しますの。
        """
        pass  # TODO: 回答生成処理の実装

    def answer_dataset(self,
                       student_search_results_path: str,
                       save_directory: str = "data/output") -> None:
        """
        検索結果ファイルをもとに、データセット全体の回答を生成しますの。
        """
        pass  # TODO: バッチ回答生成処理の実装

    def evaluate(self,
                 student_answer_path: str,
                 dataset_path: str, k: int = 5,
                 max_context_length: int = 2000) -> None:
        """
        検索結果をグラウンドトゥルース（正解データ）と比較して評価しますの。
        """
        pass  # TODO: 評価処理の実装
