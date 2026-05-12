import bm25s
import json
from .models import MinimalSource, MinimalSearchResults, StudentSearchResults
from typing import Any
import sys
import os


class BM25Searcher:
    """BM25のインデックスを管理し、検索を行うクラス"""

    def __init__(self,
                 index_dir: str) -> None:
        # 検索に必要なindexとchunksを取得
        self._minimal_search: MinimalSearchResults
        self._minimal_search_list: list[MinimalSearchResults] = []
        self._student_search: StudentSearchResults
        self._retriever = bm25s.BM25.load(index_dir + "/bm25_index", mmap=True)
        with open(index_dir + "/chunks/corpus.json",
                  mode='r',
                  encoding="utf-8") as f:
            self._chunks = json.load(f)

    def search(self,
               query: str,
               k: int,
               question_id: str) -> None:
        # 1. クエリをトークン化
        query_tokens = bm25s.tokenize([query])

        # 2. 検索実行
        # results は見つかったドキュメントID、scores はその関連度スコア
        results, scores = self._retriever.retrieve(query_tokens, k=k)
        minimal_list = []
        for i in range(k):
            chunk = self._chunks[results[0][i]]
            meta = chunk['metadata']
            last_i = meta['start_index'] + len(chunk['page_content'])
            minimal = MinimalSource(file_path=meta['source'],
                                    first_character_index=meta['start_index'],
                                    last_character_index=last_i)
            minimal_list.append(minimal)
        self._minimal_search = MinimalSearchResults(question_id=question_id,
                                                    question=query,
                                                    retrieved_sources=minimal_list)
        self._minimal_search_list.append(self._minimal_search)
        self._student_search = StudentSearchResults(search_results=self._minimal_search_list,
                                                    k=k)

    def terminal_output(self) -> None:
        "Output results terminl."
        data = self._student_search.model_dump()
        print(json.dumps(data, indent=4, ensure_ascii=False))

    def minimal_serch(self) -> MinimalSearchResults:
        return self._minimal_search

    def minimal_serch_list(self) -> list[MinimalSearchResults]:
        return self._minimal_search_list

    def student_search(self) -> StudentSearchResults:
        return self._student_search


class BM25DatasetSearcher(BM25Searcher):
    """BM25のインデックスを管理し、検索を行うクラス"""

    def __init__(self,
                 index_dir: str) -> None:
        super().__init__(index_dir=index_dir)

    def data_search(self,
                    dataset_path: str,
                    k: int) -> None:
        # get datasets
        with open(dataset_path, 'r', encoding="utf-8") as f:
            datasets = list(json.load(f).values())[0]
        self._dataset_path = dataset_path

        # Run data search
        for dataset in datasets:
            q_id, query = dataset.values()
            self.search(query=query,
                        k=k,
                        question_id=q_id)

    def output_json(self,
                    save_directory: str) -> None:
        "output results json"
        os.makedirs(save_directory, exist_ok=True)
        filename = os.path.basename(self._dataset_path)
        save_path = os.path.join(save_directory, filename)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self._student_search.model_dump(), f, indent=4, ensure_ascii=False)
        print(f"Saved student_search_results to {save_path}")


if __name__ == "__main__":
    pass
