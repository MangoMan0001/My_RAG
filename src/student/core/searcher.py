"""Module providing document search capabilities using the BM25 algorithm."""

import bm25s
import json
from .models import (MinimalSource,
                     MinimalSearchResults,
                     StudentSearchResults,
                     UnansweredQuestion,
                     RagDataset)
import os
import Stemmer


class BM25Searcher:
    """Base class for managing BM25 search operations."""

    def __init__(self,
                 index_dir: str) -> None:
        """Initialize the BM25Searcher instance.

        Args:
            index_dir (str): Directory path containing the saved BM25 index and corpus files.
        """
        # Retrieve the index and chunks required for searching
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
               question_id: str = '') -> None:
        """Execute a BM25 search for the given query and store the results.

        Args:
            query (str): The query string to search for.
            k (int): Maximum number of search result chunks to retrieve.
            question_id (str, optional): Unique identifier for the question. Defaults to ''.
        """
        # Set up the Stemmer
        stemmer = Stemmer.Stemmer('english')

        # 1. Tokenize the query
        query_tokens = bm25s.tokenize([query], stopwords='en', stemmer=stemmer)

        # 2. Execute the search
        # results: retrieved document IDs, scores: their relevance scores
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
        question = UnansweredQuestion(question=query, question_id=question_id)
        self._minimal_search = MinimalSearchResults(question_id=question.question_id,
                                                    question_str=question.question,
                                                    retrieved_sources=minimal_list)
        self._minimal_search_list.append(self._minimal_search)
        self._student_search = StudentSearchResults(search_results=self._minimal_search_list,
                                                    k=k)

    def terminal_output(self) -> None:
        """Output the current search results to the terminal in JSON format."""
        data = self._student_search.model_dump()
        print(json.dumps(data, indent=4, ensure_ascii=False))

    @property
    def minimal_serch(self) -> MinimalSearchResults:
        """Retrieve the latest single search result.

        Returns:
            MinimalSearchResults: The latest search result object.
        """
        return self._minimal_search

    @property
    def minimal_serch_list(self) -> list[MinimalSearchResults]:
        """Retrieve the accumulated list of search results.

        Returns:
            list[MinimalSearchResults]: List of accumulated search result objects.
        """
        return self._minimal_search_list

    @property
    def student_search(self) -> StudentSearchResults:
        """Retrieve the formatted submission object for the grading system.

        Returns:
            StudentSearchResults: The submission model containing all search results.
        """
        return self._student_search


class BM25DatasetSearcher(BM25Searcher):
    """Handles batch search operations for datasets containing multiple questions."""

    def __init__(self,
                 index_dir: str) -> None:
        """Initialize the BM25DatasetSearcher instance.

        Args:
            index_dir (str): Directory path containing the saved BM25 index and corpus files.
        """
        super().__init__(index_dir=index_dir)

    def data_search(self,
                    dataset_path: str,
                    k: int) -> None:
        """Perform batch searches for all questions within a dataset file.

        Args:
            dataset_path (str): File path to the JSON dataset containing questions.
            k (int): Number of search results to retrieve per question.
        """
        # Get datasets
        self._dataset_path = dataset_path
        with open(dataset_path, 'r', encoding="utf-8") as f:
            raw_data = json.load(f)
        datasets = RagDataset.model_validate(raw_data)

        # Run data search
        for dataset in datasets.rag_questions:
            self.search(query=dataset.question,
                        k=k,
                        question_id=dataset.question_id)

    def output_json(self,
                    save_directory: str) -> None:
        """Save the accumulated batch search results to a JSON file.

        Args:
            save_directory (str): Directory path where the output JSON will be saved.
        """
        os.makedirs(save_directory, exist_ok=True)
        filename = os.path.basename(self._dataset_path)
        save_path = os.path.join(save_directory, filename)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self._student_search.model_dump(), f, indent=4, ensure_ascii=False)

        print(f"Saved student_search_results to {save_path}")


if __name__ == "__main__":
    pass
