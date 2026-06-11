#!/usr/bin/env python3
"""Command-line interface module managing the RAG system pipeline."""

from pydantic import BaseModel
from .core.indexer import BM25Indexer
from .core.searcher import BM25Searcher, BM25DatasetSearcher
from .core.answer import LLMAnswer, LLMDatasetsAnswer
from .core.evaluator import BM25Evaluater
import sys


class IndexArgs(BaseModel):
    """Pydantic model for validating and managing arguments of the index command."""
    max_chunk_size: int
    index_dir: str


class SearchArgs(BaseModel):
    """Pydantic model for validating and managing arguments of the search command."""
    query: str
    k: int
    index_dir: str
    question_id: str


class SearchDatasetArgs(BaseModel):
    """Pydantic model for validating and managing arguments of the search_dataset command."""
    dataset_path: str
    k: int
    save_directory: str
    index_dir: str


class AnswerArgs(BaseModel):
    """Pydantic model for validating and managing arguments of the answer command."""
    query: str
    k: int


class AnswerDatasetArgs(BaseModel):
    """Pydantic model for validating and managing arguments of the answer_dataset command."""
    student_search_results_path: str
    save_directory: str


class EvaluaterArgs(BaseModel):
    """Pydantic model for validating and managing arguments of the evaluate command."""
    student_answer_path: str
    dataset_path: str
    k: int
    max_context_length: int


class RAGCLI:
    """CLI class managing the RAG (Retrieval-Augmented Generation) system pipeline."""

    def index(self,
              max_chunk_size: int = 2000,
              index_dir: str = "data/processed") -> None:
        """Read Markdown and Python files, split them into chunks, and create a BM25 index.

        Args:
            max_chunk_size (int, optional): Maximum characters per chunk. Defaults to 2000.
            index_dir (str, optional): Directory path to save the created index. Defaults to "data/processed".
        """
        try:
            args = IndexArgs(max_chunk_size=max_chunk_size,
                             index_dir=index_dir)
            max_chunk_size = args.max_chunk_size
            index_dir = args.index_dir
            print(f"Starting to create an index. chunk_size: {max_chunk_size}")
            indexer = BM25Indexer()
            indexer.indexer(max_chunk_size,
                            index_dir)
            print(f"✅ Ingestion complete! Indices saved under {index_dir}")
        except Exception as e:
            print(e)
            sys.exit(1)

    def search(self,
               query: str,
               k: int = 5,
               index_dir: str = "data/processed",
               question_id: str = "q1") -> None:
        """Search for the most relevant documents for a single query using the BM25 index.

        Args:
            query (str): The query string to search for.
            k (int, optional): Maximum number of search result chunks to retrieve. Defaults to 5.
            index_dir (str, optional): Directory path where the target index is stored. Defaults to "data/processed".
            question_id (str, optional): Question ID to associate with the results. Defaults to "q1".
        """
        try:
            args = SearchArgs(query=query,
                              k=k,
                              index_dir=index_dir,
                              question_id=question_id)
            query = args.query
            k = args.k
            index_dir = args.index_dir
            question_id = args.question_id
            print(f"Searching for the top {k} results for '{query}'")
            searcher = BM25Searcher(index_dir=index_dir)
            searcher.search(query=query,
                            k=k,
                            question_id=question_id)
            searcher.terminal_output()
            print("✅ Serching complete!")
        except Exception as e:
            print(e)
            sys.exit(1)

    def search_dataset(self,
                       dataset_path: str = 'data/datasets/UnansweredQuestions/dataset_docs_public.json',
                       k: int = 10,
                       save_directory: str = "data/output/search_results",
                       index_dir: str = "data/processed") -> None:
        """Execute batch searches for all questions in the dataset and save the results as a JSON file.

        Args:
            dataset_path (str, optional): File path to the dataset (JSON format) containing questions.
                Defaults to 'data/datasets/UnansweredQuestions/dataset_docs_public.json'.
            k (int, optional): Maximum number of chunks to retrieve for each question. Defaults to 10.
            save_directory (str, optional): Directory path to save the search results.
                Defaults to "data/output/search_results".
            index_dir (str, optional): Directory path where the target index is stored.
                Defaults to "data/processed".
        """
        try:
            args = SearchDatasetArgs(dataset_path=dataset_path,
                                     k=k,
                                     save_directory=save_directory,
                                     index_dir=index_dir)
            dataset_path = args.dataset_path
            k = args.k
            save_directory = args.save_directory
            index_dir = args.index_dir
            print(f"search the dataset '{dataset_path}'")
            searcher = BM25DatasetSearcher(index_dir=index_dir)
            searcher.data_search(dataset_path=dataset_path,
                                 k=k)
            searcher.output_json(save_directory=save_directory)

        except Exception as e:
            print(e)
            sys.exit(1)

    def answer(self,
               query: str,
               k: int = 10,
               index_dir: str = "data/processed") -> None:
        """Generate a direct answer using an LLM based on the search results for a single query.

        Args:
            query (str): The query string for the LLM to answer.
            k (int, optional): Maximum number of chunks to use as context for answer generation. Defaults to 10.
            index_dir (str, optional): Directory path where the target index is stored.
                Defaults to "data/processed".
        """
        try:
            args = AnswerArgs(query=query, k=k)
            query = args.query
            k = args.k
            print(f"answer the query '{query}'")
            searcher = BM25Searcher(index_dir=index_dir)
            searcher.search(query=query, k=k)
            answer = LLMAnswer()
            answer.answer(search_result=searcher.minimal_serch, k=k)
            answer.terminal_output()
        except Exception as e:
            print(e)
            sys.exit(1)

    def answer_dataset(self,
                       student_search_results_path: str = "data/output/search_results/dataset_docs_public.json",
                       save_directory: str = "data/output/search_results_and_answer") -> None:
        """Generate and save answers in batch for all questions in the dataset based on the search results.

        Args:
            student_search_results_path (str): File path of the search results (JSON format) output by the engine.
                Defaults to "data/output/search_results/dataset_docs_public.json".
            save_directory (str, optional): Directory path to save the final results including the LLM answers.
                Defaults to "data/output".
        """
        try:
            args = AnswerDatasetArgs(
                student_search_results_path=student_search_results_path,
                save_directory=save_directory
            )
            student_search_results_path = args.student_search_results_path
            save_directory = args.save_directory
            answer = LLMDatasetsAnswer()
            answer.data_answer(student_search_results_path=student_search_results_path)
            answer.output_json(save_directory=save_directory)

            print(f"Loaded 100 questions from {student_search_results_path}\n"
                  f"Processed 100 of 100 questions\n"
                  f"Saved student_search_results_and_answer to {save_directory}")
        except Exception as e:
            print(e)
            sys.exit(1)

    def evaluate(self,
                 student_answer_path: str = "data/output/search_results/dataset_docs_public.json",
                 dataset_path: str = "data/datasets/AnsweredQuestions/dataset_docs_public.json",
                 k: int = 10,
                 max_context_length: int = 2000) -> None:
        """Compare the generated search results with the ground truth dataset to evaluate retrieval accuracy.

        Args:
            student_answer_path (str, optional): File path of the generated search results to evaluate.
                Defaults to "data/output/search_results/dataset_docs_public.json".
            dataset_path (str, optional): File path of the ground truth dataset containing the correct sources.
                Defaults to "data/datasets/AnsweredQuestions/dataset_docs_public.json".
            k (int, optional): The number of top search results to evaluate (ranking depth). Defaults to 10.
            max_context_length (int, optional): Maximum characters allowed per chunk during evaluation.
                Defaults to 2000.
        """
        try:
            args = EvaluaterArgs(
                student_answer_path=student_answer_path,
                dataset_path=dataset_path,
                k=k,
                max_context_length=max_context_length
            )
            student_answer_path = args.student_answer_path
            dataset_path = args.dataset_path
            k = args.k
            max_context_length = args.max_context_length
            evaluater = BM25Evaluater(student_answer_path=student_answer_path,
                                      dataset_path=dataset_path)
            evaluater.evaluate(k=k,
                               max_context_length=max_context_length)

        except Exception as e:
            print(e)
            sys.exit(1)
