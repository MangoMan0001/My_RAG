#!/usr/bin/env python3
from pydantic import BaseModel
from .core.indexer import BM25Indexer
from .core.searcher import BM25Searcher, BM25DatasetSearcher
from .core.answer import LLMAnswer, LLMDatasetsAnswer
from .core.evaluater import BM25Evaluater
import sys


class IndexArgs(BaseModel):
    max_chunk_size: int
    index_dir: str


class SearchArgs(BaseModel):
    query: str
    k: int
    index_dir: str
    question_id: str


class SearchDatasetArgs(BaseModel):
    dataset_path: str
    k: int
    save_directory: str
    index_dir: str


class AnswerArgs(BaseModel):
    query: str
    k: int


class AnswerDatasetArgs(BaseModel):
    student_search_results_path: str
    save_directory: str


class EvaluaterArgs(BaseModel):
    student_answer_path: str
    dataset_path: str
    k: int
    max_context_length: int


class RAGCLI:
    """RAGシステムを操作するためのコマンドラインインターフェースですわ。"""

    def index(self,
              max_chunk_size: int = 2000,
              index_dir: str = "data/processed") -> None:
        """
        リポジトリのファイルを読み込み、インデックスを作成しますの。
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
        """
        単一の質問に対して検索を行いますの。
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
                       k: int = 5,
                       save_directory: str = "data/output/search_results",
                       index_dir: str = "data/processed") -> None:
        """
        データセットの複数の質問に対して一括で検索を行いますの。
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
               k: int = 5,
               index_dir: str = "data/processed") -> None:
        """
        単一の質問に対して、検索したコンテキストを用いて回答を生成しますの。
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
                       student_search_results_path: str,
                       save_directory: str = "data/output") -> None:
        """
        検索結果ファイルをもとに、データセット全体の回答を生成しますの。
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
                 student_answer_path: str,
                 dataset_path: str,
                 k: int,
                 max_context_length: int = 2000) -> None:
        """
        検索結果をグラウンドトゥルース（正解データ）と比較して評価しますの。
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
