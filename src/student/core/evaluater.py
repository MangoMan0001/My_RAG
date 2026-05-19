import json
from tqdm import tqdm
import os
import sys
from .models import (MinimalSource,
                     MinimalSearchResults,
                     MinimalAnswer,
                     AnsweredQuestion,
                     UnansweredQuestion,
                     RagDataset,
                     StudentSearchResults,
                     StudentSearchResultsAndAnswer)
from transformers import AutoTokenizer, AutoModelForCausalLM


class BM25Evaluater:
    """検索結果の整合性を評価する"""

    def __init__(self,
                 student_answer_path: str,
                 dataset_path: str) -> None:
        self._dataset_path = dataset_path
        with open(self._dataset_path, 'r', encoding="utf-8") as f:
            raw_data = json.load(f)
        self._datasets = RagDataset.model_validate(raw_data)

        self._answer_path = student_answer_path
        with open(self._answer_path, 'r', encoding="utf-8") as f:
            raw_data = json.load(f)
        self._answers = StudentSearchResults.model_validate(raw_data)

    def evaluate(self,
                 k: int,
                 max_context_length: int) -> None:
        """評価を実行"""
        scoring_pattern = {1, 3, 5, 10}
        scoring_pattern.add(k)
        for i in sorted(scoring_pattern):
            self._scoring(i, max_context_length)

    def _scoring(self,
                 k: int,
                 max_context_length: int) -> None:
        # 引数kと実際の検索ファイル数の比較
        if self._answers.k < k:
            return
        scores = []
        for dataset, answer in zip(self._datasets.rag_questions,
                                   self._answers.search_results):
            # print()
            # print(dataset.question)
            assert isinstance(dataset, AnsweredQuestion)

            if dataset.question_id != answer.question_id:
                raise ValueError('question is incorrect.')

            source_count = 0
            true_count = 0
            for true_source in dataset.sources:
                source_count += 1
                for i, user_source in enumerate(answer.retrieved_sources):
                    if k < i:
                        break
                    if true_source.file_path != user_source.file_path:
                        # print('ファイルが違う')
                        continue
                    t_start, u_start = true_source.first_character_index, user_source.first_character_index
                    t_end, u_end = true_source.last_character_index, user_source.last_character_index

                    # チャンク制限
                    if max_context_length < u_end - u_start:
                        u_end = u_start + max_context_length
                    overlap_start = max(t_start, u_start)
                    overlap_end = min(t_end, u_end)
                    overlap_length = max(0, overlap_end - overlap_start)

                    correct_length = t_end - t_start
                    if correct_length * 0.05 <= overlap_length:
                        # print("🎯 見事ソースを特定！")
                        true_count += 1
                        break
                    # else:
                        # print('ダメぽよ')
            scores.append(true_count / source_count)
        print(f'Reacall@{k}: {sum(scores) / len(scores)}')


if __name__ == "__main__":
    pass
