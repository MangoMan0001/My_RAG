"""RAGシステムにおけるLLM（大規模言語モデル）を用いた回答生成モジュール."""

import json
import os
from tqdm import tqdm
from .models import (MinimalSource,
                     MinimalSearchResults,
                     MinimalAnswer,
                     StudentSearchResults,
                     StudentSearchResultsAndAnswer)
from llama_cpp import Llama


class LLMAnswer:
    """単一の質問に対し、LLMを用いて回答を生成するベースクラス."""

    def __init__(self, model_path: str = "data/model/Qwen3-0.6B-Q8_0.gguf") -> None:
        """LLMAnswerクラスの初期化.

        Args:
            model_path (str): ダウンロードしたGGUFモデルのファイルパス。
        """
        self._student_answer: StudentSearchResultsAndAnswer
        self._minimal_answer_list: list[MinimalAnswer] = []

        print("🚀 llama.cppエンジンを起動しますわ！")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=6,
            verbose=False
        )

    def _extract_context(self, sources: list[MinimalSource]) -> str:
        """検索結果の情報源リストから、該当するテキスト部分を抽出して結合する.

        Args:
            sources (list[MinimalSource]): 検索で得られた情報源のリスト。

        Returns:
            str: プロンプトに埋め込むための、抽出・結合されたコンテキスト文字列。
        """
        context_parts = []
        for src in sources:
            try:
                with open(src.file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    chunk_text = content[src.first_character_index:src.last_character_index]
                    context_parts.append(f"[{src.file_path}]:\n{chunk_text}")
            except Exception as e:
                print(f"⚠️ ファイルの読み込みに失敗しました: {e}")

        return "\n\n".join(context_parts)

    def _build_prompt(self, raw_context: str, question: str) -> str:
        safe_context = raw_context[:1500]

        system_prompt = (
            "You are a strict, helpful AI assistant answering questions about a codebase.\n"
            "Rule 1: Answer ONLY based on the provided Context. Do NOT hallucinate.\n"
            "Rule 2: Your answer must be self-contained and directly answer the question.\n"
            "Rule 3: You MUST cite the source file for every fact. Format your citation exactly "
            "like this: (Source: [file_path]).\n"
            "Rule 4: Stop generating text immediately after you have answered the question."
            " Do NOT add any extra explanations or code blocks."
            "Example: The server is configured using the API key (Source: data/raw/.../openai.py)."
        )
        return f"{system_prompt}\n\nContext:\n{safe_context}\n\nQuestion: {question}\nAnswer:"

    def answer(self,
               search_result: MinimalSearchResults,
               k: int,
               max_new_tokens: int = 100) -> None:
        """単一の検索結果オブジェクトを受け取り、LLMを用いた回答生成を実行する.

        Args:
            search_result (MinimalSearchResults): 回答の元となる検索結果と質問を含むオブジェクト。
            k (int): 検索フェーズで指定されたkの値。
            max_new_tokens (int, optional): LLMが新しく生成する最大トークン数。デフォルトは100。
        """
        # 1. コンテキスト（検索結果）の復元
        raw_context = self._extract_context(search_result.retrieved_sources)

        prompt = self._build_prompt(raw_context, search_result.question_str)

        output = self.llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,
            stop='\n',
            echo=False
        )

        # 戻り値からテキストだけを抽出します
        answer_text = output["choices"][0]["text"].strip()  # type: ignore

        self._minimal_answer_list.append(MinimalAnswer(
            question_id=search_result.question_id,
            question_str=search_result.question_str,
            retrieved_sources=search_result.retrieved_sources,
            answer=answer_text
        ))

        self._student_answer = StudentSearchResultsAndAnswer(
            search_results=self._minimal_answer_list,
            k=k
        )

    def terminal_output(self) -> None:
        """現在保持している回答結果をJSON文字列としてターミナルに出力する."""
        data = self._student_answer.model_dump()
        print(json.dumps(data, indent=4, ensure_ascii=False))


class LLMDatasetsAnswer(LLMAnswer):
    """データセット（複数質問）に対する一括回答生成処理を担当するクラス.

    検索済みのデータセット（JSONファイル）を読み込み、
    含まれるすべての質問に対して連続で回答生成を実行してファイルに保存します。
    """

    def __init__(self, model_path: str = "data/model/Qwen3-0.6B-Q8_0.gguf") -> None:
        """LLMDatasetsAnswerクラスの初期化.

        親クラスの初期化処理を呼び出し、モデルのロードを行います。

        Args:
            model_path (str): ダウンロードしたGGUFモデルのファイルパス。
        """
        super().__init__(model_path=model_path)

    def data_answer(self, student_search_results_path: str) -> None:
        """検索結果データセットを読み込み、すべての質問に対して一括で回答を生成する.

        Args:
            student_search_results_path (str): 検索フェーズで出力されたJSONファイルのパス。
        """
        # get datasets
        self._dataset_path = student_search_results_path
        with open(self._dataset_path, 'r', encoding="utf-8") as f:
            raw_data = json.load(f)

        datasets = StudentSearchResults.model_validate(raw_data)

        for dataset in tqdm(datasets.search_results, desc='Current status: Generating'):
            self.answer(search_result=dataset, k=datasets.k)

    def output_json(self, save_directory: str) -> None:
        """蓄積された一括回答結果をJSONファイルとして保存する.

        Args:
            save_directory (str): 成果物となるJSONファイルを保存するディレクトリのパス。
        """
        os.makedirs(save_directory, exist_ok=True)
        filename = os.path.basename(self._dataset_path)
        save_path = os.path.join(save_directory, filename)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self._student_answer.model_dump(), f, indent=4, ensure_ascii=False)
        print(f"Saved student_search_results to {save_path}")


if __name__ == "__main__":
    pass
