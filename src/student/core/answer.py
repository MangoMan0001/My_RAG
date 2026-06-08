"""Module for generating answers using Large Language Models (LLM) in the RAG system."""

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
    """Base class for generating answers to single queries using an LLM."""

    def __init__(self, model_path: str = "data/model/Qwen3-0.6B-Q8_0.gguf") -> None:
        """Initialize the LLMAnswer instance and load the GGUF model.

        Args:
            model_path (str): File path to the downloaded GGUF model.
                Defaults to "data/model/Qwen3-0.6B-Q8_0.gguf".
        """
        self._student_answer: StudentSearchResultsAndAnswer
        self._minimal_answer_list: list[MinimalAnswer] = []

        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=6,
            verbose=False
        )

    def _extract_context(self, sources: list[MinimalSource]) -> str:
        """Extract and concatenate text from the retrieved source indices.

        Args:
            sources (list[MinimalSource]): List of retrieved information sources.

        Returns:
            str: Concatenated context string to be embedded in the prompt.
        """
        context_parts = []
        for src in sources:
            try:
                with open(src.file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    chunk_text = content[src.first_character_index:src.last_character_index]
                    context_parts.append(f"[{src.file_path}]:\n{chunk_text}")
            except Exception as e:
                print(f"⚠️ Failed to read the file: {e}")

        return "\n\n".join(context_parts)

    def _build_prompt(self, raw_context: str, question: str) -> str:
        """Construct the prompt containing system instructions, context, and the question.

        Args:
            raw_context (str): Extracted raw context string.
            question (str): User's question text.

        Returns:
            str: The final prompt string formatted for the LLM.
        """
        safe_context = raw_context[:1500]

        system_prompt = (
            "You are a strict, helpful AI assistant answering questions about a codebase.\n"
            "Rule 1: Answer ONLY based on the provided Context. Do NOT hallucinate.\n"
            "Rule 2: Your answer must be self-contained and directly answer the question.\n"
            "Rule 3: You MUST cite the source file for every fact. Format your citation exactly "
            "like this: (Source: [file_path]).\n"
            "Rule 4: Stop generating text immediately after you have answered the question."
            " Do NOT add any extra explanations or code blocks.\n"
            "Example: The server is configured using the API key (Source: data/raw/.../openai.py)."
        )
        return f"{system_prompt}\n\nContext:\n{safe_context}\n\nQuestion: {question}\nAnswer:"

    def answer(self,
               search_result: MinimalSearchResults,
               k: int,
               max_new_tokens: int = 100) -> None:
        """Generate an answer for a single search result using the LLM.

        Args:
            search_result (MinimalSearchResults): Object containing the search results and question.
            k (int): Number of requested results from the search phase.
            max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 100.
        """
        # 1. Restore the context (search results)
        raw_context = self._extract_context(search_result.retrieved_sources)

        prompt = self._build_prompt(raw_context, search_result.question_str)

        output = self.llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,
            stop='\n',
            echo=False
        )

        # Extract only the generated text from the model output
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
        """Output the current search results to the terminal in JSON format."""
        data = self._student_answer.model_dump()
        print(json.dumps(data, indent=4, ensure_ascii=False))


class LLMDatasetsAnswer(LLMAnswer):
    """Handles batch answer generation for datasets containing multiple queries."""

    def __init__(self, model_path: str = "data/model/Qwen3-0.6B-Q8_0.gguf") -> None:
        """Initialize the LLMDatasetsAnswer instance.

        Args:
            model_path (str, optional): File path to the downloaded GGUF model.
                Defaults to "data/model/Qwen3-0.6B-Q8_0.gguf".
        """
        super().__init__(model_path=model_path)

    def data_answer(self, student_search_results_path: str) -> None:
        """Generate answers in batch for a dataset of search results.

        Args:
            student_search_results_path (str): Path to the JSON file containing search results.
        """
        # Get datasets
        self._dataset_path = student_search_results_path
        with open(self._dataset_path, 'r', encoding="utf-8") as f:
            raw_data = json.load(f)

        datasets = StudentSearchResults.model_validate(raw_data)

        for dataset in tqdm(datasets.search_results, desc='Current status: Generating'):
            self.answer(search_result=dataset, k=datasets.k)

    def output_json(self, save_directory: str) -> None:
        """Save the accumulated batch answers to a JSON file.

        Args:
            save_directory (str): Directory path where the output JSON will be saved.
        """
        os.makedirs(save_directory, exist_ok=True)
        filename = os.path.basename(self._dataset_path)
        save_path = os.path.join(save_directory, filename)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self._student_answer.model_dump(), f, indent=4, ensure_ascii=False)
        print(f"Saved student_search_results to {save_path}")


if __name__ == "__main__":
    pass
