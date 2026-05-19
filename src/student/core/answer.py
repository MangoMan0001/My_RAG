import torch
import json
from tqdm import tqdm
import os
from .models import (MinimalSource,
                     MinimalSearchResults,
                     MinimalAnswer,
                     StudentSearchResults,
                     StudentSearchResultsAndAnswer)
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMAnswer:
    """モデルを読み込み単一のプロンプトに回答する"""

    def __init__(self, model_id: str = "Qwen/Qwen3-0.6B") -> None:

        self._student_answer: StudentSearchResultsAndAnswer
        self._minimal_answer_list: list[MinimalAnswer] = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",  # メモリ配置を自動化
            torch_dtype="auto",  # メモリ効率を自動化
            trust_remote_code=True  # Qwenのリモートコードの使用を許可
        )
        # 学習ではなく本番環境に
        self.model.eval()  # type: ignore

    def _extract_context(self, sources: list[MinimalSource]) -> str:
        """検索結果のインデックス情報から、実際のテキストを切り出して結合しますわ。"""
        context_parts = []
        for src in sources:
            try:
                with open(src.file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    chunk_text = content[src.first_character_index:src.last_character_index]
                    context_parts.append(f"[{src.file_path}]:\n{chunk_text}")
            except Exception as e:
                print(f"⚠️ ファイルの読み込みに失敗しましたわ: {e}")

        return "\n\n".join(context_parts)

    def _build_prompt(self,
                      raw_context: str,
                      question: str) -> str:
        "promptを作成"
        context_tokens = self.tokenizer.encode(raw_context)

        if len(context_tokens) > 2000:
            safe_context = self.tokenizer.decode(context_tokens[:1500], skip_special_tokens=True)
        else:
            safe_context = raw_context

        system_prompt = (
            "You are a strict, helpful AI assistant answering questions about a codebase.\n"
            "Rule 1: Answer ONLY based on the provided Context. Do NOT hallucinate.\n"
            "Rule 2: Your answer must be self-contained and directly answer the question.\n"
            "Rule 3: You MUST cite the source file for every fact. Format your citation exactly like this: (Source: [file_path]).\n"
            "Rule 4: Stop generating text immediately after you have answered the question. Do NOT add any extra explanations or code blocks."
            "Example: The server is configured using the API key (Source: data/raw/.../openai.py)."
        )
        return f"{system_prompt}\n\nContext:\n{safe_context}\n\nQuestion: {question}\nAnswer:"

    def answer(self,
               search_result: MinimalSearchResults,
               k: int,
               max_new_tokens: int = 512) -> None:
        """1つの質問に対して回答を生成しますわ。"""

        # 1. コンテキスト（検索結果）の復元
        raw_context = self._extract_context(search_result.retrieved_sources)

        prompt = self._build_prompt(raw_context, search_result.question)

        # 3. 推論の実行（ここにはもう長すぎるテキストは入りませんわ！）
        # pytorch型のtensorに入れる。modelに送る.
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():  # 学習を避けるためにメモリ保存を許可しない
            outputs = self.model.generate(  # type: ignore
                **inputs,
                max_new_tokens=max_new_tokens,  # 最大出力量。無限ループ避け
                do_sample=False,  # 常に最高スコアのみ選択
                # repetition_penalty=1.15  # 👈 同じ言葉の繰り返しを禁止する。が、めちゃんこメモリ食うので動かず..
                no_repeat_ngram_size=3      # ✅ 代わりにこの魔法を追加
            )

        # 4. 回答の抽出
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        answer_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        assert isinstance(answer_text, str)

        stop_words = ["\nAnswer:", "\nQuestion:", "\n\n"]
        for stop_word in stop_words:  # 回答の水増しをカット
            if stop_word in answer_text:
                answer_text = answer_text.split(stop_word)[0].strip()

        # 5. 結果を返す
        self._minimal_answer_list.append(MinimalAnswer(
            question_id=search_result.question_id,
            question=search_result.question,
            retrieved_sources=search_result.retrieved_sources,
            answer=answer_text
        ))

        self._student_answer = StudentSearchResultsAndAnswer(
            search_results=self._minimal_answer_list,
            k=k
        )

    def terminal_output(self) -> None:
        "Output results terminl."
        data = self._student_answer.model_dump()
        print(json.dumps(data, indent=4, ensure_ascii=False))


class LLMDatasetsAnswer(LLMAnswer):
    """モデルを読み込みプロンプトに回答する"""

    def __init__(self, model_id: str = "Qwen/Qwen3-0.6B") -> None:
        super().__init__(model_id=model_id)

    def data_answer(self, student_search_results_path: str) -> None:
        # get datasets
        self._dataset_path = student_search_results_path
        with open(self._dataset_path, 'r', encoding="utf-8") as f:
            raw_data = json.load(f)
        datasets = StudentSearchResults.model_validate(raw_data)
        for dataset in tqdm(datasets.search_results, desc='Current status: Generating'):
            self.answer(search_result=dataset, k=datasets.k)
            break

    def output_json(self,
                    save_directory: str) -> None:
        "output results json"
        os.makedirs(save_directory, exist_ok=True)
        filename = os.path.basename(self._dataset_path)
        save_path = os.path.join(save_directory, filename)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self._student_answer.model_dump(), f, indent=4, ensure_ascii=False)
        print(f"Saved student_search_results to {save_path}")


if __name__ == "__main__":
    pass
