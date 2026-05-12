# ※事前に `uv add langchain langchain-community` が必要ですわ！
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bm25s
import Stemmer
import json
import os


class BM25Indexer:

    def __init__(self, data_dir: str = "./data/raw/vllm-0.10.1"):  # ここですべてのファイルを読み込めるようにする必要あり
        # 1. Prepare loadder
        self._md_loader = DirectoryLoader(
            path=data_dir,
            glob="**/*.md",
            loader_cls=TextLoader
        )

        self._py_loader = DirectoryLoader(
            path=data_dir,
            glob="**/*.py",
            loader_cls=TextLoader
        )

    def indexer(self,
                max_chunk_size: int,
                index_dir: str) -> None:
        # === Creat Chunk ===
        print('=' * 20)

        docs = self._md_loader.load() + self._py_loader.load()
        print(f"📄 Files: {len(docs)}")

        # 2. Prepare splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=0,
            add_start_index=True
        )

        # 3. splitt！
        chunks = text_splitter.split_documents(docs)
        print(f"✂️ Chunks: {len(chunks)}")

        # 4. Save Chunks
        os.makedirs(index_dir + '/chunks', exist_ok=True)
        dict_chunks = [chunk.model_dump() for chunk in chunks]
        with open(index_dir + '/chunks/corpus.json',
                  mode='w', encoding="utf-8") as f:
            json.dump(dict_chunks, f, ensure_ascii=False, indent=4)

        # === Creat Index ===

        # Setting Stemmer
        stemmer = Stemmer.Stemmer('english')

        # 1. Extract only the text from chunk
        corpus = [chunk.page_content for chunk in chunks]

        # 2. encode corpus
        # stopwords exclude=a, is, the...
        corpus_tokens = bm25s.tokenize(corpus, stopwords='en', stemmer=stemmer)

        # 3. Generate index
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        print(f"🧷 index: {len(retriever.vocab_dict.keys())}")
        print('=' * 20)

        # 4. Save the index
        os.makedirs(index_dir + '/bm25_index', exist_ok=True)
        retriever.save(index_dir + '/bm25_index')


if __name__ == "__main__":
    pass
