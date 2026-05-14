from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# (中略)

    def indexer(self, max_chunk_size: int, index_dir: str) -> None:
        print('=' * 20)

        # 1. ドキュメントは「別々に」読み込んだままにしておきますわ
        md_docs = self._md_loader.load()
        py_docs = self._py_loader.load()
        print(f"📄 MD Files: {len(md_docs)}, PY Files: {len(py_docs)}")

        # 2. Markdown専用の賢いスプリッター
        # （# の見出しや段落を優先して綺麗に割ってくれますの）
        md_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=max_chunk_size,
            chunk_overlap=0,
            add_start_index=True
        )

        # 3. Python専用の賢いスプリッター
        # （\nclass や \ndef を優先して、関数やクラスの塊で割ってくれますの！）
        py_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=max_chunk_size,
            chunk_overlap=0,
            add_start_index=True
        )

        # 4. 別々に分割（split）してから……
        md_chunks = md_splitter.split_documents(md_docs)
        py_chunks = py_splitter.split_documents(py_docs)

        # 5. 最後に合体（+）させますわ！
        chunks = md_chunks + py_chunks
        print(f"✂️ Total Chunks: {len(chunks)}")
