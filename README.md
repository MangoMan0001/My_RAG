*This project has been created as part of the 42 curriculum by ayhirose.*

# RAG against the machine

### Prerequisites
**1. RAG**\
Retrieval-Augmented Generation (RAG)\
A method that empowers a model to answer questions by providing it with knowledge outside of its pre-trained memory. It consists of the following sequence:

- **Indexing**: Creating a dictionary-like structure from the data to be added (in this project, `.md` and `.py` files) to enable efficient searching.
- **Retrieval**: Using Natural Language Processing (NLP) techniques like `TF-IDF`, `BM25`, or semantic search to extract key terms from the prompt and retrieve highly relevant data from the index.
- **Augmentation**: Embedding the external data (context) obtained through retrieval into the user's prompt (question). In this project, to prevent hallucinations based on the model's pre-trained memory, a strong system prompt is used to force the generation of answers based solely on the retrieved information.
- **Generation**: Generating an answer from the augmented context. This computation is highly resource-intensive for LLMs, and many models achieve high speed through parallel processing supported by GPUs.

**2. TF-IDF**\
An evaluation procedure used in NLP. A classic formula that considers a word important if it appears frequently in a specific document (TF) but rarely across other documents (IDF). It has a drawback where longer texts tend to receive higher scores artificially.
> Term Frequency (TF) * Inverse Document Frequency (IDF)

**3. BM25**\
An evaluation procedure used in NLP that overcomes the long-text drawbacks of `TF-IDF`. This method ensures that the score does not inflate disproportionately even if the frequency of a word becomes extremely high.
> The package used to run the BM25 algorithm is `bm25s`.

**4. Recall@k**\
An evaluation metric for search systems in NLP. It represents the proportion of correct answers (ground truth) found within the top `k` results proposed by the model.

### Description
This task utilizes an LLM (Large Language Model). The objective is to analyze natural language prompts and generate answers based on new external sources that are not present in the existing training data.

**Personal Goal**
- High-speed answer generation using only a CPU.

**Packages Used**
> Package management is handled using `python uv`.
```
accelerate>=1.13.0
bm25s>=0.3.8
fire>=0.7.1
flake8>=7.3.0
flake8-bugbear>=25.11.29
flake8-pyproject
langchain>=1.2.17
langchain-community>=0.4.1
langchain-unstructured>=1.0.1
mypy>=1.19.1
pep8-naming>=0.15.1
pydantic>=2.12.5
pystemmer>=3.0.0
torch>=2.11.0
tqdm>=4.67.3
flake8-docstrings
llama-cpp-python>=0.3.25
```


**Directory Structure**
```
.
├── Makefile
├── README.md             # English documentation (Requirement)
├── README_JP.md          # Japanese documentation
├── pyproject.toml        # Configuration for dependencies and linters (flake8, mypy)
├── uv.lock               # uv dependency lock file
├── .gitignore
├── .python-version
│
├── src/
│   └── student/
│       ├── __main__.py   # Execution module entry point
│       ├── __init__.py
│       ├── cli.py        # CLI command definitions (Fire)
│       └── core/         # Core logic of the RAG system
│           ├── answer.py     # Answer generation module (llama.cpp)
│           ├── evaluater.py  # Evaluation module
│           ├── indexer.py    # Chunking and index creation module
│           ├── models.py     # Data model definitions using Pydantic
│           └── searcher.py   # Search module using BM25
│
├── data/                 # Data storage directory
│   ├── datasets/         # Question datasets (Answered/Unanswered)
│   ├── model/            # Downloaded LLM model file (GGUF)
│   ├── output/           # Output destination for search results and generated answer JSONs
│   ├── processed/        # Storage for chunk data and BM25 index
│   └── raw/              # Raw data for index creation (e.g., vLLM repository)
│
└── moulinette_pkg/       # Evaluation system package (Provided files)
```

### Instructions

This program requires Python 3.10 or higher. uv is used for package management.

1. **Installation**
```bash
make install
```
This sets up a virtual environment (.venv) and installs the necessary dependencies. It also simultaneously downloads the mandatory Qwen3-0.6B-Q8_0.gguf model (to data/model).

2. **Execution**
```bash
make run
```
Displays the help menu for the main program. Since there are various execution methods, the run command defaults to showing help.

>Note\
The program may not run correctly in a global environment without dependencies. Please run it through the .venv installed via make install.\
Execution format: uv run <execution_program>

**Create Index** \
Generates a searchable index using BM25. By default, the indexer module references ./data/raw/vllm-0.10.1 as the target data.
```bash
uv run python -m student index --max_chunk_size 2000
```
```bash
Available Flags:
--index_dir: str = "data/processed"     Specify the directory to save files.
--max_chunk_size: int = 2000            Specify the maximum chunk size.
```


**Search**\
Searches the index for sources highly relevant to the query using BM25. Results are output to the terminal in MinimalSearchResults format.

>See src/student/core/models.py for format details.\
Note: The index must be generated before performing this operation.
```bash
uv run python -m student search "How to configure OpenAI server?"
```

```bash
Available Flags:
--k: int = 5                            Number of sources to search for.
--query: str                            The search query.
--index_dir: str = "data/processed"     The index to reference.
--question_id: str = "q1"               Unique ID associated with the query.
```


**Dataset Search**\
Reads a dataset file in RagDataset format and performs searches in batch. Results are saved in the specified folder in StudentSearchResults format.

>See src/student/core/models.py for format details.\
Note: The index must be generated before performing this operation.
```bash
uv run python -m student search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results
```

```bash
Available Flags:
--k: int = 10
--index_dir: str = "data/processed"
--save_directory: str = "data/output/search_results"
--dataset_path: str = 'data/datasets/UnansweredQuestions/dataset_docs_public.json'
```


**Answer**\
Generates an answer using the LLM (Qwen3-0.6B) based on the sources relevant to the provided query. The result is output to the terminal in StudentSearchResultsAndAnswer format.

>See src/student/core/models.py for format details.\
Note: The index must be generated before performing this operation.

```bash
uv run python -m student answer "How to configure OpenAI server?" --k 10
```

```bash
Available Flags:
--query: str
--k: int = 10
--index_dir: str = "data/processed"
```


**Dataset Answer**\
Reads a dataset file in StudentSearchResults format and generates answers in batch. Results are saved in the specified folder in StudentSearchResultsAndAnswer format.

>See src/student/core/models.py for format details.\
Note: The index must be generated before performing this operation.
```bash
uv run python -m student answer_dataset --student_search_results_path data/output/search_results/dataset_docs_public.json --save_directory data/output/search_results_and_answer
```

```bash
Available Flags:
--save_directory: str = "data/output/search_results_and_answer"
--student_search_results_path: str = "data/output/search_results/dataset_docs_public.json"
```


**Evaluate**\
Evaluates the consistency of the searched sources using Recall@k. The evaluation content is identical to moulinette. It reads a dataset file in StudentSearchResultsAndAnswer format. Results are output to the terminal.

>See src/student/core/models.py for format details.\
Note: The index must be generated before performing this operation.

>Performance Requirements (quoted from task PDF):
>- Indexing time: Maximum 5 minutes.
>- Cold start latency: Maximum 60 seconds (First search after system startup, including model loading).
>- Warm state search throughput: Maximum 90 seconds for 1000 questions (After cold start).
>- Recall@5: 80% for doc questions, 50% for code questions.
```bash
uv run python -m student evaluate --student_answer_path data/output/search_results/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000
```

```bash
Available flags
--k: int = 10
--max_context_length: int = 2000 Maximum text length evaluated per query
--student_answer_path: str = "data/output/search_results/dataset_docs_public.json" File path to be evaluated
--dataset_path: str = "data/datasets/AnsweredQuestions/dataset_docs_public.json" File path for evaluation criteria
```


3. **Other Makefile Commands**
```bash
make lint
make lint-strict
```
Runs static type analysis and style checking using flake8 and mypy.

```bash
make debug
```
Runs the program in debug mode using pdb.

```bash
make clean
```
Removes cache files. fclean can also be used to remove the virtual environment.


## Additional sections

### System Architecture
Regarding the components of the RAG pipeline and their interactions:

The entry point command-line is managed by the python fire package, which orchestrates each Python module. The RAG pipeline is built on the langchain framework, with indexing and searching handled by the bm25s package. Answer generation using Qwen is implemented via llama_cpp.

###	Chunking Strategy
Approach to document segmentation:

- Set a 100-character overlap between chunks to prevent disjointed fragments that lack context.
- Configured stopwords='en' in bm25s to remove meaningless frequent words (like 'a', 'is', 'the'), contributing to both speed and accuracy improvements.
- Added Stemming to strip suffixes and prefixes, reducing words to their root form (e.g., "fishing", "fished", "fisher" to "fish"). This significantly contributed to improving BM25 accuracy.

### Search Method
Details of the search algorithm and ranking mechanism\
The IDs of the chunks retrieved using the bm25 algorithm are looked up in the locally stored index. This process is repeated `k` times.\
The `retrieve` method of `bm25s` retrieves `k` files at a time. Since files with lower list indices are more relevant to the query, a ranking is created in order based on this.

### Performance Analysis
Recall@k-score and System Performance
>Performance Requirements (cited from the challenge PDF)
>- Index creation time: Maximum 5 minutes
>- Cold-start latency: Maximum 60 seconds (first search after system startup, including model loading)
>- Search throughput in warm state: Maximum 90 seconds for 1,000 queries (after cold start)
>- Recall@5: 80% for document-based questions, 50% for code-based questions

Index creation is completed in 10 seconds or less. Delays are likely minimized because the process primarily extracts only the `.py` and `.md` files used for this task from the raw `vllm-0.10.1` data, and because `Numpy` and `Scipy`—the foundations of the `bm25s` algorithm used—are written in C.
Searches also use `bm25s`.
Recall@k shows a significant difference between the score from the standard `bm25s` search alone and the score after implementing `stemming` and `stopwords` (as mentioned in the section on chunking strategies), indicating that these measures contributed to improved accuracy.

```python
<pre score> .md
Recall@1: 0.44
Recall@3: 0.46
Recall@5: 0.51
Recall@10: 0.56
```
```python
<post score> .md
Recall@1: 0.72
Recall@3: 0.79
Recall@5: 0.82
Recall@10: 0.86
```
```python
<pre score> .py
Recall@1: 0.14
Recall@3: 0.19
Recall@5: 0.25
Recall@10: 0.28
```
```python
<post score> .py
Recall@1: 0.41
Recall@3: 0.51
Recall@5: 0.55
Recall@10: 0.59
```

### Design Decisions
Key choices made during implementation\
- Since this was my first time using the package, I decided to write the provided methods exactly as they were, without abbreviating them.\
- With `flake8`'s default maximum column length of 79 characters, unintended line breaks occurred frequently, significantly reducing readability; therefore, I set the limit to 120 characters.

### Challenges Encountered
A record of difficulties encountered and their solutions\

**Response Speed**
- Issue: Most model manipulation packages recommended in the problem PDF, such as `transformers`, support parallel processing using GPUs. When running these packages on a school PC with only a CPU, each response took over 60 seconds. Combined with a drop in processing speed due to heat, it took over two hours to process the 100 questions in the dataset.

- Solution: We switched the model operations to the `llama_cpp` package, which is written in C++ and designed for CPU execution. We also changed the specification to install the local LLM as an 8-bit local file in the GGUF (GPT-Generated Unified Format) before execution.

- Result: Response time was reduced to 4 seconds per question, allowing 100 questions to be processed in 6–7 minutes.

**Low Recall@k Scores**
- Issue: `Recall@k` did not meet performance requirements.

- Solution: The `stemming` and `stopwords` settings for `bm25s`, which were implemented in the `indexer` module, were not being carried over to the `searcher` module. Although chunking was functioning, we modified the system so that these settings are also carried over during loading at the time of search.

- Result: Scores increased by 1.5x for `.md` files and 2.2x for `.py` files.

### Usage Example: Clearly demonstrate system execution examples
Please refer to the `Instructions` section for basic operations.

Here is an explanation of the `moulinette` operation, which was previously difficult to understand.
```bash
cd moulinette_pkg
./moulinette-ubuntu evaluate_student_search_results ../data/output/search_results/dataset_docs_public.json ../data/datasets/AnsweredQuestions/dataset_docs_public.json --threshold 0.8
```

### Resources
AI
- Brainstorming on algorithm design and debugging for logit manipulation in constrained decoding.
- Analyzing error logs from `flake8` and `mypy`, and optimizing `pyproject.toml`.
- Translating docstrings and READMEs into English, and assisting with documentation organization.


`python uv` official documentation, official AI\
`langchain` official documentation, official AI\
`llama-cpp-python` official documentation, official AI
