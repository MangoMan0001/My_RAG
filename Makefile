# ==========================================
#  RAG Project Makefile
# ==========================================

# プロジェクト名とメインスクリプト
NAME        = rag

MAIN_DERECTRY = student

# ==========================================
#  Rules
# ==========================================

.PHONY: all install run debug clean lint lint-strict build re

all: install

# ------------------------------------------
#  Environment Setup
# ------------------------------------------
install: ## 仮想環境を作成し、依存関係をインストールする
	@echo "Creating virtual environment..."
	@echo "Installing dependencies..."
	@uv sync
	@mkdir -p data/datasets data/model data/output data/processed data/raw
	@wget https://cdn.intra.42.fr/document/document/48985/datasets_public.zip https://cdn.intra.42.fr/document/document/48987/vllm-0.10.1.zip https://cdn.intra.42.fr/document/document/48988/moulinette.zip
	@-unzip "*.zip"
	@mv datasets_public/public/* data/datasets
	@mv vllm-0.10.1 data/raw
	@wget -nc https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf -O data/model/Qwen3-0.6B-Q8_0.gguf
	@rm *.zip
	@rm -rf datasets_public
	@echo "Setup complete! Run 'make run' to start."

# ------------------------------------------
#  Execution
# ------------------------------------------
run: ## メインプログラムを実行
	@echo "Running $(NAME)..."
	uv run python -m $(MAIN_DERECTRY)

debug: ## pdbデバッガを使って実行
	@echo "Debugging $(NAME)..."
	uv run python -pdb -m $(MAIN_DERECTRY)

# ------------------------------------------
#  Quality Control
# ------------------------------------------
lint: ## Flake8とMypyによる静的解析を実行
	@echo "Running Linter (Standard)..."
	uv run flake8 .
	uv run mypy .

lint-strict: ## より厳しいMypyチェックを実行
	@echo "Running Linter (Strict)..."
	uv run flake8 .
	uv run mypy --strict .

# ------------------------------------------
#  Cleanup
# ------------------------------------------
clean: ## 一時ファイルやキャッシュを削除
	@echo "Cleaning up..."
	@rm -rf __pycache__
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf .mypy_cache
	@rm -rf .pytest_cache
	@rm -rf dist
	@rm -rf build
	@rm -rf *.egg-info
	@rm -rf .ruff_cache
	@echo "Clean complete."

fclean: clean ## cleanに加えて仮想環境も削除
	@echo "Full Cleaning up..."
	@rm -rf .venv
	@rm -rf ~/.cache/huggingface/hub/
	@rm -rf data
	@rm -rf moulinette_pkg
	@rm -rf datasets_public.zip vllm-0.10.1.zip moulinette.zip
	@rm -rf datasets_public evaluations exams exams_pkg private vllm-0.10.1
	@echo "Full Clean complete."

re: fclean all
