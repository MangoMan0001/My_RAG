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
	uv sync
	mkdir -p data/model
	wget -nc https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf -O data/model/Qwen3-0.6B-Q8_0.gguf
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
	@rm -rf data/processed
	@rm -rf data/output
	@echo "Clean complete."

fclean: clean ## cleanに加えて仮想環境も削除
	@echo "Full Cleaning up..."
	@rm -rf .venv
	@rm -rf ~/.cache/huggingface/hub/
	@echo "Full Clean complete."

re: fclean all
