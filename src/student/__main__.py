"""CLI entry point for the RAG (Retrieval-Augmented Generation) system."""
import fire
from .cli import RAGCLI


def main() -> None:
    """Initialize and execute the command-line interface for the RAG system."""
    fire.Fire(RAGCLI)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'error: {e}')
