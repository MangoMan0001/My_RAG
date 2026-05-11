# main.py
import fire
from .cli import RAGCLI


def main() -> None:
    fire.Fire(RAGCLI)


if __name__ == '__main__':
    main()
