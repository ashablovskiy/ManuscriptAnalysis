from __future__ import annotations

import sys
from pathlib import Path


def _inject_src() -> None:
    repo_root = Path(__file__).resolve().parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main() -> None:
    _inject_src()
    from arabic_witness.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
