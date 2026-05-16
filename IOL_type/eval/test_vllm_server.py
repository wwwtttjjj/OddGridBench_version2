#!/usr/bin/env python3
from pathlib import Path
import runpy


ROOT_SCRIPT = Path(__file__).resolve().parents[3] / "test_8080.py"


if __name__ == "__main__":
    runpy.run_path(str(ROOT_SCRIPT), run_name="__main__")
