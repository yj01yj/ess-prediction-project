from __future__ import annotations

from pathlib import Path
import importlib
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


if __name__ == "__main__":
    module = importlib.import_module("feature_engineering")
    if not hasattr(module, "main"):
        raise AttributeError("src/feature_engineering.py 안에 main() 함수를 만들어 주세요.")
    module.main()
