#!/usr/bin/env python3
"""CLI wrapper for the frozen independent final-test evaluator."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.frozen_final_evaluation import main


if __name__ == "__main__":
    main()
