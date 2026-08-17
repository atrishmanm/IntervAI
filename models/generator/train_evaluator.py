"""
models/generator/train_evaluator.py
===================================
Stage 5: Answer Evaluation Tuning (Mohler real student answers).
Thin wrapper — delegates to unified train.py.

Run:
    python models/generator/train_evaluator.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from models.generator.train import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--stage", "evaluator"] + [a for a in sys.argv[1:] if a not in ("evaluator",)]
    main()
