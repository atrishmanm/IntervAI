"""
models/generator/train_instruction.py
=====================================
Stage 3: Instruction Tuning.
Thin wrapper — delegates to unified train.py.

Run:
    python models/generator/train_instruction.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from models.generator.train import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--stage", "instruction"] + [a for a in sys.argv[1:] if a not in ("instruction",)]
    main()
