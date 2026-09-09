"""
run_smoke.py — fast end-to-end MOCK training run (CPU-friendly).
=================================================================
Runs the FULL 8-stage curriculum (pretrain → … → negotiation) on a tiny
slice of data with the small model, writing checkpoints to an ISOLATED
directory so your real models/generator/saved checkpoints are untouched.

Use it to verify training-code changes before burning a Kaggle session:

    python run_smoke.py            # 30 examples per file (default)
    python run_smoke.py 100        # more examples per file

Real-Kaggle equivalents:
    python kaggle/train_on_kaggle.py --stage all --limit 30 --time-budget 30
    kaggle/intervai_research_notebook.py --smoke-test   (notebook pipeline)
"""

import os
import sys
import shutil
from pathlib import Path

# Force the 14M model even on big-GPU machines (must run before env_config import)
os.environ.setdefault("INTERVUE_MODEL", "small")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 30

import env_config

SMOKE_DIR = ROOT / "smoke_run"
SMOKE_CKPT = SMOKE_DIR / "checkpoints"
if SMOKE_DIR.exists():
    shutil.rmtree(SMOKE_DIR)
SMOKE_CKPT.mkdir(parents=True)

# Redirect checkpoints before importing train (train.py binds SAVE_ROOT by value)
env_config.SAVE_ROOT = SMOKE_CKPT

import models.generator.train as T

T.SAVE_ROOT = SMOKE_CKPT

# Quick tokenizer health check: newlines must not be [UNK] after the
# initial_alphabet fix (retrain with `python tokenizer/train_tokenizer.py`).
from models.generator.train_utils import load_tokenizer

_tok = load_tokenizer(env_config.ROOT / "tokenizer" / "saved" / "tokenizer.json")
_sample = _tok.encode("line one\n    indented code").tokens
if "[UNK]" in _sample:
    print("[SMOKE] WARN: tokenizer still emits [UNK] for newlines — "
          "retrain it: python tokenizer/train_tokenizer.py")
else:
    print("[SMOKE] Tokenizer newline check: OK (no [UNK])")

print(f"[SMOKE] Running ALL 8 stages with limit={LIMIT} examples/file, "
      f"small model, checkpoints -> {SMOKE_CKPT}\n")

sys.argv = ["train.py", "--stage", "all", "--limit", str(LIMIT)]
T.main()

EXPECTED = [
    "pretrained.pt", "domain_tuned.pt", "instruction_tuned.pt",
    "interview_tuned.pt", "evaluator.pt", "final_model.pt",
    "resume_finetuned.pt", "negotiation_tuned.pt",
]
produced = sorted(p.name for p in SMOKE_CKPT.glob("*.pt"))
missing = [n for n in EXPECTED if n not in produced]

print("\n" + "=" * 60)
if missing:
    print(f"[SMOKE] FAIL — missing checkpoints: {missing}")
    print(f"[SMOKE] produced: {produced}")
    sys.exit(1)
print(f"[SMOKE] PASS — all {len(EXPECTED)} stage checkpoints produced:")
for n in produced:
    print(f"    - {n}")
print(f"[SMOKE] Artifacts in {SMOKE_DIR} (safe to delete)")
print("=" * 60)
