# INTERVUE — Task Tracker

> Status of every task. Anyone (human or AI) should be able to pick up from here.
> Last updated: 2026-08-17

**Legend:** ✅ done · 🔄 in progress · ⬜ not started · ❌ blocked

---

## Phase 0 — Data Engineering

| # | Task | File | Status | Notes |
|---|------|------|--------|-------|
| 0.1 | Download StandardLogic 100K | `download_datasets.py` | ✅ | 286 MB, 100K conversations |
| 0.2 | Download CodeAlpaca 20K | `download_datasets.py` | ✅ | 6.7 MB |
| 0.3 | Download CodeFeedback 156K | `download_datasets.py` | ✅ | 115 MB |
| 0.4 | Download Mohler ASAG | `download_datasets.py` | ✅ | 2,273 real student answers |
| 0.5 | Download StarCoder large | `download_more_data.py` | ✅ | 1.7 GB, ~200K samples |
| 0.6 | Download CodeSearchNet | `download_more_data.py` | ✅ | 178 MB |
| 0.7 | Download OpenCodeInstruct | `download_more_data.py` | ✅ | 1.2 GB |
| 0.8 | Download OASST coding | `download_more_data.py` | ✅ | 23 MB |
| 0.9 | Clean data → questions | `clean_data.py` | ✅ | → `questions.jsonl` |
| 0.10 | Build question bank (SQLite) | `build_question_bank.py` | ✅ | 32,436 questions |
| 0.11 | Build concept graph | `build_concept_graph.py` | ✅ | 20 concepts |
| 0.12 | Transform → training files | `transform_training_data.py` | ✅ | instruction/eval/followup/dialogue |

**TOTAL DATA: ~3.5 GB** (data/raw/)

---

## Phase 1 — Tokenizer

| # | Task | File | Status | Notes |
|---|------|------|--------|-------|
| 1.1 | Train 16K BPE tokenizer | `tokenizer/train_tokenizer.py` | ✅ | ByteLevel, new special tokens |
| 1.2 | Verify tokenizer output | sanity check | ⬜ | current saved tokenizer has vocab=2981 — must re-train to 16K before full training |

---

## Phase 2 — Model Architecture

| # | Task | File | Status | Notes |
|---|------|------|--------|-------|
| 2.1 | Decoder-only Transformer | `models/generator/model.py` | ✅ | small/medium/large sizes |
| 2.2 | Forward + loss + generation | verified | ✅ | all 3 sizes run |
| 2.3 | env_config (dual-env) | `env_config.py` | ✅ | auto-detect GPU/env |
| 2.4 | Hardware-aware train_utils | `models/generator/train_utils.py` | ✅ | grad accum, FP16, checkpoints, multi-schema dataset |

---

## Phase 3 — Training (Curriculum)

| # | Task | File | Status | Notes |
|---|------|------|--------|-------|
| 3.1 | Unified trainer | `models/generator/train.py` | ✅ | `--stage` flag, resume, checkpoints, smoke-tested |
| 3.2 | Stage wrappers | `train_pretrain.py` … `train_followup.py` | ✅ | delegate to unified trainer |
| 3.3 | Kaggle run script | `kaggle/train_on_kaggle.py` | ✅ | clones repo, runs all stages |
| 3.4 | Kaggle guide | `kaggle/README.md` | ✅ | dataset upload, notebook setup, resume |
| 3.5 | Full training run (Kaggle) | — | ⬜ | requires user to run on Kaggle |
| 3.6 | Retrain tokenizer to 16K | `tokenizer/train_tokenizer.py` | ⬜ | vocab currently 2981 |

---

## Phase 4 — Runtime Integration

| # | Task | File | Status | Notes |
|---|------|------|--------|-------|
| 4.1 | Contextual answer scorer | `analysis/semantic_scorer.py` | ✅ | semantic + concept coverage + accuracy + quality |
| 4.2 | Legacy keyword scorer fallback | `analysis/scorer.py` | ✅ | unchanged, used on error |
| 4.3 | Candidate state tracking | `orchestrator/candidate_state.py` | ✅ | EMA concept scores, normalized 0-1 |
| 4.4 | Orchestrator uses candidate state | `orchestrator/state_machine.py` | ✅ | difficulty + concept-guided next question |
| 4.5 | Backend endpoints | `backend/main.py` | ✅ | start/chat/status/report |
| 4.6 | Panel report generation | `analysis/report.py` | ✅ | weak concepts + gaps + model answer + suggestions |
| 4.7 | Frontend display | `frontend/index.html` | ✅ | new report schema + panel report renderer |

---

## Phase 5 — Evaluation

| # | Task | File | Status | Notes |
|---|------|------|--------|-------|
| 5.1 | Perplexity check | evaluation | ⬜ | target < 25 |
| 5.2 | Mohler score correlation | evaluation | ⬜ | Pearson r > 0.6 |
| 5.3 | Human interview eval | evaluation | ⬜ | 5+ people |

---

## Known Issues / Decisions

1. **StarCoder is not gated** — connection was flaky on Windows; retry-with-backoff in
   the download script fixed it; full 1.7GB downloaded.
2. **Saved tokenizer is stale** (vocab=2981). The rewritten `train_tokenizer.py`
   produces 16K vocab, but it must be re-run against the full corpus before any real
   training. The smoke tests above used the stale tokenizer.
3. **GTX 1650 memory:** batch=2 + grad-accum 8 → effective batch 16. Confirmed working.
4. **Smoke-test checkpoints deleted** after verification — the `saved/` dir is clean.
5. **Data is real** — no synthetic student answers. Synthetic *transformations* of real
   data are OK and clearly labeled.
6. **Model evaluation:** for Mohler correlation, only use the held-out portion of the
   real student answers (never the training split).