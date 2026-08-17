# INTERVUE — Implementation Plan

> A ChatGPT-like AI technical interviewer, trained from scratch on real coding/CS data.
> The goal is a small (~6-24M parameter) decoder-only Transformer that understands,
> evaluates, and responds to candidate answers the way a real human interviewer would —
> not just by matching keywords, but by understanding context, accuracy, and completeness.

---

## 1. What We Are Building

### The Product
A web-based AI interviewer (FastAPI + single-page frontend) where a candidate:
1. Starts an interview session
2. Answers technical questions in natural language
3. Receives **contextual, human-like evaluation** — not just "keywords matched = score"
4. Gets follow-up questions that probe gaps in their understanding
5. At the end, receives a **full panel-style report**: weak concepts, where the answer
   had gaps, the model/correct answer, and concrete ways to improve

### The Model
A **decoder-only GPT-style Transformer trained from scratch** on ~3.5 GB of real coding data:

| Size | Params | embed_dim | layers | when to use |
|------|--------|-----------|--------|-------------|
| small | ~5.9M | 192 | 6 | GTX 1650 4GB |
| medium | ~10.7M | 256 | 8 | T4/P100 (Kaggle/Colab) |
| large | ~24.3M | 384 | 10 | 16GB+ VRAM |

### Training Stages (curriculum)

| Stage | Name | Data | Purpose |
|-------|------|------|---------|
| 1 | Code Pretraining | starcoder_large (1.7GB) + codesearchnet (0.18GB) | Learn code structure/language |
| 2 | CS Domain | conversations (0.28GB) + opencodeinstruct (1.2GB) + codefeedback (0.11GB) | Learn CS concepts in ChatML |
| 3 | Instruction | codealpaca + opencodeinstruct | Follow instructions |
| 4 | Interview Dialogue | generated interview dialogues | Behave like an interviewer |
| 5 | Answer Evaluation | Mohler ASAG (real student answers) + synthetic | Evaluate answers contextually |
| 6 | Follow-up Generation | followup_training | Generate probing questions |

---

## 2. Current Data Inventory (~3.5 GB)

| File | Size | Records | Use |
|------|------|---------|-----|
| `starcoder_large.jsonl` | 1690 MB | ~200K | Stage 1 pretraining (code) |
| `opencodeinstruct.jsonl` | 1221 MB | ~150K | Stage 2/3 instruction |
| `conversations.jsonl` | 286 MB | 100K | Stage 2 interview dialogues |
| `codesearchnet.jsonl` | 178 MB | ~120K | Stage 1 pretraining (code+docs) |
| `codefeedback.jsonl` | 115 MB | 156K | Stage 2/3 |
| `oasst_coding.jsonl` | 23 MB | ~50K | Stage 2 conversation |
| `codealpaca.jsonl` | 7 MB | 20K | Stage 3 instruction |
| `mohler_asag.jsonl` | 0.9 MB | 2,273 | Stage 5 real student answers |
| `mmlu_cs.json` | 0.3 MB | ~500 | Stage 2 CS knowledge |
| `cruxeval.jsonl` | 0.2 MB | 800 | Stage 2 code reasoning |

---

## 3. Architecture

```
                    ┌────────────────────────────┐
                    │      RAW DATASETS (3.5GB)  │
                    └─────────────┬──────────────┘
                                  ▼
                    ┌────────────────────────────┐
                    │    DATA ENGINEERING         │
                    │  clean → transform → split  │
                    └─────────────┬──────────────┘
                                  ▼
                    ┌────────────────────────────┐
                    │   TOKENIZER (16K vocab)    │
                    └─────────────┬──────────────┘
                                  ▼
                    ┌────────────────────────────┐
                    │  INTERVIEW GENERATOR LM    │
                    │  6-stage curriculum train  │
                    └─────────────┬──────────────┘
                                  ▼
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
  ┌──────────┐            ┌──────────────┐          ┌──────────────┐
  │ GENERATE │            │  EVALUATE    │          │   REPORT     │
  │ response │            │  answer      │          │   panel      │
  └──────────┘            └──────────────┘          └──────────────┘
        │                         │                         │
        └─────────────┬───────────┘                         │
                      ▼                                     │
           ┌────────────────────┐                           │
           │ CANDIDATE STATE    │                           │
           │ (concept mastery)  │◄──────────────────────────┘
           └────────────────────┘
```

### Answer Evaluation (the "ChatGPT-like" part)

The scorer must go beyond keyword matching:

```
Input:
  - question
  - candidate answer
  - reference/model answer
  - question type & topic
        │
        ▼
1. Semantic similarity (embedding) between answer and reference
2. Concept coverage (which required concepts were hit/missed)
3. Accuracy check (wrong statements vs reference)
4. Completeness (how much of the expected structure is present)
5. Complexity analysis (for theoretical questions)
6. Quality of explanation (depth, structure)
        │
        ▼
Output: score 0-100 + list of covered/missing concepts + feedback
```

---

## 4. Runtime System (FastAPI + Frontend)

```
POST /api/start → greeting + first question
POST /api/chat  → candidate answer → contextual evaluation + next question
GET  /api/status → system readiness
GET  /api/report → panel-style final report (after interview)
```

The orchestrator tracks:
- concept scores per candidate (EMA updates)
- weaknesses & strengths
- current difficulty (auto-adjusts)
- question history
- final report generation

---

## 5. Hardware Strategy

| | Local laptop | Kaggle | Colab |
|---|---|---|---|
| GPU | GTX 1650 (4GB) | T4 / P100 (16GB) | T4 (16GB) |
| Model | small (~5.9M) | medium (~10.7M) | medium |
| Batch | 2 (accum x8) | 16 (accum x4) | 16 |
| FP16 | yes | yes | yes |
| Checkpoints | `models/generator/saved/` | `/kaggle/working/...` | `/content/...` |

**Key:** `env_config.py` auto-detects the environment, so the SAME training script runs on all three with zero code changes.

---

## 6. Milestones

| Milestone | Deliverable | Status |
|-----------|-------------|--------|
| M1 | Data downloaded (~3.5GB) | ✅ Done |
| M2 | Data transforms produce training files | ✅ Done |
| M3 | Tokenizer trained (16K vocab) | ✅ Done |
| M4 | Model architecture verified | ✅ Done |
| M5 | Stage 1 pretraining (Kaggle) | 🔄 Next |
| M6 | Stages 2-6 curriculum training | ⬜ |
| M7 | Runtime integration (backend + frontend) | ⬜ |
| M8 | Panel report generation | ⬜ |
| M9 | Human evaluation | ⬜ |