# INTERVUE — Context & Learnings

> This file exists so that ANY AI model (or human) can pick up this project without
> re-discovering everything the hard way. It documents what was tried, what failed,
> what worked, and the key architecture decisions.

---

## Project One-Liner
Build a small from-scratch decoder-only Transformer that conducts **contextual technical
interviews** — understanding candidate answers (not keyword matching), giving human-like
feedback, probing gaps, and producing a panel-style final report.

---

## Environment & Hardware (critical constraints)

- **Local laptop:** Windows, GTX 1650 **4GB VRAM**, Python 3.14.
  - 4GB VRAM means batch=2, grad_accum=8 (effective 16), FP16, small model (~5.9M).
  - Anything bigger OOMs. This was the single most important constraint discovered.
- **Kaggle:** `/kaggle/working`, T4/P100 16GB → batch=16, medium model (~10.7M).
- **Colab:** `/content`, T4 16GB → same as Kaggle.
- The fix: `env_config.py` auto-detects `Path("/kaggle/working").exists()` and
  `torch.cuda.get_device_properties(0).total_memory`. **Same script, both machines.**

### The mistake that shaped everything
The original plan proposed training from scratch on a GTX 1650. Early overfitting appeared
after ~11 epochs on a tiny (~530-record) dataset. The lesson: **more real data, not more
training time, is what fixes overfitting.** This is why we downloaded to 3.5GB.

---

## Data (what we actually have)

All in `data/raw/`, real data only (no fake/synthetic student answers):

| File | MB | What it is |
|------|----|------------|
| starcoder_large.jsonl | 1690 | Real GitHub Python code (~200K files) |
| opencodeinstruct.jsonl | 1221 | Code instruction/response pairs |
| conversations.jsonl | 286 | 100K coding-interview conversations |
| codesearchnet.jsonl | 178 | Code + documentation pairs |
| codefeedback.jsonl | 115 | High-complexity code instructions |
| oasst_coding.jsonl | 23 | Coding conversations |
| codealpaca.jsonl | 7 | Code instruction following |
| mohler_asag.jsonl | 0.9 | **Real student answers, human-graded 0-5** |
| mmlu_cs.json | 0.3 | CS conceptual MCQs |
| cruxeval.jsonl | 0.2 | Code execution/trace problems |

### Key insight about Mohler
Mohler ASAG (`nkazi/MohlerASAG`, split `raw`, subset `open_ended`) gives 2,273 real
student answers, each scored by two graders (average = gold). It is the ONLY source of
real candidate answers in the project. It's tiny (0.9MB) but **structurally priceless**:
it teaches the evaluator what a real partial/incorrect answer looks like vs a reference.

---

## Dataset access gotchas (Windows + HuggingFace)

1. **StarCoder (`bigcode/starcoderdata`) is NOT gated**, but:
   - The `python` subdir path via `data_dir=` failed; use `split="train"` directly.
   - Streaming over HTTPS is flaky on Windows → `[WinError 10054] connection forcibly closed`.
   - Retry with backoff inside the script; it eventually completed (1.7GB / ~200K files).
2. **Symlinks warning**: HuggingFace cache warns about symlinks on Windows. Harmless —
   disable with `HF_HUB_DISABLE_SYMLINKS_WARNING=1` if it bothers you.
3. **Parquet auto-conversion** sometimes requires the `datasets` + `pyarrow` combo —
   already in requirements.
4. `stindardlogic/coding-interview-sft-100k` field is `conversations` = list of
   `{from: human|gpt, value: text}`. Transform accordingly.

---

## Tokenizer learnings

- 8K vocab (original) was too small for a generative model. Bumped to **16K**.
- Switched pre-tokenizer from `Whitespace` to **ByteLevel** — vastly better for code
  (handles `(){}[]_=` etc. without exploding token counts).
- Special tokens now: `[PAD] [UNK] [CLS] [SEP] [MASK] <|system|> <|user|> <|assistant|>
  <|end|> <|code|> <|/code|>`.
- Trained on ALL datasets combined, not just one corpus. This single change improves
  coverage of both prose and code tokens.

---

## Model architecture decisions

- **Decoder-only GPT-style** (not encoder) because we need generation (follow-ups,
  feedback, dialogue) not just classification.
- **Pre-LayerNorm** (not post-LN): markedly more stable for small models trained from
  scratch — avoids loss spikes.
- **Learned positional embedding** (not sinusoidal): the model learns position patterns
  better on limited data.
- **Weight tying**: LM head shares weights with token embedding → saves ~4M params.
- **GELU** in FFN.
- Attention is computed manually (Q/K/V split, scaled dot-product, causal mask) rather
  than using `nn.MultiheadAttention` — full control and fewer surprises.
- Effective batch via gradient accumulation is what makes 4GB VRAM feasible.

---

## Training curriculum (why ordered this way)

1. **Pretrain on raw code** → the model learns Python syntax/structure. Without this,
   later stages produce garbage tokens.
2. **Domain (ChatML CS conversations)** → the model learns CS facts and the
   `<|user|>/<|assistant|>` format.
3. **Instruction** → follows instructions ("Explain X", "What is Y?").
4. **Interview dialogue** (MOST critical) → acts as the interviewer: asks questions,
   acknowledges answers, asks follow-ups.
5. **Evaluator** (Mohler) → scores a student answer, says what's covered/missing.
6. **Follow-up generation** → turns detected gaps into probing questions.

**Warning encountered:** a 1M-param model cannot do open-ended generation well (it
overfits and memorizes rather than generalizing). 6-24M params is the pragmatic range
for this project — enough to produce short coherent CS text.

---

## Answer evaluation design (the "not just keywords" part)

The current rule-based scorer (`analysis/scorer.py`) matches keywords and counts words.
That is the baseline. The upgrade path:

1. **Semantic similarity**: embed candidate answer + reference, cosine similarity.
   (Use the trained ranker bi-encoder, or a pooled embedding from the generator.)
2. **Concept coverage**: for each concept in the question's concept list, check if the
   candidate's answer is semantically close to a canonical phrase for that concept.
3. **Accuracy**: check for statements that contradict the reference (via similarity to
   "wrong answer" prototypes or explicit negation check).
4. **Completeness**: fraction of expected reasoning steps present.
5. **Depth/quality**: length, structure words, code present, complexity mentioned.

Output: 0-100 score + per-concept pass/partial/miss + human-like feedback string.

---

## Candidate state & adaptation

`orchestrator/candidate_state.py` tracks per-concept scores using an **EMA**:
`new = 0.3 * normalized_score + 0.7 * old`. Weakness = score < 0.5, strength = score >= 0.7.
Difficulty auto-adjusts from recent 3-question average (>=4.0 → harder, <=2.0 → easier).
The concept graph (20 concepts with prerequisites) lets the system ask prerequisite
questions before advanced ones (e.g., don't ask Dijkstra if priority queues are weak).

---

## Panel report (final output of an interview)

After the interview ends (10 turns), the system produces:
- Overall score and verdict
- Per-concept mastery (mastered / developing / beginner / not started)
- Top weaknesses with the specific **question + answer + where the gap was**
- The **model/correct answer** for each weak question
- Concrete **improvement suggestions** (topics to study, what to practice)
- Difficulty trajectory over the session

This is what makes it feel like a real interview panel rather than a quiz bot.

---

## Known failures & their fixes

| Problem | Root cause | Fix |
|---------|-----------|-----|
| OOM on GTX 1650 | batch 8 with 4GB VRAM | batch 2 + grad accum 8 + FP16 |
| Overfitting after 11 epochs | tiny dataset (~530 records) | get real data, 3.5GB |
| StarCoder connection reset | flaky HTTPS on Windows | retry w/ backoff, larger cap |
| 8K vocab too small | whitespace tokenizer on code | 16K vocab + ByteLevel |
| model.py top_p bug | scatter mask logic | simplified top-k + top-p sampling |
| Training crash mid-epoch | corrupt checkpoint | atomic save (tmp + rename) + resume |
| Epoch resumes from 0 | no step tracking | store epoch/step in checkpoint |

---

## How to verify a new change (quick loop)

1. `python -c "import env_config; env_config.print_env_summary()"` → confirms env.
2. `python models/generator/model.py` → confirms forward/backward/generate.
3. Run a training script with a tiny `limit` to smoke-test before full run.
4. Check checkpoints exist in `models/generator/saved/` with expected epoch.
5. Run backend, hit `/api/start` then `/api/chat` with a sample answer, inspect the
   evaluation JSON.

---

## Kaggle workflow (for whoever runs it next)

1. Create a Kaggle notebook, set GPU P100/T4 x2 accelerator.
2. **Add Data** → the raw dataset files (upload `data/raw/*.jsonl` as a Kaggle Dataset,
   or use `upload_dataset` API). They appear in `/kaggle/input/`.
3. Run `!git clone https://.../IntervAI` into `/kaggle/working/`.
4. Run the stage scripts in order (see `kaggle/README.md`).
5. Checkpoints auto-save to `/kaggle/working/IntervAI/models/generator/saved/`.
6. Download the `saved/` folder back to local `models/generator/saved/`.

**Important:** never run training locally on the GTX 1650 for stages 1-3 (huge data).
Only stages 4-6 (smaller data) are feasible locally. Kaggle does the heavy lifting.