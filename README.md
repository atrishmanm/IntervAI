<div align="center">

# INTERVUE — AI Interview Simulator

A **state-of-the-art interview simulator** trained from scratch that conducts real interviews — not a chatbot, not a coding platform.

**Upload your resume → Get asked personalized questions → Receive a structured evaluation report.**

Unlike ChatGPT/Claude (which just chat), INTERVUE:
- Parses YOUR resume and asks about YOUR experience
- Adapts question difficulty based on YOUR performance  
- Scores on 5 dimensions (technical, behavioral, communication, problem-solving, cultural fit)
- Evaluates STAR method for behavioral answers
- Gives you a structured HIRE/MAYBE/NO HIRE recommendation

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

</div>

## Why This Isn't Replaceable by ChatGPT/Claude

| Feature | ChatGPT/Claude | INTERVUE |
|---------|---------------|----------|
| Resume-based questions | Generic | Asks about YOUR projects |
| Adaptive difficulty | Fixed | Gets harder if you answer well |
| Multi-dim scoring | "Good answer" | 15+ rubric dimensions |
| STAR evaluation | None | Scores S/T/A/R completeness |
| Structured report | None | HIRE/MAYBE/NO HIRE with breakdown |
| Interview simulation | Chat | Timed questions, phases, flow |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  INTERVUE Stack                      │
├─────────────────────────────────────────────────────┤
│  Resume Parser → Skill Extractor → Question Selector │
│       ↓              ↓                  ↓            │
│  Adaptive Difficulty Engine → Interview Flow         │
│       ↓              ↓                  ↓            │
│  125M Transformer ←→ STAR Evaluator ←→ Score Rubric  │
│       ↓              ↓                  ↓            │
│  Structured Feedback Report (5 dimensions)          │
└─────────────────────────────────────────────────────┘
```

## Interview Phases

1. **Warm-up** — Tell me about yourself, your background
2. **Technical** — Coding, system design, algorithms (adaptive difficulty)
3. **Behavioral** — STAR method questions (conflict, leadership, failure)
4. **Problem-solving** — Case studies, tradeoffs, architecture decisions
5. **Wrap-up** — Your questions, closing thoughts

## Model Training (125M params, SOTA)

- **Muon optimizer** — 2x faster than AdamW (proven at 16B scale)
- **Sequence packing** — 2-3x throughput
- **torch.compile** — 20-30% JIT speedup
- **Dynamic dropout** — Regularization adapts during training
- **WSD schedule** — Warmup-Stable-Decay learning rate
- **EMA** — Exponential moving average for stable checkpoints
- **15+ analytics metrics** — Throughput, gradient norms, stability score

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train tokenizer (case-sensitive, 16K vocab)
python tokenizer/train_tokenizer.py

# Run training (6 stages, ~2h on T4x2)
python models/generator/train.py --stage all

# Start backend
python backend/main.py
```

## Training Stages

| Stage | Data | Purpose |
|-------|------|---------|
| Pretrain | Code + educational text | Language fundamentals |
| Domain | CS conversations | Technical dialogue |
| Instruction | Instruction datasets | Following directions |
| Interview | Interview transcripts | Interview flow |
| Evaluator | Scoring datasets | Answer evaluation |
| Follow-up | Conversation data | Natural follow-ups |

## Project Structure

```
IntervAI/
├── models/generator/          # Transformer model + training
│   ├── model.py              # 125M param decoder-only Transformer
│   ├── train.py              # Unified training script
│   ├── train_utils.py        # Muon, WSD, EMA, packing
│   ├── analytics.py          # 15+ training metrics
│   └── interview_metrics.py  # BLEU, ROUGE, concept accuracy
├── orchestrator/
│   ├── interview_engine.py   # Resume parser + adaptive interview
│   └── state_machine.py      # Interview state management
├── tokenizer/                # Case-sensitive BPE tokenizer
├── backend/                  # FastAPI server
├── data_pipeline/            # Data processing
├── scripts/                  # Download scripts
└── kaggle/                   # Kaggle training runner
```

## License

MIT
