# INTERVUE - Architecture Context

## Project Overview
**INTERVUE** is a state-of-the-art AI interview simulator that conducts real interviews — parses resumes, asks personalized questions (technical, HR/behavioral, coding), adapts difficulty, evaluates on 5 dimensions, and gives structured HIRE/MAYBE/NO HIRE recommendations.

Not a chatbot, not a coding platform — a full interview simulator that ChatGPT/Claude cannot replicate.

## Key Differentiators
1. **Resume-Aware**: Questions come FROM the resume content, not random topics
2. **Adaptive Difficulty**: Automatically adjusts from easy → expert based on performance
3. **STAR Method Evaluation**: Structured behavioral interview scoring
4. **5-Dimension Scoring**: Technical, Behavioral, Communication, Problem-solving, Cultural fit
5. **Real-Time Analytics**: Live performance tracking with radar charts, predictions, trends
6. **Company-Specific Templates**: FAANG, Big Tech, Unicorn, Startup patterns
7. **Industry-Specific Modules**: Backend, Frontend, Data Science, DevOps tracks
8. **Salary Negotiation Practice**: Multi-round negotiation with feedback
9. **Multi-Session Tracking**: Historical improvement and goal tracking
10. **Comprehensive Reports**: Exportable in JSON, Markdown, or HTML formats

---

## Model Architecture (~125.6M params Large)
```
GeneratorConfig:
  vocab_size: 16000
  embed_dim: 768
  n_heads: 12
  n_layers: 16
  ff_dim: 2048
  max_len: 2048
  dropout: 0.1 (dynamic, decays to 0.02)
  label_smoothing: 0.05
  tie_weights: True
  gradient_checkpointing: True
```

## Training Stack
- **Muon optimizer**: 2x over AdamW, Newton-Schulz orthogonalization
- **Sequence packing**: PackedDataset, 2-3x throughput
- **torch.compile**: 20-30% JIT speedup
- **WSD schedule**: warmup → stable → cosine decay
- **EMA (0.999)**: Stabilizes validation metrics
- **Gradient checkpointing**: Fits large model on small GPUs
- **Dynamic dropout**: Decays from 0.1 → 0.02 over training

## 8 Training Stages (REAL DATA ONLY)
1. **pretrain**: General code (starcoder, codesearchnet, cruxeval, codefeedback)
2. **domain**: CS knowledge (opencodeinstruct, oasst_coding, codealpaca, kodcode_verified)
3. **instruction**: Instruction following (opencodeinstruct, codealpaca, conversations, interview_sft_100k)
4. **interview**: Interview dialogue (conversations, opencodeinstruct, oasst_coding, interview_sft_100k)
5. **evaluator**: Answer evaluation (mohler_asag)
6. **followup**: Follow-up questions (oasst_coding, conversations, interview_sft_100k)
7. **resume_finetune**: Resume-specific (resumes_54k, interview_sft_100k, conversations)
8. **negotiation**: Salary negotiation (negotiation_sft_100k, conversations)

**NO SYNTHETIC DATA** - All training uses real, genuine datasets.

---

## Kaggle Training

### Quick Start
```python
# Clone repo
!git clone https://github.com/atrishmanm/IntervAI.git /kaggle/working/IntervAI

# Run training (<8 hours)
!python /kaggle/working/IntervAI/kaggle/train_on_kaggle.py --stage all --time-budget 480
```

### Resume After Timeout
```python
!python /kaggle/working/IntervAI/kaggle/train_on_kaggle.py --stage all --resume
```

### Features
- Auto-clone repo
- Auto-copy dataset from /kaggle/input
- Auto-train tokenizer
- 8 stages with time budget (<8 hours)
- Checkpoint resume support
- Progress tracking (training_progress.json)
- Error handling (continues on stage failure)

---

## Test Results
| Test Suite | Tests | Status |
|------------|-------|--------|
| test_complete_system.py | 10/10 | ✅ ALL PASSING |
| test_sota_features.py | 4/4 | ✅ ALL PASSING |
| Kaggle script tests | 3/3 | ✅ ALL PASSING |
| **Total** | **17/17** | **✅ ALL PASSING** |

---

## User Directives
- Do NOT commit regularly — commit only when user says so
- Do NOT commit the opencode folder
- Use real data only — NO synthetic data
- Training must complete in <8 hours on Kaggle
- Must be state-of-the-art project, not a random side project
