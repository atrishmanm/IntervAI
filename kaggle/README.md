# INTERVUE - Kaggle Training Guide

## Quick Start (5 minutes)

### Step 1: Upload Dataset to Kaggle
1. Go to [kaggle.com](https://kaggle.com) → **Datasets** → **New Dataset**
2. Upload the folder containing all files from `data/raw/`
3. Name it `intervai-data`
4. Make sure it contains these files:
   - `starcoder_large.jsonl` (1.7GB)
   - `codesearchnet.jsonl` (187MB)
   - `codefeedback.jsonl` (120MB)
   - `opencodeinstruct.jsonl` (1.2GB)
   - `conversations.jsonl` (299MB)
   - `oasst_coding.jsonl` (23MB)
   - `codealpaca.jsonl` (7MB)
   - `cruxeval/cruxeval.jsonl` (181KB)
   - `mohler_asag.jsonl` (913KB)

### Step 2: Create Kaggle Notebook
1. Go to **Code** → **New Notebook**
2. Settings → **Accelerator** → **GPU T4 x2** (or P100)
3. Internet → **On** (needed for git clone)

### Step 3: Add Dataset
1. **Add Data** → search for `intervai-data`
2. Click **Add** (it will appear at `/kaggle/input/intervai-data/`)

### Step 4: Run Training
Paste this in a cell and run:

```python
!git clone https://github.com/atrishmanm/IntervAI.git /kaggle/working/IntervAI
!python /kaggle/working/IntervAI/kaggle/train_on_kaggle.py --stage all --time-budget 480
```

### Step 5: Download Checkpoints
After training completes:
1. Go to **Output** tab
2. Find `/kaggle/working/IntervAI/models/generator/saved/`
3. Download all `.pt` files

---

## Resume After Timeout

If training times out (9 hours max on Kaggle):

```python
!python /kaggle/working/IntervAI/kaggle/train_on_kaggle.py --stage all --resume
```

Checkpoints are saved after each stage, so you can resume from where it stopped.

---

## Train Single Stage

```python
# Train only the interview stage
!python /kaggle/working/IntervAI/kaggle/train_on_kaggle.py --stage interview

# Available stages: pretrain, domain, instruction, interview, evaluator, followup, resume_finetune, negotiation
```

---

## Time Budget

Default: 480 minutes (8 hours). Adjust if needed:

```python
!python /kaggle/working/IntervAI/kaggle/train_on_kaggle.py --stage all --time-budget 420  # 7 hours
```

---

## Dataset Requirements

All datasets must be REAL (no synthetic data):

| File | Source | Size | Stage |
|------|--------|------|-------|
| starcoder_large.jsonl | BigCode | 1.7GB | pretrain |
| codesearchnet.jsonl | Microsoft | 187MB | pretrain |
| codefeedback.jsonl | Various | 120MB | pretrain |
| cruxeval/cruxeval.jsonl | Facebook | 181KB | pretrain |
| opencodeinstruct.jsonl | HuggingFace | 1.2GB | domain, instruction, interview |
| oasst_coding.jsonl | OpenAssistant | 23MB | domain, interview, followup |
| codealpaca.jsonl | Various | 7MB | domain, instruction |
| conversations.jsonl | Various | 299MB | instruction, interview, followup, resume_finetune, negotiation |
| mohler_asag.jsonl | Mohler et al. | 913KB | evaluator |

---

## Training Stages

| Stage | Purpose | Data | Time Est. |
|-------|---------|------|-----------|
| pretrain | General code understanding | starcoder, codesearchnet, cruxeval, codefeedback | ~90 min |
| domain | CS-specific knowledge | opencodeinstruct, oasst_coding, codealpaca | ~60 min |
| instruction | Instruction following | opencodeinstruct, codealpaca, conversations | ~60 min |
| interview | Interview dialogue | conversations, opencodeinstruct, oasst_coding | ~60 min |
| evaluator | Answer evaluation | mohler_asag | ~30 min |
| followup | Follow-up questions | oasst_coding, conversations | ~45 min |
| resume_finetune | Resume-specific Q&A | conversations, opencodeinstruct | ~45 min |
| negotiation | Salary negotiation | conversations, opencodeinstruct | ~45 min |
| **Total** | | | **~7 hours** |

---

## Checkpoints

Checkpoints are saved to `/kaggle/working/IntervAI/models/generator/saved/`:

- `pretrained.pt` - After pretrain stage
- `domain_tuned.pt` - After domain stage
- `instruction_tuned.pt` - After instruction stage
- `interview_tuned.pt` - After interview stage
- `evaluator.pt` - After evaluator stage
- `final_model.pt` - After followup stage
- `resume_finetuned.pt` - After resume_finetune stage
- `negotiation_tuned.pt` - After negotiation stage (final)
- `training_progress.json` - Training progress tracking

---

## Troubleshooting

### CUDA Out of Memory
The script auto-selects model size based on GPU:
- 4GB VRAM → Small (13.6M params)
- 8GB VRAM → Medium (46.7M params)
- 12GB+ VRAM → Large (125.6M params)

If still OOM, set environment variable:
```python
os.environ["INTERVUE_MODEL"] = "medium"
```

### Training Fails on a Stage
The script continues to next stage even if one fails. Check `training_progress.json` for status.

### Time Runs Out
Use `--resume` flag to continue from last checkpoint:
```python
!python /kaggle/working/IntervAI/kaggle/train_on_kaggle.py --stage all --resume
```

### Dataset Not Found
Make sure your dataset is named `intervai-data` and added to the notebook.

---

## Using the Trained Model

After training, use the checkpoint with the interview engine:

```python
from models.generator.model import create_large_model, GeneratorConfig
from models.generator.train_utils import load_checkpoint

# Load model
config = GeneratorConfig()
model = create_large_model(config)
load_checkpoint(model, None, None, path="models/generator/saved/negotiation_tuned.pt")

# Use with interview engine
from orchestrator.mock_interview import MockInterviewSimulator
simulator = MockInterviewSimulator(model=model)
```
