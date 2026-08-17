# INTERVUE — Running Training on Kaggle

This guide covers the **only** realistic way to train the heavy stages
(pretrain, domain, instruction) given the local GPU is a GTX 1650 with 4GB VRAM.

Local training is feasible only for stages with small data (interview/evaluator/followup
with `--limit` for smoke tests). Everything else: **use Kaggle**.

---

## 1. Prepare the data as a Kaggle Dataset

1. Upload your real data files to Kaggle as a Dataset (or use the API):

   ```bash
   kaggle datasets init -p data/raw
   # edit dataset-metadata.json (slug, title)
   kaggle datasets create -p data/raw
   ```

2. Files land in `/kaggle/input/<dataset-slug>/...`. The unified trainer
   (`models/generator/train.py`) scans `/kaggle/input` recursively, so it finds
   them automatically by filename.

---

## 2. Create the notebook

1. **New Notebook** → Settings → **Accelerator: GPU T4 x2** (or P100).
2. **Add Data** → your uploaded dataset from step 1.
3. In the first cell:

   ```python
   !git clone https://github.com/<YOUR_USER>/IntervAI.git /kaggle/working/IntervAI
   !python /kaggle/working/IntervAI/kaggle/train_on_kaggle.py --stage all
   ```

   (Or open `kaggle/train_on_kaggle.py` and run it directly in the notebook.)

---

## 3. What happens

- `env_config.py` detects `/kaggle/working` → **batch=16, accum=4, FP16, medium
  model (10.7M)**. No code changes needed vs. your laptop.
- Each stage saves an atomic checkpoint to
  `/kaggle/working/IntervAI/models/generator/saved/<stage>.pt`.
- Resuming: re-run the same stage script — it auto-loads the last checkpoint.
- Kaggle sessions stop after ~9h. To continue, re-run the same script; it resumes.

---

## 4. Downloading results

After training, download the checkpoints back to your laptop:

```python
from IPython.display import FileLink
FileLink('/kaggle/working/IntervAI/models/generator/saved/final_model.pt')
```

Or use the Kaggle API:
```bash
kaggle kernels output <your-kernel-slug> -p models/generator/saved
```

Copy them into `models/generator/saved/` locally.

---

## 5. Local smoke test (before any Kaggle run)

Verify the pipeline works end-to-end on your laptop first (CPU is fine for a few samples):

```bash
python models/generator/train.py --stage pretrain --limit 20
python models/generator/train.py --stage interview --limit 20
```

These use batch=1, no FP16, and finish in ~1 minute each.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No usable data found` | Data not mounted — check `/kaggle/input` contents, re-add the dataset |
| OOM on Kaggle | Shouldn't happen (batch=16 fits T4). Reduce via `GRAD_ACCUM_STEPS` in env_config if needed |
| Session timeout | Resume is automatic — just rerun the script |
| `Tokenizer not found` | Run the tokenizer cell first, or upload your locally-trained `tokenizer.json` |
| Repo clone fails | `github.com/<YOUR_USER>` — replace with your actual repo URL |
