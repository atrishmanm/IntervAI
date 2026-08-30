#!/usr/bin/env python3
"""
scripts/download_evaluator_data.py
===================================
Downloads REAL scoring/grading datasets from HuggingFace and converts to JSONL.
All datasets contain question + student_answer + score (no synthetic data).

Usage:
    python scripts/download_evaluator_data.py

Output (in data/raw/):
    mohlerasag_hf.jsonl   - Expanded Mohler ASAG (7.3K, CS short answers)
    scientsbank.jsonl      - SciEntsBank (11K, science short answers)
    beetle.jsonl           - Beetle (3K, electronics short answers)
    asap_aes.jsonl         - ASAP-AES essay scoring (13K essays)
    asap_sas.jsonl         - ASAP-SAS short answer scoring (3.4K)
"""

import json
import os
import sys
import csv
import io
import zipfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_jsonl(records, filename):
    path = DATA_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records -> {path}")
    return len(records)


# ─────────────────────────────────────────────────────────────
# 1. Mohler ASAG (expanded version from HuggingFace)
# ─────────────────────────────────────────────────────────────

def download_mohlerasag():
    """Download expanded Mohler ASAG from nkazi/MohlerASAG on HuggingFace."""
    print("\n[1/5] Mohler ASAG (expanded)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("nkazi/MohlerASAG", split="open_ended", trust_remote_code=True)
        records = []
        for row in ds:
            q = row.get("question", "")
            sa = row.get("student_answer", "")
            ia = row.get("instructor_answer", "")
            score = row.get("score_avg", 0.0)
            if sa and q:
                records.append({
                    "question": q.strip(),
                    "student_answer": sa.strip(),
                    "instructor_answer": ia.strip() if ia else "",
                    "score": float(score),
                    "max_score": 5.0,
                    "source": "mohler_asag_hf",
                })
        return save_jsonl(records, "mohlerasag_hf.jsonl")
    except Exception as e:
        print(f"  [WARN] datasets library failed ({e}), trying direct download...")
        # Fallback: use the existing local file if available
        local = DATA_DIR / "mohler_asag.jsonl"
        if local.exists():
            print(f"  Using existing local file: {local}")
            return 0
        print(f"  [ERROR] No Mohler data available")
        return 0


# ─────────────────────────────────────────────────────────────
# 2. SciEntsBank (science short answers, entailment labels)
# ─────────────────────────────────────────────────────────────

def download_scientsbank():
    """Download SciEntsBank from nkazi/SciEntsBank on HuggingFace."""
    print("\n[2/5] SciEntsBank...")
    try:
        from datasets import load_dataset
        ds = load_dataset("nkazi/SciEntsBank", trust_remote_code=True)
        records = []
        # Process train + test splits
        for split_name in ds.keys():
            for row in ds[split_name]:
                q = row.get("question", "") or row.get("sentence1", "")
                sa = row.get("answer", "") or row.get("sentence2", "")
                label = row.get("label", "") or row.get("gold_label", "")

                # Convert entailment labels to numeric scores
                label_map = {
                    "Contradictory": 0.0,
                    "Incorrect": 0.0,
                    "Partially_correct/incomplete": 2.5,
                    "Correct": 5.0,
                    # Also handle numeric labels
                }
                if isinstance(label, str):
                    score = label_map.get(label, 2.5)
                elif isinstance(label, (int, float)):
                    # 5-way: 0-4 -> 0-5 scale; 3-way/2-way: already small
                    if label >= 4:
                        score = 5.0
                    elif label >= 2:
                        score = 2.5
                    else:
                        score = 0.0
                else:
                    score = 2.5

                if sa and q:
                    records.append({
                        "question": q.strip(),
                        "student_answer": sa.strip(),
                        "instructor_answer": "",
                        "score": score,
                        "max_score": 5.0,
                        "source": f"scientsbank_{split_name}",
                    })
        return save_jsonl(records, "scientsbank.jsonl")
    except Exception as e:
        print(f"  [ERROR] SciEntsBank download failed: {e}")
        return 0


# ─────────────────────────────────────────────────────────────
# 3. Beetle (electronics short answers, entailment labels)
# ─────────────────────────────────────────────────────────────

def download_beetle():
    """Download Beetle from nkazi/Beetle on HuggingFace."""
    print("\n[3/5] Beetle...")
    try:
        from datasets import load_dataset
        ds = load_dataset("nkazi/Beetle", trust_remote_code=True)
        records = []
        for split_name in ds.keys():
            for row in ds[split_name]:
                q = row.get("question", "") or row.get("sentence1", "")
                sa = row.get("answer", "") or row.get("sentence2", "")
                label = row.get("label", "") or row.get("gold_label", "")

                label_map = {
                    "Contradictory": 0.0,
                    "Incorrect": 0.0,
                    "Partially_correct/incomplete": 2.5,
                    "Correct": 5.0,
                }
                if isinstance(label, str):
                    score = label_map.get(label, 2.5)
                elif isinstance(label, (int, float)):
                    if label >= 4:
                        score = 5.0
                    elif label >= 2:
                        score = 2.5
                    else:
                        score = 0.0
                else:
                    score = 2.5

                if sa and q:
                    records.append({
                        "question": q.strip(),
                        "student_answer": sa.strip(),
                        "instructor_answer": "",
                        "score": score,
                        "max_score": 5.0,
                        "source": f"beetle_{split_name}",
                    })
        return save_jsonl(records, "beetle.jsonl")
    except Exception as e:
        print(f"  [ERROR] Beetle download failed: {e}")
        return 0


# ─────────────────────────────────────────────────────────────
# 4. ASAP-AES (essay scoring)
# ─────────────────────────────────────────────────────────────

def download_asap_aes():
    """Download ASAP-AES from llm-aes/asap-8-original on HuggingFace."""
    print("\n[4/5] ASAP-AES essay scoring...")
    try:
        from datasets import load_dataset
        ds = load_dataset("llm-aes/asap-8-original", trust_remote_code=True)
        records = []
        for split_name in ds.keys():
            for row in ds[split_name]:
                essay = row.get("essay", "")
                prompt = row.get("prompt", "") or row.get("essay_set", "")
                score = row.get("score", 0) or row.get("domain1_score", 0)
                essay_id = row.get("essay_id", "")

                if essay:
                    records.append({
                        "question": str(prompt).strip() if prompt else "Write an essay.",
                        "student_answer": essay.strip(),
                        "instructor_answer": "",
                        "score": float(score),
                        "max_score": 60.0,  # ASAP-AES has variable max scores
                        "source": f"asap_aes_{split_name}",
                        "essay_id": essay_id,
                    })
        return save_jsonl(records, "asap_aes.jsonl")
    except Exception as e:
        print(f"  [ERROR] ASAP-AES download failed: {e}")
        return 0


# ─────────────────────────────────────────────────────────────
# 5. ASAP-SAS (short answer scoring)
# ─────────────────────────────────────────────────────────────

def download_asap_sas():
    """Download ASAP-SAS from HuggingFace or Kaggle."""
    print("\n[5/5] ASAP-SAS short answer scoring...")
    # Try HuggingFace first
    try:
        from datasets import load_dataset
        # Try common HF mirrors of ASAP-SAS
        for repo in ["TasfiaS/ASAP-SAS", "llm-aes/asap-sas"]:
            try:
                ds = load_dataset(repo, trust_remote_code=True)
                records = []
                for split_name in ds.keys():
                    for row in ds[split_name]:
                        q = row.get("question", "") or row.get("prompt", "")
                        sa = row.get("student_answer", "") or row.get("answer", "")
                        score = row.get("score", 0) or row.get("normalized_score", 0)

                        if sa:
                            records.append({
                                "question": q.strip() if q else "",
                                "student_answer": sa.strip(),
                                "instructor_answer": "",
                                "score": float(score),
                                "max_score": 3.0,
                                "source": f"asap_sas_{split_name}",
                            })
                if records:
                    return save_jsonl(records, "asap_sas.jsonl")
            except Exception:
                continue
    except Exception:
        pass

    # If HF fails, try downloading from Kaggle (requires kaggle CLI)
    print("  [INFO] ASAP-SAS not found on HF. Download from Kaggle manually:")
    print("         kaggle competitions download -c asap-sas -p data/raw/")
    print("         Then unzip and convert to JSONL.")
    return 0


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Evaluator Data Download (Real Scoring Datasets)")
    print("=" * 60)
    print(f"Output directory: {DATA_DIR}")

    ensure_data_dir()

    total = 0
    total += download_mohlerasag()
    total += download_scientsbank()
    total += download_beetle()
    total += download_asap_aes()
    total += download_asap_sas()

    print(f"\n{'=' * 60}")
    print(f"Total evaluator scoring examples: {total}")
    print(f"{'=' * 60}")

    if total == 0:
        print("\n[WARNING] No data downloaded. Check internet connection.")
        print("You may need to install the datasets library: pip install datasets")


if __name__ == "__main__":
    main()
