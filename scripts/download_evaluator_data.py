#!/usr/bin/env python3
"""
scripts/download_evaluator_data.py
==================================
Downloads REAL scoring/grading datasets from HuggingFace and converts to JSONL.

Usage:
    python scripts/download_evaluator_data.py

Output (in data/raw/):
    mohlerasag_hf.jsonl   - Expanded Mohler ASAG (2.3K, CS short answers)
    scientsbank.jsonl      - SciEntsBank (10.8K, science short answers)
    beetle.jsonl           - Beetle (5.2K, electronics short answers)
    asap_aes.jsonl         - ASAP-AES essay scoring (723 essays)
"""

import json
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


def download_mohlerasag():
    print("\n[1/4] Mohler ASAG (expanded)...")
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
        print(f"  [WARN] Failed: {e}")
        return 0


def download_scientsbank():
    print("\n[2/4] SciEntsBank...")
    try:
        from datasets import load_dataset
        ds = load_dataset("nkazi/SciEntsBank", trust_remote_code=True)
        records = []
        for split_name in ds.keys():
            for row in ds[split_name]:
                q = (row.get("question") or row.get("Sentence1") or
                     row.get("sentence1") or row.get("prompt") or "")
                sa = (row.get("answer") or row.get("Sentence2") or
                      row.get("sentence2") or row.get("student_answer") or "")
                label = (row.get("label") or row.get("gold_label") or
                         row.get("Gold_label") or "")
                label_map = {
                    "Contradictory": 0.0, "Incorrect": 0.0,
                    "Partially_correct/incomplete": 2.5, "Correct": 5.0,
                    "Entailment": 5.0, "Contradiction": 0.0, "Neutral": 2.5,
                }
                if isinstance(label, str):
                    score = label_map.get(label, 2.5)
                elif isinstance(label, (int, float)):
                    score = 5.0 if label >= 4 else (2.5 if label >= 2 else 0.0)
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


def download_beetle():
    print("\n[3/4] Beetle...")
    try:
        from datasets import load_dataset
        ds = load_dataset("nkazi/Beetle", trust_remote_code=True)
        records = []
        for split_name in ds.keys():
            for row in ds[split_name]:
                q = (row.get("question") or row.get("Sentence1") or
                     row.get("sentence1") or row.get("prompt") or "")
                sa = (row.get("answer") or row.get("Sentence2") or
                      row.get("sentence2") or row.get("student_answer") or "")
                label = (row.get("label") or row.get("gold_label") or
                         row.get("Gold_label") or "")
                label_map = {
                    "Contradictory": 0.0, "Incorrect": 0.0,
                    "Partially_correct/incomplete": 2.5, "Correct": 5.0,
                    "Entailment": 5.0, "Contradiction": 0.0, "Neutral": 2.5,
                }
                if isinstance(label, str):
                    score = label_map.get(label, 2.5)
                elif isinstance(label, (int, float)):
                    score = 5.0 if label >= 4 else (2.5 if label >= 2 else 0.0)
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


def download_asap_aes():
    print("\n[4/4] ASAP-AES essay scoring...")
    try:
        from datasets import load_dataset
        ds = load_dataset("llm-aes/asap-8-original", trust_remote_code=True)
        records = []
        for split_name in ds.keys():
            for row in ds[split_name]:
                essay = row.get("essay", "")
                prompt = row.get("prompt", "") or row.get("essay_set", "")
                score = row.get("score", 0) or row.get("domain1_score", 0)
                if essay:
                    records.append({
                        "question": str(prompt).strip() if prompt else "Write an essay.",
                        "student_answer": essay.strip(),
                        "instructor_answer": "",
                        "score": float(score),
                        "max_score": 60.0,
                        "source": f"asap_aes_{split_name}",
                    })
        return save_jsonl(records, "asap_aes.jsonl")
    except Exception as e:
        print(f"  [ERROR] ASAP-AES download failed: {e}")
        return 0


def main():
    print("=" * 60)
    print("Evaluator Data Download (Real Scoring Datasets)")
    print("=" * 60)
    ensure_data_dir()
    total = 0
    total += download_mohlerasag()
    total += download_scientsbank()
    total += download_beetle()
    total += download_asap_aes()
    print(f"\n{'=' * 60}")
    print(f"Total evaluator scoring examples: {total}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
