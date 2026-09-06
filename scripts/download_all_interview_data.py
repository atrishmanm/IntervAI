"""
scripts/download_all_interview_data.py
======================================
Downloads ALL real interview datasets for comprehensive training.

Datasets:
  1. HR Interview Dataset (2.5M rows) - behavioral/HR questions
  2. Coding Interview Transcripts - real coding interviews
  3. Resume Dataset - real resumes for parsing
  4. Existing data (starcoder, codesearchnet, etc.)
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


def download_hr_interview():
    """Download HR/Behavioral interview dataset (2.5M rows)."""
    print("\n[1/4] HR Interview Dataset (behavioral/HR)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("Ankshi/hr-interview-dataset", split="train", streaming=True)
        records = []
        for i, row in enumerate(ds):
            if i >= 50000:  # Sample 50K for training
                break
            question = row.get("question", "")
            answer = row.get("answer", "")
            category = row.get("category", "")
            difficulty = row.get("difficulty", "")
            q_type = row.get("type", "")
            role = row.get("role", "")

            if question and answer:
                records.append({
                    "messages": [
                        {"role": "system", "content": f"You are an HR interviewer conducting a {q_type.lower()} interview for a {role} position. The question is {difficulty} difficulty."},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    "category": category,
                    "difficulty": difficulty,
                    "type": q_type,
                })
            if (i + 1) % 10000 == 0:
                print(f"    Processed {i+1} examples, kept {len(records)}")
        return save_jsonl(records, "hr_interview_real.jsonl")
    except Exception as e:
        print(f"  [ERROR] HR interview download failed: {e}")
        return 0


def download_coding_interviews():
    """Download real coding interview transcripts."""
    print("\n[2/4] Coding Interview Transcripts...")
    try:
        from datasets import load_dataset
        ds = load_dataset("iamanisin/coding_interview_transcripts", split="train", streaming=True)
        records = []
        for i, row in enumerate(ds):
            if i >= 20000:  # Sample 20K
                break
            # The dataset has conversation format
            conversation = row.get("conversation", "")
            if conversation and len(conversation) > 100:
                records.append({
                    "content": conversation,
                    "source": "coding_interview_transcript",
                })
            if (i + 1) % 5000 == 0:
                print(f"    Processed {i+1} examples, kept {len(records)}")
        return save_jsonl(records, "coding_interviews_real.jsonl")
    except Exception as e:
        print(f"  [ERROR] Coding interview download failed: {e}")
        return 0


def download_resumes():
    """Download real resume dataset for parsing."""
    print("\n[3/4] Resume Dataset (real resumes)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("datasetmaster/resumes", split="train", streaming=True)
        records = []
        for i, row in enumerate(ds):
            if i >= 10000:  # Sample 10K
                break
            # Extract resume text
            resume_data = {
                "name": row.get("name", ""),
                "email": row.get("email", ""),
                "phone": row.get("phone", ""),
                "location": row.get("location", ""),
                "summary": row.get("summary", ""),
                "skills": row.get("skills", []),
                "experience": row.get("experience", []),
                "education": row.get("education", []),
                "projects": row.get("projects", []),
            }
            # Convert to text for training
            resume_text = f"Name: {resume_data['name']}\n"
            if resume_data["summary"]:
                resume_text += f"Summary: {resume_data['summary']}\n"
            if resume_data["skills"]:
                resume_text += f"Skills: {', '.join(resume_data['skills'])}\n"
            if resume_data["experience"]:
                resume_text += "Experience:\n"
                for exp in resume_data["experience"][:3]:
                    if isinstance(exp, dict):
                        resume_text += f"  - {exp.get('title', '')} at {exp.get('company', '')}\n"
            if resume_data["projects"]:
                resume_text += "Projects:\n"
                for proj in resume_data["projects"][:3]:
                    if isinstance(proj, dict):
                        resume_text += f"  - {proj.get('name', '')}\n"

            records.append({
                "resume_text": resume_text,
                "skills": resume_data["skills"],
                "source": "real_resume",
            })
            if (i + 1) % 2000 == 0:
                print(f"    Processed {i+1} examples, kept {len(records)}")
        return save_jsonl(records, "resumes_real.jsonl")
    except Exception as e:
        print(f"  [ERROR] Resume download failed: {e}")
        return 0


def download_oasst():
    """Download OpenAssistant conversational dataset."""
    print("\n[4/4] OpenAssistant Conversations...")
    try:
        from datasets import load_dataset
        ds = load_dataset("OpenAssistant/oasst2", split="train", streaming=True)
        records = []
        for i, row in enumerate(ds):
            if i >= 30000:  # Sample 30K
                break
            text = row.get("text", "")
            role = row.get("role", "")
            if text and role == "assistant" and len(text) > 50:
                records.append({
                    "content": text,
                    "source": "oasst2",
                })
            if (i + 1) % 10000 == 0:
                print(f"    Processed {i+1} examples, kept {len(records)}")
        return save_jsonl(records, "oasst2_conversations.jsonl")
    except Exception as e:
        print(f"  [ERROR] OASST download failed: {e}")
        return 0


def main():
    print("=" * 60)
    print("  Downloading ALL Real Interview Datasets")
    print("=" * 60)
    ensure_data_dir()
    total = 0
    total += download_hr_interview()
    total += download_coding_interviews()
    total += download_resumes()
    total += download_oasst()
    print(f"\n{'=' * 60}")
    print(f"Total real interview examples: {total}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
