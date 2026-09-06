"""
scripts/download_new_datasets.py
================================
Downloads real datasets for new features:
1. Coding interview dataset (100K) - behavioral + technical + system design
2. Salary negotiation dataset (100K) - real negotiation conversations
3. Real resume dataset (54K) - for resume parsing training
4. Verified coding dataset (450K) - for technical questions
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


def download_interview_sft():
    """Download coding-interview-sft-100k (behavioral + technical + system design)."""
    print("\n[1/4] Coding Interview SFT 100K...")
    try:
        from datasets import load_dataset
        ds = load_dataset("stindardlogic/coding-interview-sft-100k", split="train", streaming=True)
        records = []
        for i, row in enumerate(ds):
            if i >= 100000:
                break
            conversations = row.get("conversations", [])
            if conversations and len(conversations) >= 2:
                # Convert ShareGPT format to our format
                messages = []
                for turn in conversations:
                    role = turn.get("from", turn.get("role", ""))
                    content = turn.get("value", turn.get("content", ""))
                    if role in ["human", "user", "gpt", "assistant", "system"]:
                        if role == "human":
                            role = "user"
                        elif role == "gpt":
                            role = "assistant"
                        messages.append({"role": role, "content": content})
                
                if messages:
                    records.append({
                        "messages": messages,
                        "source": "coding-interview-sft-100k",
                    })
            if (i + 1) % 20000 == 0:
                print(f"    Processed {i+1} examples, kept {len(records)}")
        return save_jsonl(records, "interview_sft_100k.jsonl")
    except Exception as e:
        print(f"  [ERROR] Interview SFT download failed: {e}")
        return 0


def download_negotiation():
    """Download negotiation-strategy-sft-100k (salary negotiation conversations)."""
    print("\n[2/4] Salary Negotiation SFT 100K...")
    try:
        from datasets import load_dataset
        ds = load_dataset("stindardlogic/negotiation-strategy-sft-100k", split="train", streaming=True)
        records = []
        for i, row in enumerate(ds):
            if i >= 50000:  # Sample 50K negotiation examples
                break
            conversations = row.get("conversations", [])
            category = row.get("category", "")
            
            if conversations and len(conversations) >= 2:
                messages = []
                for turn in conversations:
                    role = turn.get("from", turn.get("role", ""))
                    content = turn.get("value", turn.get("content", ""))
                    if role in ["human", "user", "gpt", "assistant", "system"]:
                        if role == "human":
                            role = "user"
                        elif role == "gpt":
                            role = "assistant"
                        messages.append({"role": role, "content": content})
                
                if messages:
                    records.append({
                        "messages": messages,
                        "category": category,
                        "source": "negotiation-strategy-sft-100k",
                    })
            if (i + 1) % 10000 == 0:
                print(f"    Processed {i+1} examples, kept {len(records)}")
        return save_jsonl(records, "negotiation_sft_100k.jsonl")
    except Exception as e:
        print(f"  [ERROR] Negotiation download failed: {e}")
        return 0


def download_resumes():
    """Download nscg/54k-resume (real resume dataset)."""
    print("\n[3/4] Real Resume Dataset 54K...")
    try:
        from datasets import load_dataset
        ds = load_dataset("nscg/54k-resume", split="train", streaming=True)
        records = []
        for i, row in enumerate(ds):
            if i >= 30000:  # Sample 30K resumes
                break
            
            # Extract resume text
            resume_parts = []
            if row.get("name"):
                resume_parts.append(f"Name: {row['name']}")
            if row.get("email"):
                resume_parts.append(f"Email: {row['email']}")
            if row.get("phone"):
                resume_parts.append(f"Phone: {row['phone']}")
            if row.get("location"):
                resume_parts.append(f"Location: {row['location']}")
            if row.get("summary"):
                resume_parts.append(f"Summary: {row['summary']}")
            if row.get("skills"):
                skills = row['skills']
                if isinstance(skills, list):
                    resume_parts.append(f"Skills: {', '.join(skills)}")
                else:
                    resume_parts.append(f"Skills: {skills}")
            if row.get("experience"):
                exp = row['experience']
                if isinstance(exp, list) and len(exp) > 0:
                    resume_parts.append("Experience:")
                    for e in exp[:5]:
                        if isinstance(e, dict):
                            resume_parts.append(f"  - {e.get('title', '')} at {e.get('company', '')}")
            if row.get("education"):
                edu = row['education']
                if isinstance(edu, list) and len(edu) > 0:
                    resume_parts.append("Education:")
                    for e in edu[:3]:
                        if isinstance(e, dict):
                            resume_parts.append(f"  - {e.get('degree', '')} from {e.get('institution', '')}")
            
            resume_text = "\n".join(resume_parts)
            
            if resume_text and len(resume_text) > 50:
                records.append({
                    "resume_text": resume_text,
                    "skills": row.get("skills", []),
                    "source": "nscg/54k-resume",
                })
            
            if (i + 1) % 10000 == 0:
                print(f"    Processed {i+1} examples, kept {len(records)}")
        return save_jsonl(records, "resumes_54k.jsonl")
    except Exception as e:
        print(f"  [ERROR] Resume download failed: {e}")
        return 0


def download_kodcode():
    """Download KodCode-V1-SFT-R1 (verified coding dataset)."""
    print("\n[4/4] KodCode Verified Coding Dataset...")
    try:
        from datasets import load_dataset
        # Try SFT version first
        ds = load_dataset("KodCode/KodCode-V1-SFT-R1", split="train", streaming=True)
        records = []
        for i, row in enumerate(ds):
            if i >= 50000:  # Sample 50K
                break
            
            messages = row.get("messages", [])
            if messages and len(messages) >= 2:
                # Filter for interview-relevant content
                content = " ".join([m.get("content", "") for m in messages])
                if any(kw in content.lower() for kw in ["interview", "leetcode", "algorithm", "data structure", "system design", "behavioral"]):
                    records.append({
                        "messages": messages,
                        "source": "KodCode-V1-SFT-R1",
                    })
            
            if (i + 1) % 20000 == 0:
                print(f"    Processed {i+1} examples, kept {len(records)}")
        return save_jsonl(records, "kodcode_verified.jsonl")
    except Exception as e:
        print(f"  [ERROR] KodCode download failed: {e}")
        return 0


def main():
    print("=" * 60)
    print("  Downloading Real Datasets for New Features")
    print("=" * 60)
    ensure_data_dir()
    total = 0
    total += download_interview_sft()
    total += download_negotiation()
    total += download_resumes()
    total += download_kodcode()
    print(f"\n{'=' * 60}")
    print(f"Total new examples: {total}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
