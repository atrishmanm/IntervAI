"""
data_pipeline/download_more_data.py
====================================
Downloads additional data to reach 3GB+ total.
Targets: larger StarCoder sample, CodeSearchNet, OpenAssistant coding.
"""

import json
import sys
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"


def download_starcoder_large(max_lines=200_000):
    """Download a larger StarCoder sample with retry logic."""
    out_path = RAW_DIR / "starcoder_large.jsonl"
    if out_path.exists():
        existing = sum(1 for _ in open(out_path, encoding="utf-8"))
        if existing >= max_lines * 0.9:
            print(f"  [SKIP] StarCoder large already exists ({existing} records)")
            return

    print("[1/3] Downloading StarCoder large sample (streaming)...")
    try:
        ds = load_dataset("bigcode/starcoderdata", split="train", streaming=True)
    except Exception as e:
        print(f"  Error loading StarCoder: {e}")
        print("  Trying alternative: bigcode/the-stack-deduplicated...")
        try:
            ds = load_dataset(
                "bigcode/the-stack-deduplicated",
                data_dir="python",
                split="train",
                streaming=True,
            )
        except Exception as e2:
            print(f"  Error: {e2}")
            print("  Skipping StarCoder. Will use other data.")
            return

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in ds:
            if count >= max_lines:
                break
            content = item.get("content", "")
            if content and len(content) > 200 and len(content) < 50000:
                # Filter: only keep files that look like actual code
                if any(kw in content[:500] for kw in ["def ", "class ", "import ", "function ", "const ", "public "]):
                    record = {
                        "content": content[:15000],
                        "repo": item.get("repo", ""),
                        "path": item.get("path", ""),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                    if count % 10000 == 0:
                        print(f"    {count:,} samples...")

    print(f"  Saved {count} code samples to {out_path}")


def download_codesearchnet(max_per_lang=20_000):
    """Download CodeSearchNet for Java, Python, JavaScript, Go."""
    out_path = RAW_DIR / "codesearchnet.jsonl"
    if out_path.exists():
        print(f"  [SKIP] CodeSearchNet already exists")
        return

    print("[2/3] Downloading CodeSearchNet...")
    languages = ["python", "java", "javascript", "go"]
    count = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for lang in languages:
            print(f"    Loading {lang}...")
            try:
                ds = load_dataset("code_search_net", lang, split="train", streaming=True)
                lang_count = 0
                for item in ds:
                    if lang_count >= max_per_lang:
                        break
                    code = item.get("func_code_string", "")
                    doc = item.get("func_documentation_string", "")
                    name = item.get("func_name", "")
                    if code and len(code) > 50:
                        record = {
                            "language": lang,
                            "name": name,
                            "code": code[:10000],
                            "doc": doc[:2000] if doc else "",
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1
                        lang_count += 1
                print(f"    {lang}: {lang_count} samples")
            except Exception as e:
                print(f"    Error loading {lang}: {e}")
                continue

    print(f"  Total CodeSearchNet: {count} samples")


def download_oasst_coding(max_records=50_000):
    """Download OpenAssistant for coding conversations."""
    out_path = RAW_DIR / "oasst_coding.jsonl"
    if out_path.exists():
        print(f"  [SKIP] OASST coding already exists")
        return

    print("[3/3] Downloading OpenAssistant coding conversations...")
    try:
        ds = load_dataset("OpenAssistant/oasst2", split="train")
    except Exception as e:
        print(f"  Error: {e}")
        return

    coding_keywords = [
        "code", "programming", "python", "java", "function", "algorithm",
        "debug", "compile", "variable", "loop", "array", "class", "api",
        "database", "sql", "html", "css", "javascript", "typescript",
        "data structure", "binary", "sort", "search", "recursion",
    ]

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in ds:
            if count >= max_records:
                break
            text = item.get("text", "").lower()
            if any(kw in text for kw in coding_keywords):
                record = {
                    "text": item.get("text", ""),
                    "role": item.get("role", ""),
                    "lang": item.get("lang", ""),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    print(f"  Saved {count} coding conversations")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("INTERVUE — Additional Data Download")
    print("=" * 60)

    download_starcoder_large(max_lines=200_000)
    download_codesearchnet(max_per_lang=20_000)
    download_oasst_coding(max_records=50_000)

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary:")
    print("=" * 60)
    total = 0
    for f in sorted(RAW_DIR.rglob("*.jsonl")):
        lines = sum(1 for _ in open(f, encoding="utf-8"))
        size_mb = f.stat().st_size / (1024 * 1024)
        total += size_mb
        print(f"  {f.name:40s} {lines:>8,} records  ({size_mb:.1f} MB)")
    print(f"  {'TOTAL':40s} {'':>8s}  ({total:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
