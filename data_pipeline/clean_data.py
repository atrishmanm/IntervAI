"""
data_pipeline/clean_data.py
===========================
Processes the raw HuggingFace conversations into 3 question types:
  1. Theoretical — "What is the time/space complexity of X?"
  2. Output Prediction — "What does this code output for input Y?"
  3. Concept Explanation — "Explain the [approach] for [problem]"

Each question has a reference_answer for evaluation.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "conversations.jsonl"
OUT_DIR  = ROOT / "data" / "processed"

# Common algorithm/approach patterns to detect
APPROACHES = [
    "two pointers", "sliding window", "binary search", "bfs", "dfs",
    "breadth-first", "depth-first", "dynamic programming", "dp",
    "greedy", "backtracking", "divide and conquer", "hash map",
    "hash set", "hash table", "stack", "queue", "heap", "priority queue",
    "linked list", "tree", "graph", "trie", "recursion", "memoization",
    "topological sort", "union find", "disjoint set", "monotonic stack",
    "prefix sum", "bit manipulation", "sorting", "merge sort",
    "quick sort", "kadane", "floyd", "dijkstra", "bellman-ford",
]


def extract_title_and_difficulty(human_text: str):
    """Extract problem title and difficulty from the first human message."""
    # Pattern: **Title** (Difficulty)
    m = re.match(r"\*\*(.+?)\*\*\s*\((\w+)\)", human_text.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip().lower()
    # Fallback: just grab first line
    first_line = human_text.strip().split("\n")[0]
    title = re.sub(r"[*#`]", "", first_line).strip()[:80]
    return title if title else "Coding Problem", "medium"


def extract_code_blocks(text: str) -> list:
    """Extract all fenced code blocks from text."""
    pattern = r"```(?:python|java|cpp|c\+\+|javascript|js)?\s*\n(.*?)```"
    blocks = re.findall(pattern, text, re.DOTALL)
    return [b.strip() for b in blocks if len(b.strip()) > 20]


def extract_complexity(text: str) -> dict:
    """Extract time and space complexity from text."""
    result = {"time": None, "space": None}
    
    # Time complexity
    time_patterns = [
        r"[Tt]ime[:\s]*O\(([^)]+)\)",
        r"[Tt]ime\s*(?:complexity)?[:\s]*O\(([^)]+)\)",
    ]
    for p in time_patterns:
        m = re.search(p, text)
        if m:
            result["time"] = f"O({m.group(1)})"
            break
    
    # Space complexity
    space_patterns = [
        r"[Ss]pace[:\s]*O\(([^)]+)\)",
        r"[Ss]pace\s*(?:complexity)?[:\s]*O\(([^)]+)\)",
    ]
    for p in space_patterns:
        m = re.search(p, text)
        if m:
            result["space"] = f"O({m.group(1)})"
            break
    
    return result


def detect_approach(text: str) -> str:
    """Detect the algorithm approach from text."""
    text_lower = text.lower()
    for approach in APPROACHES:
        if approach in text_lower:
            return approach.title()
    return "General"


def extract_explanation(gpt_text: str) -> str:
    """Extract the explanation part (non-code) from the GPT response."""
    # Remove code blocks
    text = re.sub(r"```.*?```", "", gpt_text, flags=re.DOTALL)
    # Remove walkthrough blocks
    text = re.sub(r"(?:Walk-through|Example|Walk through).*?(?=\*\*|\Z)", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Clean up
    text = re.sub(r"\*\*.*?\*\*", "", text)  # Remove bold headers
    text = re.sub(r"\s+", " ", text).strip()
    # Take first 500 chars of explanation
    return text[:500] if text else ""


def generate_questions(conv_id: str, human_text: str, gpt_text: str) -> list:
    """Generate multiple question types from a single conversation."""
    questions = []
    
    title, difficulty = extract_title_and_difficulty(human_text)
    complexity = extract_complexity(gpt_text)
    approach = detect_approach(gpt_text)
    code_blocks = extract_code_blocks(gpt_text)
    explanation = extract_explanation(gpt_text)
    
    # ─── Type 1: Theoretical (Time Complexity) ───
    if complexity["time"]:
        questions.append({
            "id": f"{conv_id}_tc",
            "type": "theoretical",
            "topic": approach,
            "difficulty": difficulty,
            "question": f"What is the optimal time complexity for solving '{title}'?",
            "reference_answer": f"The optimal time complexity is {complexity['time']}.",
            "key_phrases": [complexity["time"]],
            "problem_title": title,
            "original_explanation": explanation,
        })
    
    # ─── Type 1b: Theoretical (Space Complexity) ───
    if complexity["space"]:
        questions.append({
            "id": f"{conv_id}_sc",
            "type": "theoretical",
            "topic": approach,
            "difficulty": difficulty,
            "question": f"What is the space complexity of the optimal solution for '{title}'?",
            "reference_answer": f"The space complexity is {complexity['space']}.",
            "key_phrases": [complexity["space"]],
            "problem_title": title,
            "original_explanation": explanation,
        })
    
    # ─── Type 2: Concept Explanation ───
    if approach != "General" and explanation:
        questions.append({
            "id": f"{conv_id}_concept",
            "type": "concept",
            "topic": approach,
            "difficulty": difficulty,
            "question": f"Explain the {approach} approach for solving '{title}'. Why does it work?",
            "reference_answer": explanation,
            "key_phrases": [approach.lower()],
            "problem_title": title,
            "original_explanation": explanation,
        })
    
    # ─── Type 3: Output Prediction ───
    if code_blocks:
        code = code_blocks[0]
        # Try to find example outputs in the GPT text
        example_pattern = r"#.*?(?:->|→|returns?|output)\s*(.+)"
        examples = re.findall(example_pattern, gpt_text)
        
        if examples:
            expected = examples[0].strip().rstrip(")")
            questions.append({
                "id": f"{conv_id}_output",
                "type": "output_prediction",
                "topic": approach,
                "difficulty": difficulty,
                "question": f"What does the following code return/output?\n\n```python\n{code}\n```",
                "reference_answer": expected,
                "key_phrases": [expected],
                "problem_title": title,
                "original_explanation": explanation,
            })
    
    return questions


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    q_path = OUT_DIR / "questions.jsonl"
    all_text_path = OUT_DIR / "all_explanations.jsonl"
    CRUX_PATH = ROOT / "data" / "raw" / "cruxeval" / "cruxeval.jsonl"

    if not RAW_PATH.exists():
        print(f"X Raw data not found at {RAW_PATH}")
        return

    questions_generated = 0
    explanations_saved = 0

    with open(RAW_PATH, "r", encoding="utf-8") as fin, \
         open(q_path, "w", encoding="utf-8") as fq, \
         open(all_text_path, "w", encoding="utf-8") as fa:
        
        for line_num, line in enumerate(fin):
            if not line.strip():
                continue
            record = json.loads(line)
            convs = record.get("conversations", [])
            
            if not convs or len(convs) < 2:
                continue
            
            # First human message = problem, first gpt message = solution
            human_msg = convs[0]
            gpt_msg = convs[1]
            
            if human_msg.get("from") != "human" or gpt_msg.get("from") != "gpt":
                continue
            
            human_text = human_msg.get("value", "")
            gpt_text = gpt_msg.get("value", "")
            
            if len(human_text) < 20 or len(gpt_text) < 50:
                continue
            
            conv_id = record.get("id", f"conv_{line_num}")
            
            # Generate questions
            qs = generate_questions(conv_id, human_text, gpt_text)
            for q in qs:
                fq.write(json.dumps(q, ensure_ascii=False) + "\n")
                questions_generated += 1
            
            # Save full explanation for TF-IDF corpus
            fa.write(json.dumps({
                "id": conv_id,
                "text": gpt_text,
                "title": extract_title_and_difficulty(human_text)[0],
            }, ensure_ascii=False) + "\n")
            explanations_saved += 1

    print(f"+ Generated {questions_generated:,} questions -> {q_path}")
    print(f"+ Saved {explanations_saved:,} explanations -> {all_text_path}")
    
    # ─── Process CRUXEval ───
    if CRUX_PATH.exists():
        crux_count = 0
        with open(CRUX_PATH, "r", encoding="utf-8") as fin, \
             open(q_path, "a", encoding="utf-8") as fq:
             for line in fin:
                 if not line.strip(): continue
                 record = json.loads(line)
                 code = record.get("code", "")
                 inp = record.get("input", "")
                 expected = record.get("output", "")
                 qid = record.get("id", f"crux_{crux_count}")
                 
                 q_text = f"What does the following code return/output?\n\n```python\n{code}\n```\n\nInput: `{inp}`"
                 
                 q = {
                     "id": f"crux_{qid}",
                     "type": "output_prediction",
                     "topic": "Python Execution",
                     "difficulty": "hard",
                     "question": q_text,
                     "reference_answer": expected,
                     "key_phrases": [expected],
                     "problem_title": "Code Trace",
                     "original_explanation": f"The correct output is {expected}",
                 }
                 fq.write(json.dumps(q, ensure_ascii=False) + "\n")
                 crux_count += 1
        print(f"+ Appended {crux_count:,} CRUXEval output_prediction questions")
    
    # Print type distribution
    type_counts = {}
    with open(q_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                t = r["type"]
                type_counts[t] = type_counts.get(t, 0) + 1
    print("\nQuestion type distribution:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c:,}")


if __name__ == "__main__":
    main()
