"""
data_pipeline/transform_training_data.py
=========================================
Converts raw datasets into structured training formats for the generator model.

Outputs:
  - instruction_training.jsonl   (Format A: ChatML instruction tuning)
  - evaluation_training.jsonl    (Format B: answer evaluation)
  - followup_training.jsonl      (Format C: follow-up generation)
  - concept_training.jsonl       (Format D: concept explanations)
  - interview_dialogues.jsonl    (Format E: interview dialogue)
  - pretrain_corpus.txt          (raw text for Stage 1 pretraining)
"""

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

SYSTEM_PROMPT = "You are an expert technical interviewer specializing in data structures, algorithms, and programming concepts. You ask insightful questions, evaluate answers thoroughly, and provide constructive feedback."

EVAL_SYSTEM_PROMPT = "You are evaluating a student's answer to a CS interview question. Provide a score (0-5), identify concepts covered and missing, and give constructive feedback."

FOLLOWUP_SYSTEM_PROMPT = "You are a technical interviewer. Based on the candidate's answer, generate an appropriate follow-up question that probes deeper or addresses gaps in their understanding."


def read_jsonl(path):
    """Read a JSONL file and yield records."""
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def format_chatml(messages):
    """Format a list of messages into ChatML format."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|{role}|> {content} <|end|>")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────
# Format A: Instruction Tuning
# ─────────────────────────────────────────────────────────────

def transform_standardlogic_to_instruction(out_file):
    """Convert StandardLogic conversations to instruction tuning format."""
    path = RAW_DIR / "conversations.jsonl"
    if not path.exists():
        return
    count = 0
    with open(out_file, "a", encoding="utf-8") as out:
        for rec in read_jsonl(path):
            convs = rec.get("conversations", [])
            if len(convs) < 2:
                continue
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for turn in convs:
                role = "user" if turn.get("from") == "human" else "assistant"
                content = turn.get("value", "").strip()
                if content:
                    messages.append({"role": role, "content": content})
            if len(messages) >= 3:
                out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                count += 1
    print(f"  StandardLogic -> instruction: {count} examples")


def transform_codealpaca_to_instruction(out_file):
    """Convert CodeAlpaca to instruction tuning format."""
    path = RAW_DIR / "codealpaca.jsonl"
    if not path.exists():
        return
    count = 0
    with open(out_file, "a", encoding="utf-8") as out:
        for rec in read_jsonl(path):
            instruction = rec.get("instruction", "").strip()
            inp = rec.get("input", "").strip()
            output = rec.get("output", "").strip()
            if not instruction or not output:
                continue
            user_msg = instruction
            if inp:
                user_msg += f"\n\nInput:\n{inp}"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": output},
            ]
            out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1
    print(f"  CodeAlpaca -> instruction: {count} examples")


def transform_codefeedback_to_instruction(out_file):
    """Convert CodeFeedback to instruction tuning format."""
    path = RAW_DIR / "codefeedback.jsonl"
    if not path.exists():
        return
    count = 0
    with open(out_file, "a", encoding="utf-8") as out:
        for rec in read_jsonl(path):
            instruction = rec.get("instruction", "").strip()
            response = rec.get("response", "").strip()
            if not instruction or not response:
                continue
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]
            out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1
    print(f"  CodeFeedback -> instruction: {count} examples")


# ─────────────────────────────────────────────────────────────
# Format B: Answer Evaluation
# ─────────────────────────────────────────────────────────────

def score_to_label(score):
    """Convert numeric score (0-5) to grade label."""
    if score >= 4.5:
        return "correct"
    elif score >= 3.0:
        return "partially_correct"
    elif score >= 1.0:
        return "incorrect"
    else:
        return "off_topic"


def generate_feedback(score, question, student_answer, reference):
    """Generate feedback text based on score and content."""
    label = score_to_label(score)
    if label == "correct":
        return (
            f"Score: {score}/5 — Excellent answer! "
            f"You correctly addressed the key concepts. "
            f"Your understanding of this topic is strong."
        )
    elif label == "partially_correct":
        return (
            f"Score: {score}/5 — Partially correct. "
            f"You captured some key ideas but missed important details. "
            f"Review the reference answer to identify what you overlooked."
        )
    elif label == "incorrect":
        return (
            f"Score: {score}/5 — Needs improvement. "
            f"Your answer has significant gaps. "
            f"Study the reference answer carefully to understand the correct approach."
        )
    else:
        return (
            f"Score: {score}/5 — Off topic. "
            f"Your answer does not address the question asked. "
            f"Please review the material and try again."
        )


def transform_mohler_to_evaluation(out_file):
    """Convert Mohler ASAG to answer evaluation format."""
    path = RAW_DIR / "mohler_asag.jsonl"
    if not path.exists():
        return
    count = 0
    with open(out_file, "a", encoding="utf-8") as out:
        for rec in read_jsonl(path):
            question = rec.get("question", "").strip()
            student = rec.get("student_answer", "").strip()
            reference = rec.get("instructor_answer", "").strip()
            score = rec.get("score_avg", 0)
            if not question or not student or not reference:
                continue
            grade = score_to_label(score)
            feedback = generate_feedback(score, question, student, reference)
            user_msg = (
                f"Question: {question}\n"
                f"Student Answer: {student}\n"
                f"Reference Answer: {reference}"
            )
            assistant_msg = (
                f"Score: {score}/5 ({grade})\n\n"
                f"Feedback: {feedback}"
            )
            messages = [
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
            out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1
    print(f"  Mohler -> evaluation: {count} examples")


def generate_synthetic_evaluation_examples(out_file):
    """Generate synthetic evaluation training examples from the question bank."""
    import sqlite3
    db_path = ROOT / "data" / "question_bank.db"
    if not db_path.exists():
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM questions LIMIT 10000").fetchall()
    conn.close()

    count = 0
    with open(out_file, "a", encoding="utf-8") as out:
        for row in rows:
            q = dict(row)
            question = q.get("question", "")
            reference = q.get("reference_answer", "")
            if not question or not reference:
                continue

            # Correct answer (score 5)
            messages = [
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}\nStudent Answer: {reference}\nReference Answer: {reference}"},
                {"role": "assistant", "content": f"Score: 5/5 (correct)\n\nFeedback: Excellent answer! You covered all the key concepts correctly."},
            ]
            out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1

            # Partial answer (score 3) — first half of reference
            words = reference.split()
            partial = " ".join(words[: len(words) // 2]) if len(words) > 5 else reference
            messages = [
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}\nStudent Answer: {partial}\nReference Answer: {reference}"},
                {"role": "assistant", "content": f"Score: 3/5 (partially_correct)\n\nFeedback: Partially correct. You captured some key ideas but missed important details. Review the reference answer to see what you overlooked."},
            ]
            out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1

            # Incorrect answer (score 1) — unrelated reference
            messages = [
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}\nStudent Answer: I think it has something to do with arrays and loops maybe.\nReference Answer: {reference}"},
                {"role": "assistant", "content": f"Score: 1/5 (incorrect)\n\nFeedback: Needs significant improvement. Your answer does not demonstrate understanding of the core concept. Please study the reference material carefully."},
            ]
            out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Synthetic evaluation examples: {count}")


# ─────────────────────────────────────────────────────────────
# Format C: Follow-up Generation
# ─────────────────────────────────────────────────────────────

def transform_to_followup(out_file):
    """Generate follow-up training examples from question bank."""
    import sqlite3
    db_path = ROOT / "data" / "question_bank.db"
    if not db_path.exists():
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM questions").fetchall()
    conn.close()

    # Build topic -> followups mapping
    topic_followups = {}
    for row in rows:
        q = dict(row)
        topic = q.get("topic", "General")
        key_phrases = json.loads(q.get("key_phrases") or "[]")
        if topic not in topic_followups:
            topic_followups[topic] = []
        topic_followups[topic].append(key_phrases)

    count = 0
    with open(out_file, "a", encoding="utf-8") as out:
        for row in rows:
            q = dict(row)
            question = q.get("question", "")
            reference = q.get("reference_answer", "")
            topic = q.get("topic", "General")
            key_phrases = json.loads(q.get("key_phrases") or "[]")
            if not question or not reference:
                continue

            # Partial answer — generate follow-up
            user_msg = (
                f"Question: {question}\n"
                f"Candidate Answer: It's related to {topic.lower()} and involves some algorithms.\n"
                f"Concepts covered: general topic awareness\n"
                f"Concepts missing: {', '.join(key_phrases[:3]) if key_phrases else 'specific details'}"
            )
            followup = f"Good start — you identified the general area. Now, can you be more specific? For example, {key_phrases[0] if key_phrases else 'what are the key concepts involved'}?"

            messages = [
                {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": followup},
            ]
            out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1

            # Wrong answer — generate corrective follow-up
            messages = [
                {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}\nCandidate Answer: I'm not sure, maybe it uses a hash map?\nConcepts covered: none\nConcepts missing: {', '.join(key_phrases[:3]) if key_phrases else 'all key concepts'}"},
                {"role": "assistant", "content": f"That's not quite right. Let me help you think through this. The question is about {topic.lower()}. What do you know about {key_phrases[0] if key_phrases else 'the fundamental concepts'}?"},
            ]
            out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Follow-up examples: {count}")


# ─────────────────────────────────────────────────────────────
# Format D: Concept Explanation
# ─────────────────────────────────────────────────────────────

CS_CONCEPTS = {
    "arrays": "An array is a collection of elements stored at contiguous memory locations. Arrays provide O(1) random access by index but have fixed size. Operations: access O(1), search O(n), insert O(n) for unsorted.",
    "linked_lists": "A linked list is a linear data structure where elements are stored in nodes. Each node contains data and a pointer to the next node. Provides O(1) insertion/deletion but O(n) access.",
    "stacks": "A stack is a LIFO (Last In, First Out) data structure. Core operations: push (add to top), pop (remove from top), peek (view top). Used in function calls, expression evaluation, undo operations.",
    "queues": "A queue is a FIFO (First In, First Out) data structure. Core operations: enqueue (add to rear), dequeue (remove from front). Variants: priority queue, deque, circular queue.",
    "hash_tables": "A hash table maps keys to values using a hash function to compute an index. Provides average O(1) lookup/insert. Collision handling: chaining (linked lists at each bucket) or open addressing (probe for next slot).",
    "binary_search": "Binary search finds an element in a sorted array by repeatedly dividing the search interval in half. Compare target with mid element, discard half. Requires sorted input. Time: O(log n), Space: O(1).",
    "sorting_basics": "Common sorting algorithms: Bubble Sort O(n^2), Selection Sort O(n^2), Insertion Sort O(n^2), Merge Sort O(n log n), Quick Sort O(n log n) average, Heap Sort O(n log n).",
    "trees": "A tree is a hierarchical data structure with nodes. Binary tree: each node has at most 2 children. BST: left < parent < right. Traversals: in-order, pre-order, post-order, level-order (BFS).",
    "graphs": "A graph consists of vertices (nodes) and edges (connections). Types: directed/undirected, weighted/unweighted. Representations: adjacency matrix O(V^2), adjacency list O(V+E).",
    "graph_traversal": "BFS uses a queue, explores level by level, finds shortest path in unweighted graphs. DFS uses a stack/recursion, explores as deep as possible first. Both O(V+E).",
    "dynamic_programming": "DP solves problems by breaking into overlapping subproblems. Key properties: optimal substructure + overlapping subproblems. Approaches: top-down (memoization), bottom-up (tabulation).",
    "greedy_algorithms": "Greedy algorithms make locally optimal choices at each step. Works when problem has greedy choice property + optimal substructure. Examples: activity selection, Huffman coding, Dijkstra's.",
    "recursion": "Recursion is when a function calls itself with a smaller input. Base case stops recursion. Each call adds a stack frame. Can be converted to iteration with an explicit stack.",
    "time_complexity": "Big-O describes upper bound of growth rate. Common: O(1) constant, O(log n) logarithmic, O(n) linear, O(n log n) linearithmic, O(n^2) quadratic, O(2^n) exponential.",
    "space_complexity": "Space complexity measures memory usage relative to input size. Includes: input space, auxiliary space, stack space for recursion. O(1) auxiliary = constant extra space.",
    "two_pointers": "Two pointers technique uses two indices moving through a data structure. Used for: pair sum, palindrome check, removing duplicates, merging sorted arrays. Often O(n).",
    "sliding_window": "Sliding window maintains a subset of elements as a 'window' over the data. Used for: maximum/minimum subarray, substring problems, running averages. Often O(n).",
    "binary_trees": "A binary tree has at most 2 children per node. Complete: all levels filled except possibly last. Perfect: all internal nodes have 2 children. Height-balanced: left and right heights differ by at most 1.",
    "heaps": "A heap is a complete binary tree satisfying heap property: parent >= children (max-heap) or parent <= children (min-heap). Used for priority queues. Insert O(log n), extract O(log n).",
    "tries": "A trie is a tree-like data structure for storing strings. Each node represents a character. Useful for: prefix search, autocomplete, spell checking. Search/insert O(m) where m = key length.",
}


def generate_concept_training(out_file):
    """Generate concept explanation training examples."""
    count = 0
    with open(out_file, "a", encoding="utf-8") as out:
        for concept, explanation in CS_CONCEPTS.items():
            concept_name = concept.replace("_", " ").title()
            # Direct explanation
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Explain {concept_name}."},
                {"role": "assistant", "content": explanation},
            ]
            out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1

            # With example request
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Can you give me an example of {concept_name}?"},
                {"role": "assistant", "content": f"Sure! {explanation}\n\nFor example, consider a practical use case and walk through how it works step by step."},
            ]
            out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1

            # Comparison question
            if concept == "stacks":
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "What is the difference between a stack and a queue?"},
                    {"role": "assistant", "content": "A stack follows LIFO (Last In, First Out) — the most recently added element is removed first. Think of a stack of plates. A queue follows FIFO (First In, First Out) — the earliest added element is removed first, like a line at a store. Stacks are used for function calls, undo operations, and expression parsing. Queues are used for BFS, task scheduling, and print queues."},
                ]
                out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                count += 1

    print(f"  Concept explanations: {count}")


# ─────────────────────────────────────────────────────────────
# Format E: Interview Dialogue
# ─────────────────────────────────────────────────────────────

def generate_interview_dialogues(out_file):
    """Generate interview dialogue training examples from question bank."""
    import sqlite3
    db_path = ROOT / "data" / "question_bank.db"
    if not db_path.exists():
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM questions").fetchall()
    conn.close()

    count = 0
    with open(out_file, "a", encoding="utf-8") as out:
        for row in rows:
            q = dict(row)
            question = q.get("question", "")
            reference = q.get("reference_answer", "")
            topic = q.get("topic", "General")
            if not question or not reference:
                continue

            # Dialogue 1: Good answer
            dialogue = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "assistant", "content": f"Let's discuss {topic.lower()}. {question}"},
                {"role": "user", "content": reference},
                {"role": "assistant", "content": "Excellent! That's a thorough answer. You covered the key concepts well. Let me ask a follow-up to test your understanding deeper."},
            ]
            out.write(json.dumps({"messages": dialogue}, ensure_ascii=False) + "\n")
            count += 1

            # Dialogue 2: Partial answer
            words = reference.split()
            partial = " ".join(words[: len(words) // 2]) if len(words) > 5 else "I think it's related to the topic."
            dialogue = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "assistant", "content": f"Now, {question}"},
                {"role": "user", "content": partial},
                {"role": "assistant", "content": "That's partially correct. You've identified some key aspects, but there are important details missing. Can you elaborate on the parts you're less sure about?"},
            ]
            out.write(json.dumps({"messages": dialogue}, ensure_ascii=False) + "\n")
            count += 1

            # Dialogue 3: Wrong answer
            dialogue = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "assistant", "content": f"Let's move on. {question}"},
                {"role": "user", "content": "I'm not sure about this one."},
                {"role": "assistant", "content": "That's okay — let me help you think through it. What do you know about this topic, even if you're not fully confident?"},
            ]
            out.write(json.dumps({"messages": dialogue}, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Interview dialogues: {count}")


# ─────────────────────────────────────────────────────────────
# Pretraining Corpus
# ─────────────────────────────────────────────────────────────

def build_pretrain_corpus(out_file):
    """Build raw text corpus for Stage 1 pretraining."""
    count = 0

    # From StandardLogic conversations
    conv_path = RAW_DIR / "conversations.jsonl"
    if conv_path.exists():
        with open(out_file, "a", encoding="utf-8") as out:
            for rec in read_jsonl(conv_path):
                for turn in rec.get("conversations", []):
                    text = turn.get("value", "").strip()
                    if text and len(text) > 20:
                        out.write(text + "\n")
                        count += 1

    # From CodeAlpaca
    alpaca_path = RAW_DIR / "codealpaca.jsonl"
    if alpaca_path.exists():
        with open(out_file, "a", encoding="utf-8") as out:
            for rec in read_jsonl(alpaca_path):
                for field in ["instruction", "input", "output"]:
                    text = rec.get(field, "").strip()
                    if text and len(text) > 20:
                        out.write(text + "\n")
                        count += 1

    # From CodeFeedback
    feedback_path = RAW_DIR / "codefeedback.jsonl"
    if feedback_path.exists():
        with open(out_file, "a", encoding="utf-8") as out:
            for rec in read_jsonl(feedback_path):
                for field in ["instruction", "response"]:
                    text = rec.get(field, "").strip()
                    if text and len(text) > 20:
                        out.write(text + "\n")
                        count += 1

    # From StarCoder samples
    star_path = RAW_DIR / "starcoder_sample.jsonl"
    if star_path.exists():
        with open(out_file, "a", encoding="utf-8") as out:
            for rec in read_jsonl(star_path):
                content = rec.get("content", "").strip()
                if content:
                    out.write(content + "\n")
                    count += 1

    print(f"  Pretrain corpus: {count} lines")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("INTERVUE — Data Transformation Pipeline")
    print("=" * 60)

    instruction_file = PROCESSED_DIR / "instruction_training.jsonl"
    evaluation_file = PROCESSED_DIR / "evaluation_training.jsonl"
    followup_file = PROCESSED_DIR / "followup_training.jsonl"
    concept_file = PROCESSED_DIR / "concept_training.jsonl"
    dialogue_file = PROCESSED_DIR / "interview_dialogues.jsonl"
    pretrain_file = PROCESSED_DIR / "pretrain_corpus.txt"

    # Clear existing files
    for f in [instruction_file, evaluation_file, followup_file, concept_file, dialogue_file, pretrain_file]:
        if f.exists():
            f.unlink()

    print("\n[1/6] Instruction tuning data...")
    transform_standardlogic_to_instruction(instruction_file)
    transform_codealpaca_to_instruction(instruction_file)
    transform_codefeedback_to_instruction(instruction_file)

    print("\n[2/6] Evaluation training data...")
    transform_mohler_to_evaluation(evaluation_file)
    generate_synthetic_evaluation_examples(evaluation_file)

    print("\n[3/6] Follow-up training data...")
    transform_to_followup(followup_file)

    print("\n[4/6] Concept explanation data...")
    generate_concept_training(concept_file)

    print("\n[5/6] Interview dialogue data...")
    generate_interview_dialogues(dialogue_file)

    print("\n[6/6] Pretraining corpus...")
    build_pretrain_corpus(pretrain_file)

    # Summary
    print("\n" + "=" * 60)
    print("Transformation Summary:")
    print("=" * 60)
    for f in sorted(PROCESSED_DIR.glob("*.jsonl")):
        lines = sum(1 for _ in open(f, encoding="utf-8"))
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:40s} {lines:>8,} examples  ({size_mb:.1f} MB)")
    if pretrain_file.exists():
        lines = sum(1 for _ in open(pretrain_file, encoding="utf-8"))
        print(f"  {'pretrain_corpus.txt':40s} {lines:>8,} lines")
    print("=" * 60)


if __name__ == "__main__":
    main()
