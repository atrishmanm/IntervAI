"""
data_pipeline/build_question_bank.py
======================================
Builds the SQLite question bank from the extracted questions.
"""

import json
import random
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
DB_PATH       = ROOT / "data" / "question_bank.db"

SEED = 42

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS questions (
    id              TEXT PRIMARY KEY,
    type            TEXT,
    topic           TEXT,
    difficulty      TEXT,
    question        TEXT,
    reference_answer TEXT,
    key_phrases     TEXT,
    problem_title   TEXT,
    original_explanation TEXT
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_type       ON questions(type);",
    "CREATE INDEX IF NOT EXISTS idx_topic      ON questions(topic);",
    "CREATE INDEX IF NOT EXISTS idx_difficulty ON questions(difficulty);",
    "CREATE INDEX IF NOT EXISTS idx_type_diff  ON questions(type, difficulty);",
]


def main():
    print("=" * 60)
    print("AI Interview Prep -- Building Question Bank")
    print("=" * 60)

    q_path = PROCESSED_DIR / "questions.jsonl"
    if not q_path.exists():
        print(f"X {q_path} not found. Run clean_data.py first.")
        return

    questions = []
    with open(q_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    print(f"\n+ Loaded {len(questions):,} questions")

    # Build SQLite
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS questions")
    cur.execute(CREATE_TABLE)
    for idx_sql in CREATE_INDEXES:
        cur.execute(idx_sql)

    cur.executemany(
        """INSERT OR REPLACE INTO questions
           (id, type, topic, difficulty, question, reference_answer, key_phrases, problem_title, original_explanation)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        [
            (
                r["id"], r["type"], r.get("topic", "General"),
                r.get("difficulty", "medium"), r["question"],
                r["reference_answer"],
                json.dumps(r.get("key_phrases", [])),
                r.get("problem_title", ""),
                r.get("original_explanation", ""),
            )
            for r in questions
        ]
    )
    conn.commit()

    # Print stats
    cur.execute("SELECT type, COUNT(*) FROM questions GROUP BY type")
    print("\nQuestion bank stats:")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]:,}")

    cur.execute("SELECT COUNT(*) FROM questions")
    total = cur.fetchone()[0]
    print(f"\n  Total: {total:,} questions")
    
    conn.close()
    print(f"\n+ SQLite DB -> {DB_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
