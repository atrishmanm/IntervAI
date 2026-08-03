"""
backend/inference_service.py
=============================
Provides analyze_answer() for the backend using purely rule-based and 
keyword-based scoring. The expert_text is now provided directly from the DB.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def analyze_answer(
    student_answer: str,
    reference_answer: str,
    key_phrases: list,
    question_type: str,
    expert_text: str = "",
) -> dict:
    """
    Analyze a student's answer using keyword scoring.
    Returns a structured analysis report dict.
    """
    from analysis.scorer import score_answer

    report = score_answer(
        student_answer=student_answer,
        reference_answer=reference_answer,
        key_phrases=key_phrases,
        question_type=question_type,
        expert_text=expert_text,
        expert_score=1.0,  # 100% exact match since it's pulled from DB
    )

    return report.to_dict()


def models_ready() -> dict:
    """Return status of each component."""
    db_path = ROOT / "data" / "question_bank.db"
    return {
        "question_bank": db_path.exists(),
        "tfidf_index": True,  # Legacy key, no longer needed but kept for frontend compatibility
    }
