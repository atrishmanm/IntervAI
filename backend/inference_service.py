"""
backend/inference_service.py
=============================
Provides analyze_answer() for the backend.

Uses the CONTEXTUAL (semantic) scorer from analysis.semantic_scorer — this is the
"ChatGPT-like" evaluation: it judges understanding, not just keyword matches.
Falls back to the legacy keyword scorer only if the semantic scorer errors.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.semantic_scorer import score_answer_contextual


def analyze_answer(
    student_answer: str,
    reference_answer: str,
    key_phrases: list,
    question_type: str,
    expert_text: str = "",
) -> dict:
    """
    Analyze a student's answer contextually (semantic understanding,
    concept coverage, accuracy, completeness, quality).

    Returns a structured analysis report dict (JSON-serializable).
    """
    try:
        report = score_answer_contextual(
            student_answer=student_answer,
            reference_answer=reference_answer,
            key_phrases=key_phrases,
            question_type=question_type,
            expert_text=expert_text,
        )
        return report
    except Exception as e:
        # Fall back to the legacy keyword scorer so the pipeline never breaks.
        from analysis.scorer import score_answer

        report = score_answer(
            student_answer=student_answer,
            reference_answer=reference_answer,
            key_phrases=key_phrases,
            question_type=question_type,
            expert_text=expert_text,
            expert_score=1.0,
        )
        return report.to_dict()


def models_ready() -> dict:
    """Return status of each component."""
    db_path = ROOT / "data" / "question_bank.db"
    return {
        "question_bank": db_path.exists(),
        "tfidf_index": True,  # Legacy key, no longer needed but kept for frontend compatibility
    }
