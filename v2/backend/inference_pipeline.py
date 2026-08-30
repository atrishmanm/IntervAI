"""
backend/inference_pipeline.py
==============================
Chains the 6 trained INTERVUE models for a complete interview flow:

  1. Generate question:    interview_tuned.pt
  2. Evaluate answer:      evaluator.pt
  3. Generate follow-up:   final_model.pt

Falls back to the SQLite question bank + semantic scorer if models aren't loaded.
"""

import json
import random
import sqlite3
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.model_service import get_model_service, ModelService

DB_PATH = ROOT / "data" / "question_bank.db"

QUESTION_TYPES = ["theoretical", "output_prediction", "concept"]


def _get_question_from_bank(exclude_ids=None, q_type=None) -> Optional[dict]:
    """Fallback: pick a random question from the SQLite bank."""
    if not DB_PATH.exists():
        return None
    exclude_ids = exclude_ids or []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    clauses, params = [], []
    if q_type:
        clauses.append("type=?")
        params.append(q_type)
    if exclude_ids:
        placeholders = ",".join("?" * len(exclude_ids))
        clauses.append(f"id NOT IN ({placeholders})")
        params += exclude_ids
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    row = conn.execute(
        f"SELECT * FROM questions{where} ORDER BY RANDOM() LIMIT 1", params
    ).fetchone()
    conn.close()
    if row:
        r = dict(row)
        r["key_phrases"] = json.loads(r.get("key_phrases") or "[]")
        return r
    return None


class InterviewPipeline:
    """End-to-end interview pipeline using trained models + question bank fallback."""

    def __init__(self, use_models: bool = True):
        self.use_models = use_models
        self._service = None
        if use_models:
            try:
                self._service = get_model_service()
                self._service.load_all()
            except Exception as e:
                print(f"  WARN: Model loading failed ({e}), using question bank fallback")
                self.use_models = False

    def generate_question(self, topic: str = None,
                          difficulty: str = "medium",
                          question_type: str = None,
                          context: str = None) -> dict:
        """Generate an interview question.

        Uses the question bank for reliable, curated questions.
        The domain_tuned.pt model can augment with code examples.
        """
        q_type = question_type or random.choice(QUESTION_TYPES)

        # Primary: question bank (curated, validated)
        q = _get_question_from_bank(q_type=q_type)
        if q:
            return {
                "source": "bank",
                "type": q["type"],
                "topic": q.get("topic", topic or "General"),
                "difficulty": q.get("difficulty", difficulty),
                "question_text": q["question"],
                "key_phrases": q.get("key_phrases", []),
                "reference_answer": q.get("reference_answer", ""),
            }

        # Fallback: generate from model
        if self.use_models and self._service.get_model("domain"):
            prompt_parts = [
                f"Generate a {q_type} coding interview question"
            ]
            if topic:
                prompt_parts.append(f"about {topic}")
            prompt_parts.append(f"at {difficulty} difficulty level.")

            prompt = " ".join(prompt_parts)
            try:
                generated = self._service.generate(
                    "domain", prompt,
                    max_new_tokens=150,
                    temperature=0.8,
                    top_p=0.9,
                )
                return {
                    "source": "model",
                    "type": q_type,
                    "topic": topic or "General",
                    "difficulty": difficulty,
                    "question_text": generated,
                    "key_phrases": [],
                }
            except Exception as e:
                print(f"  WARN: Model generation failed ({e})")

        # Final fallback
        return {
            "source": "fallback",
            "type": q_type,
            "topic": topic or "General",
            "difficulty": difficulty,
            "question_text": f"Explain a key concept in {topic or 'computer science'} at {difficulty} level.",
            "key_phrases": [],
        }

    def evaluate_answer(self, question: str, answer: str,
                        reference: str = "",
                        key_phrases: list = None) -> dict:
        """Evaluate a candidate's answer.

        Uses evaluator.pt + semantic scorer for comprehensive evaluation.
        """
        key_phrases = key_phrases or []

        # Model-based evaluation
        model_result = None
        if self.use_models and self._service.get_model("evaluator"):
            try:
                model_result = self._service.evaluate(question, answer)
            except Exception as e:
                print(f"  WARN: Model evaluation failed ({e})")

        # Semantic scorer evaluation
        semantic_result = None
        try:
            from analysis.semantic_scorer import score_answer_contextual
            semantic_result = score_answer_contextual(
                student_answer=answer,
                reference_answer=reference,
                key_phrases=key_phrases,
                question_type="theoretical",
            )
        except Exception as e:
            print(f"  WARN: Semantic scorer failed ({e})")

        # Merge results — prefer semantic scorer for detailed feedback,
        # model for a secondary score signal
        if semantic_result:
            result = semantic_result
            if model_result:
                # Blend scores: 70% semantic + 30% model
                blended = int(0.7 * semantic_result["score"] + 0.3 * model_result["score"])
                result["score"] = blended
                result["model_score"] = model_result["score"]
                result["model_feedback"] = model_result["feedback"]
        elif model_result:
            result = model_result
        else:
            result = {"score": 50, "feedback": "Evaluation unavailable", "missing_concepts": []}

        result["model_used"] = "evaluator+semantic" if (model_result and semantic_result) else (
            "evaluator" if model_result else "semantic"
        )
        return result

    def generate_followup(self, question: str, answer: str,
                          evaluation: dict) -> dict:
        """Generate a follow-up question based on the evaluation.

        Uses final_model.pt to create targeted follow-ups.
        """
        if self.use_models and self._service.get_model("followup"):
            missing = evaluation.get("missing_concepts", [])
            score = evaluation.get("score", 50)

            # Simple, direct prompt
            prompt_parts = []
            if missing:
                prompt_parts.append(f"Follow up on: {', '.join(missing[:2])}")
            else:
                prompt_parts.append("Ask a deeper follow-up question")
            prompt_parts.append(f"Score was {score}/100.")

            prompt = " ".join(prompt_parts)
            try:
                generated = self._service.generate(
                    "followup", prompt,
                    max_new_tokens=80,
                    temperature=0.7,
                    top_p=0.9,
                )
                if generated and len(generated) > 10:
                    return {
                        "source": "model",
                        "question_text": generated,
                        "type": "followup",
                    }
            except Exception as e:
                print(f"  WARN: Follow-up generation failed ({e})")

        # Fallback: no follow-up
        return {"source": "none", "question_text": "", "type": "followup"}


# Convenience
def create_pipeline(use_models: bool = True) -> InterviewPipeline:
    return InterviewPipeline(use_models=use_models)
