"""
orchestrator/state_machine.py
=============================
Interview orchestrator — manages session state and question flow.
Asks theoretical, output prediction, and concept explanation questions.
Tracks candidate state (concepts, difficulty) and produces a panel report
at session end.
"""

import json
import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "question_bank.db"

QUESTION_TYPES = ["theoretical", "output_prediction", "concept"]
MAX_TURNS_PER_SESSION = 10


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_random_question(exclude_ids: list = None, q_type: str = None, difficulty: str = None) -> Optional[dict]:
    """Pick a random question, optionally filtered by type and/or difficulty."""
    exclude_ids = exclude_ids or []
    conn = _get_conn()

    clauses = []
    params = []

    if q_type:
        clauses.append("type=?")
        params.append(q_type)
    if difficulty:
        clauses.append("difficulty=?")
        params.append(difficulty)
    if exclude_ids:
        placeholders = ",".join("?" * len(exclude_ids))
        clauses.append(f"id NOT IN ({placeholders})")
        params += exclude_ids

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    cur = conn.execute(
        f"SELECT * FROM questions{where} ORDER BY RANDOM() LIMIT 1", params
    )

    row = cur.fetchone()
    conn.close()
    if row:
        r = dict(row)
        r["key_phrases"] = json.loads(r.get("key_phrases") or "[]")
        return r
    return None


@dataclass
class InterviewState:
    session_id: str
    turn_count: int = 0
    score_total: int = 0
    questions_asked: int = 0
    asked_ids: list = field(default_factory=list)
    current_question: dict = field(default_factory=dict)
    history: list = field(default_factory=list)  # [{question, answer, report}]
    finished: bool = False
    report: dict = field(default_factory=dict)


class Orchestrator:
    def __init__(self):
        from orchestrator.candidate_state import CandidateStateManager
        self._candidates = CandidateStateManager()
        self._pipeline = None

    def _get_pipeline(self):
        """Lazy-load the inference pipeline."""
        if self._pipeline is None:
            try:
                from backend.inference_pipeline import create_pipeline
                self._pipeline = create_pipeline(use_models=True)
            except Exception as e:
                print(f"  WARN: Pipeline init failed ({e}), using question bank only")
                self._pipeline = create_pipeline(use_models=False)
        return self._pipeline

    def start_session(self, state: InterviewState) -> dict:
        """Start a session. Returns the first question."""
        from orchestrator.candidate_state import CandidateState
        # Register candidate state (adapts difficulty + concepts)
        cand = self._candidates.get_or_create(state.session_id)

        # Generate question using pipeline (model or bank)
        pipeline = self._get_pipeline()
        q_type = random.choice(QUESTION_TYPES)
        q = pipeline.generate_question(
            difficulty=cand.current_difficulty,
            question_type=q_type,
        )

        # Add id field for tracking
        q["id"] = q.get("id", f"gen_{state.session_id}_{state.questions_asked}")

        state.current_question = q
        state.asked_ids.append(q["id"])
        state.questions_asked = 1
        state.turn_count = 1

        return {
            "type": "question",
            "question_number": 1,
            "question_type": q["type"],
            "topic": q.get("topic", "General"),
            "difficulty": q.get("difficulty", "medium"),
            "question_text": q["question_text"],
            "source": q.get("source", "bank"),
        }

    def process_answer(self, state: InterviewState, student_answer: str) -> dict:
        """Process the student's answer. Returns analysis + next question (or session end)."""
        if state.finished:
            return {"type": "finished", "message": "Session already complete."}

        q = state.current_question
        if not q:
            return {"type": "error", "message": "No current question."}

        cand = self._candidates.get_or_create(state.session_id)

        # Evaluate answer using pipeline (evaluator model + semantic scorer)
        pipeline = self._get_pipeline()
        report = pipeline.evaluate_answer(
            question=q["question_text"],
            answer=student_answer,
            reference=q.get("reference_answer", ""),
            key_phrases=q.get("key_phrases", []),
        )

        state.score_total += report["score"]

        # Save to history
        state.history.append({
            "question": q["question_text"],
            "question_type": q["type"],
            "candidate_answer": student_answer,
            "score": report["score"],
            "reference_answer": q.get("reference_answer", ""),
            "feedback": report.get("feedback", ""),
            "concepts_covered": report.get("covered_concepts", []),
            "missing_concepts": report.get("missing_concepts", []),
            "model_used": report.get("model_used", "unknown"),
        })

        # Update candidate concept state
        cand.update_from_answer(
            concept=q.get("topic", "General"),
            score=report["score"] / 100.0,
            concepts_covered=report.get("covered_concepts", []),
            concepts_missing=report.get("missing_concepts", []),
        )

        state.turn_count += 1

        # Check if session is done
        if state.questions_asked >= MAX_TURNS_PER_SESSION:
            state.finished = True
            state.report = self._build_panel_report(state, cand)
            avg_score = state.score_total // max(state.questions_asked, 1)
            return {
                "type": "session_complete",
                "report": report,
                "panel_report": state.report,
                "summary": {
                    "total_questions": state.questions_asked,
                    "average_score": avg_score,
                    "verdict": "Excellent!" if avg_score >= 70 else "Good effort!" if avg_score >= 40 else "Keep practicing!",
                },
            }

        # Pick next question — try follow-up first, then fallback
        followup = pipeline.generate_followup(
            question=q["question_text"],
            answer=student_answer,
            evaluation=report,
        )

        if followup.get("question_text"):
            next_q = {
                "id": f"followup_{state.session_id}_{state.questions_asked}",
                "type": "followup",
                "topic": q.get("topic", "General"),
                "difficulty": q.get("difficulty", "medium"),
                "question_text": followup["question_text"],
                "key_phrases": [],
                "reference_answer": "",
                "source": "model",
            }
        else:
            next_q = self._pick_next_question(state, cand)

        if next_q is None:
            state.finished = True
            state.report = self._build_panel_report(state, cand)
            return {
                "type": "session_complete",
                "report": report,
                "panel_report": state.report,
                "summary": {
                    "total_questions": state.questions_asked,
                    "average_score": state.score_total // max(state.questions_asked, 1),
                    "verdict": "No more questions available!",
                },
            }

        state.current_question = next_q
        state.asked_ids.append(next_q["id"])
        state.questions_asked += 1

        return {
            "type": "analysis_and_next",
            "report": report,
            "next_question": {
                "question_number": state.questions_asked,
                "question_type": next_q["type"],
                "topic": next_q.get("topic", "General"),
                "difficulty": next_q.get("difficulty", "medium"),
                "question_text": next_q["question_text"],
                "source": next_q.get("source", "bank"),
            },
        }

    def _pick_next_question(self, state: InterviewState, cand) -> Optional[dict]:
        """Choose the next question using concept weaknesses + difficulty."""
        # Rotate question types for variety
        type_idx = state.questions_asked % len(QUESTION_TYPES)
        next_type = QUESTION_TYPES[type_idx]

        # Use pipeline for question generation
        pipeline = self._get_pipeline()
        suggested = cand.suggest_next_concept(self._candidates.concept_graph)
        q = pipeline.generate_question(
            topic=suggested,
            difficulty=cand.current_difficulty,
            question_type=next_type,
        )

        if q:
            q["id"] = q.get("id", f"gen_{state.session_id}_{state.questions_asked}")
            return q
        return None

    @staticmethod
    def _get_question_by_topic(topic, exclude_ids, difficulty):
        """Find an unasked question for a topic, preferring a difficulty."""
        conn = _get_conn()
        if exclude_ids:
            placeholders = ",".join("?" * len(exclude_ids))
            cur = conn.execute(
                f"SELECT * FROM questions WHERE topic=? AND id NOT IN ({placeholders}) ORDER BY RANDOM() LIMIT 1",
                [topic] + exclude_ids,
            )
        else:
            cur = conn.execute(
                "SELECT * FROM questions WHERE topic=? ORDER BY RANDOM() LIMIT 1",
                (topic,),
            )
        row = cur.fetchone()
        conn.close()
        if row:
            r = dict(row)
            r["key_phrases"] = json.loads(r.get("key_phrases") or "[]")
            return r
        return None

    @staticmethod
    def _build_panel_report(state: InterviewState, cand) -> dict:
        """Generate the panel-style final report."""
        try:
            from analysis.report import build_panel_report
            return build_panel_report(cand)
        except Exception as e:
            return {
                "error": f"Panel report generation failed: {e}",
                "overall": {"overall_score": state.score_total / max(state.questions_asked, 1),
                            "verdict": "N/A"},
            }


orchestrator = Orchestrator()
