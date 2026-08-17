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

    def start_session(self, state: InterviewState) -> dict:
        """Start a session. Returns the first question."""
        from orchestrator.candidate_state import CandidateState
        # Register candidate state (adapts difficulty + concepts)
        cand = self._candidates.get_or_create(state.session_id)

        # Pick a random type to start with
        q_type = random.choice(QUESTION_TYPES)
        q = get_random_question(q_type=q_type, difficulty=cand.current_difficulty)
        if q is None:
            q = get_random_question(q_type=q_type)
        if q is None:
            q = get_random_question()
        if q is None:
            return {
                "type": "error",
                "message": "Could not load questions. Please run the training pipeline first.",
            }

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
            "question_text": q["question"],
        }

    def process_answer(self, state: InterviewState, student_answer: str) -> dict:
        """Process the student's answer. Returns analysis + next question (or session end)."""
        if state.finished:
            return {"type": "finished", "message": "Session already complete."}

        q = state.current_question
        if not q:
            return {"type": "error", "message": "No current question."}

        cand = self._candidates.get_or_create(state.session_id)

        # Analyze the answer
        from backend.inference_service import analyze_answer
        report = analyze_answer(
            student_answer=student_answer,
            reference_answer=q.get("reference_answer", ""),
            key_phrases=q.get("key_phrases", []),
            question_type=q.get("type", "theoretical"),
            expert_text=q.get("original_explanation", ""),
        )

        state.score_total += report["score"]

        # Save to history (report format used by frontend + panel report)
        state.history.append({
            "question": q["question"],
            "question_type": q["type"],
            "candidate_answer": student_answer,
            "score": report["score"],
            "reference_answer": q.get("reference_answer", ""),
            "expert_text": q.get("original_explanation", ""),
            "missing_concepts": report.get("missing_concepts", []),
            "feedback": report.get("feedback", ""),
            "concepts_covered": report.get("covered_concepts", []),
        })

        # Update candidate concept state (normalized 0-1)
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

        # Pick next question — use candidate state to choose concept/difficulty
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
                "question_text": next_q["question"],
            },
        }

    def _pick_next_question(self, state: InterviewState, cand) -> Optional[dict]:
        """Choose the next question using concept weaknesses + difficulty."""
        # Rotate question types for variety
        type_idx = state.questions_asked % len(QUESTION_TYPES)
        next_type = QUESTION_TYPES[type_idx]

        # Prefer questions on weak concepts (concept graph suggestion)
        suggested = cand.suggest_next_concept(self._candidates.concept_graph)
        q = None
        if suggested:
            q = self._get_question_by_topic(suggested, state.asked_ids,
                                            cand.current_difficulty)
        if q is None:
            q = get_random_question(exclude_ids=state.asked_ids,
                                    q_type=next_type,
                                    difficulty=cand.current_difficulty)
        if q is None:
            q = get_random_question(exclude_ids=state.asked_ids, q_type=next_type)
        if q is None:
            q = get_random_question(exclude_ids=state.asked_ids)
        return q

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
