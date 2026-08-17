"""
analysis/report.py
===================
Panel-style final interview report.

After the interview ends, this assembles a human-readable, panel-style analysis:
  - Overall score + verdict
  - Per-concept mastery (mastered / developing / beginner / not started)
  - Top weaknesses with the specific question, answer, and where the gap was
  - The model/correct answer for each weak question
  - Concrete improvement suggestions
  - Difficulty trajectory over the session

Consumed by backend/main.py (/api/report) and the frontend.
"""


def _verdict_from_score(avg_score):
    if avg_score >= 85:
        return "Strong Candidate"
    if avg_score >= 70:
        return "Solid Candidate"
    if avg_score >= 55:
        return "Developing Candidate"
    if avg_score >= 40:
        return "Entry-Level Candidate"
    return "Needs Preparation"


def _concept_level(avg_score, n_attempts):
    """Map concept stats to panel language."""
    if n_attempts == 0:
        return "Not Started", "Not assessed"
    if avg_score >= 80:
        return "Mastered", f"avg {avg_score:.0f}% over {n_attempts} question(s)"
    if avg_score >= 60:
        return "Developing", f"avg {avg_score:.0f}% over {n_attempts} question(s)"
    return "Beginner", f"avg {avg_score:.0f}% over {n_attempts} question(s)"


def build_panel_report(candidate_state, concept_graph=None):
    """
    candidate_state: instance of orchestrator.candidate_state.CandidateState
                     (has .history, .concept_scores, .concept_attempts,
                      .difficulty_level, .question_count, .score_history)
    concept_graph: optional dict from data/processed/concept_graph.json

    Returns a dict report ready for JSON serialization + display.
    """
    history = getattr(candidate_state, "history", [])
    concept_scores = getattr(candidate_state, "concept_scores", {})
    concept_attempts = getattr(candidate_state, "concept_attempts", {})
    difficulty = getattr(candidate_state, "difficulty_level", 1.0)
    score_history = getattr(candidate_state, "score_history", [])

    # ── Overall ──
    answered = [h for h in history if h.get("score") is not None]
    if answered:
        raw_avg = sum(h["score"] for h in answered) / len(answered)
        # CandidateState scores are normalized 0-1; InterviewState stores 0-100.
        avg_score = raw_avg if raw_avg > 1.0 else raw_avg * 100.0
    else:
        avg_score = 0.0
    overall = {
        "overall_score": round(avg_score, 1),
        "verdict": _verdict_from_score(avg_score),
        "questions_asked": len(history),
        "questions_answered": len(answered),
        "max_difficulty_reached": difficulty,
    }

    # ── Per-concept mastery ──
    # concept_scores may map name -> float (0-1) or name -> ConceptScore object.
    concepts = []
    concept_items = []
    for concept, val in concept_scores.items():
        if hasattr(val, "score"):  # ConceptScore object
            concept_items.append((concept, val.score, val.questions_attempted))
        else:                      # plain float
            concept_items.append((concept, val, concept_attempts.get(concept, 0)))
    for concept, avg, n_attempts in sorted(concept_items, key=lambda x: -x[1]):
        level, detail = _concept_level(avg * 100, n_attempts)
        concepts.append({
            "concept": concept,
            "level": level,
            "score": round(avg * 100, 1),
            "attempts": n_attempts,
            "detail": detail,
        })

    # ── Top weaknesses (with the specific question + gap + model answer) ──
    weaknesses = []
    for h in history:
        raw = h.get("score", 1.0)
        score_100 = raw if raw > 1.0 else raw * 100.0  # normalize 0-1 → 0-100
        if score_100 < 60:
            weaknesses.append({
                "question": h.get("question", ""),
                "candidate_answer": h.get("candidate_answer", ""),
                "score": score_100,
                "missing_concepts": h.get("missing_concepts", []),
                "feedback": h.get("feedback", ""),
                "model_answer": h.get("reference_answer", "")
                or h.get("expert_text", "")
                or _no_reference_note(),
            })
    weaknesses.sort(key=lambda w: w["score"])
    weaknesses = weaknesses[:5]

    # ── Improvement suggestions ──
    suggestions = _build_suggestions(concepts, weaknesses)

    # ── Difficulty trajectory ──
    trajectory = {
        "per_question": score_history,
        "trend": _trend_label(score_history),
    }

    return {
        "overall": overall,
        "concepts": concepts,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
        "trajectory": trajectory,
    }


def _no_reference_note():
    return "_(No model answer recorded for this question.)_"


def _trend_label(scores):
    if len(scores) < 3:
        return "Too few questions to determine a trend"
    first = sum(scores[: len(scores) // 2]) / max(len(scores) // 2, 1)
    second = sum(scores[len(scores) // 2 :]) / max(len(scores) - len(scores) // 2, 1)
    if second >= first + 10:
        return "Improving"
    if second <= first - 10:
        return "Declining"
    return "Steady"


def _build_suggestions(concepts, weaknesses):
    """Concrete, panel-style improvement advice."""
    suggestions = []

    beginner = [c for c in concepts if c["level"] == "Beginner"]
    developing = [c for c in concepts if c["level"] == "Developing"]

    if beginner:
        names = ", ".join(c["concept"] for c in beginner[:3])
        suggestions.append(
            f"Review foundational topics first: {names}. These appeared as gaps and "
            "will block harder questions."
        )

    if developing:
        names = ", ".join(c["concept"] for c in developing[:3])
        suggestions.append(
            f"Solidify your understanding of {names}. You have partial understanding — "
            "practice with examples and edge cases to reach mastery."
        )

    for w in weaknesses:
        if w["missing_concepts"]:
            suggestions.append(
                f"For \"{w['question']}\", study: {', '.join(w['missing_concepts'][:3])}."
            )

    if weaknesses:
        suggestions.append(
            "Practice structured answering: (1) define the concept, (2) explain how "
            "it works, (3) state complexity, (4) give an example, (5) mention trade-offs."
        )

    if not suggestions:
        suggestions.append(
            "Strong performance across all topics. Consider attempting harder, "
            "system-design-style questions to challenge yourself further."
        )

    return suggestions[:6]
