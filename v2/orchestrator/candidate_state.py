"""
orchestrator/candidate_state.py
================================
Tracks candidate knowledge, weaknesses, and adaptation state
for intelligent question selection and difficulty adjustment.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ConceptScore:
    """Score for a single concept."""
    name: str
    score: float = 0.0  # 0.0 to 1.0
    questions_attempted: int = 0
    correct_count: int = 0

    @property
    def mastery_level(self) -> str:
        if self.score >= 0.8:
            return "mastered"
        elif self.score >= 0.5:
            return "developing"
        elif self.score >= 0.2:
            return "beginner"
        else:
            return "not_started"


@dataclass
class CandidateState:
    """Full state of a candidate during an interview session."""
    session_id: str
    candidate_id: str = ""
    concept_scores: dict = field(default_factory=dict)  # concept_name -> ConceptScore
    weaknesses: list = field(default_factory=list)
    strengths: list = field(default_factory=list)
    current_difficulty: str = "easy"
    questions_answered: int = 0
    total_score: int = 0
    history: list = field(default_factory=list)
    asked_concepts: list = field(default_factory=list)

    def __post_init__(self):
        if not self.candidate_id:
            self.candidate_id = self.session_id

    def update_from_answer(
        self,
        concept: str,
        score: float,
        concepts_covered: list = None,
        concepts_missing: list = None,
    ):
        """Update state based on answer evaluation.

        `score` is normalized as a fraction (0.0-1.0). To pass a 0-100 score
        (e.g. from the semantic scorer), use `score/100`.
        """
        concepts_covered = concepts_covered or []
        concepts_missing = concepts_missing or []

        # Update concept score with exponential moving average
        if concept not in self.concept_scores:
            self.concept_scores[concept] = ConceptScore(name=concept)

        cs = self.concept_scores[concept]
        cs.questions_attempted += 1
        normalized = min(max(score, 0.0), 1.0)
        # EMA: new_score = alpha * normalized + (1-alpha) * old_score
        alpha = 0.3
        cs.score = alpha * normalized + (1 - alpha) * cs.score
        if normalized >= 0.6:
            cs.correct_count += 1

        # Update all covered concepts (boost slightly)
        for c in concepts_covered:
            if c not in self.concept_scores:
                self.concept_scores[c] = ConceptScore(name=c)
            self.concept_scores[c].score = min(
                1.0, self.concept_scores[c].score + 0.05
            )

        # Update missing concepts (reduce slightly)
        for c in concepts_missing:
            if c not in self.concept_scores:
                self.concept_scores[c] = ConceptScore(name=c)
            self.concept_scores[c].score = max(
                0.0, self.concept_scores[c].score - 0.1
            )

        # Record in history
        self.history.append({
            "concept": concept,
            "score": score,
            "concepts_covered": concepts_covered,
            "concepts_missing": concepts_missing,
        })

        self.questions_answered += 1
        self.total_score += score
        self.asked_concepts.append(concept)

        # Recalculate weaknesses and strengths
        self._update_weaknesses_strengths()
        self._adjust_difficulty()

    def _update_weaknesses_strengths(self):
        """Recalculate weaknesses and strengths from concept scores."""
        scored = [(name, cs.score) for name, cs in self.concept_scores.items()]
        scored.sort(key=lambda x: x[1])

        self.weaknesses = [name for name, s in scored if s < 0.5]
        self.strengths = [name for name, s in scored if s >= 0.7]

    def _adjust_difficulty(self):
        """Adjust difficulty based on recent performance."""
        if len(self.history) < 2:
            return

        recent = self.history[-3:]
        avg_recent = sum(h["score"] for h in recent) / len(recent)

        if avg_recent >= 0.8:
            if self.current_difficulty == "easy":
                self.current_difficulty = "medium"
            elif self.current_difficulty == "medium":
                self.current_difficulty = "hard"
        elif avg_recent <= 0.4:
            if self.current_difficulty == "hard":
                self.current_difficulty = "medium"
            elif self.current_difficulty == "medium":
                self.current_difficulty = "easy"

    @property
    def average_score(self) -> float:
        if self.questions_answered == 0:
            return 0.0
        return self.total_score / self.questions_answered

    @property
    def score_history(self) -> list:
        """Recent per-question normalized scores (0-1), for report trends."""
        return [min(h["score"], 1.0) for h in self.history]

    @property
    def difficulty_level(self) -> float:
        """Numeric difficulty (1-3) for report trajectory display."""
        return {"easy": 1.0, "medium": 2.0, "hard": 3.0}.get(
            self.current_difficulty, 1.0
        )

    @property
    def question_count(self) -> int:
        return self.questions_answered

    @property
    def weakest_concept(self) -> Optional[str]:
        if not self.weaknesses:
            return None
        return self.weaknesses[0]

    @property
    def strongest_concept(self) -> Optional[str]:
        if not self.strengths:
            return None
        return self.strengths[-1]

    def get_prerequisite_gaps(self, concept: str, concept_graph: dict) -> list:
        """Return prerequisites of a concept that the candidate hasn't mastered."""
        if concept not in concept_graph:
            return []
        prereqs = concept_graph[concept].get("prerequisites", [])
        gaps = []
        for prereq in prereqs:
            score = self.concept_scores.get(prereq, ConceptScore(name=prereq))
            if score.score < 0.5:
                gaps.append(prereq)
        return gaps

    def suggest_next_concept(self, concept_graph: dict) -> Optional[str]:
        """Suggest the next concept to test based on weaknesses and prerequisites."""
        # First, fill prerequisite gaps
        for concept in list(self.weaknesses):
            gaps = self.get_prerequisite_gaps(concept, concept_graph)
            if gaps:
                # Test the weakest prerequisite first
                return min(gaps, key=lambda c: self.concept_scores.get(c, ConceptScore(name=c)).score)

        # Then, test weaknesses directly
        for concept in self.weaknesses:
            if concept not in self.asked_concepts[-3:]:  # Don't repeat recent
                return concept

        # If no weaknesses, test new concepts
        all_concepts = set(concept_graph.keys())
        untested = all_concepts - set(self.asked_concepts)
        if untested:
            return min(
                untested,
                key=lambda c: self.concept_scores.get(c, ConceptScore(name=c)).score,
            )

        # All tested, revisit weakest
        return self.weakest_concept

    def to_dict(self) -> dict:
        """Serialize state to dictionary."""
        return {
            "session_id": self.session_id,
            "candidate_id": self.candidate_id,
            "concept_scores": {
                name: {"score": cs.score, "attempts": cs.questions_attempted}
                for name, cs in self.concept_scores.items()
            },
            "weaknesses": self.weaknesses,
            "strengths": self.strengths,
            "current_difficulty": self.current_difficulty,
            "questions_answered": self.questions_answered,
            "total_score": self.total_score,
            "average_score": self.average_score,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CandidateState":
        """Deserialize state from dictionary."""
        state = cls(
            session_id=data["session_id"],
            candidate_id=data.get("candidate_id", ""),
            weaknesses=data.get("weaknesses", []),
            strengths=data.get("strengths", []),
            current_difficulty=data.get("current_difficulty", "easy"),
            questions_answered=data.get("questions_answered", 0),
            total_score=data.get("total_score", 0),
            history=data.get("history", []),
        )
        for name, cs_data in data.get("concept_scores", {}).items():
            state.concept_scores[name] = ConceptScore(
                name=name,
                score=cs_data.get("score", 0.0),
                questions_attempted=cs_data.get("attempts", 0),
            )
        return state


class CandidateStateManager:
    """Manages candidate states for all active sessions."""

    def __init__(self):
        self.states: dict[str, CandidateState] = {}
        self.concept_graph: dict = {}
        self._load_concept_graph()

    def _load_concept_graph(self):
        """Load the concept graph from disk."""
        graph_path = ROOT / "data" / "processed" / "concept_graph.json"
        if graph_path.exists():
            with open(graph_path, encoding="utf-8") as f:
                self.concept_graph = json.load(f)

    def get_or_create(self, session_id: str) -> CandidateState:
        """Get existing state or create new one."""
        if session_id not in self.states:
            self.states[session_id] = CandidateState(session_id=session_id)
        return self.states[session_id]

    def remove(self, session_id: str):
        """Remove a session state."""
        self.states.pop(session_id, None)

    def get_suggestion(self, session_id: str) -> dict:
        """Get a suggestion for what to test next."""
        state = self.get_or_create(session_id)
        concept = state.suggest_next_concept(self.concept_graph)
        return {
            "suggested_concept": concept,
            "difficulty": state.current_difficulty,
            "weaknesses": state.weaknesses,
            "strengths": state.strengths,
            "average_score": state.average_score,
        }


# Global instance
candidate_manager = CandidateStateManager()
