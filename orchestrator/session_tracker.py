"""
orchestrator/session_tracker.py
================================
Multi-session tracking and improvement analytics.

Features:
1. Persistent session storage (JSON-based)
2. Historical performance tracking
3. Improvement trend analysis
4. Skill progress over time
5. Weakness identification across sessions
6. Comparative analysis (session vs session)
7. Goal setting and tracking
8. Predictive analytics based on history
"""

import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta


@dataclass
class SessionRecord:
    """Record of a single interview session."""
    session_id: str
    timestamp: float
    duration_minutes: float
    overall_score: float
    recommendation: str
    dimension_scores: Dict[str, float]
    phase_scores: Dict[str, float]
    skills_assessed: List[str]
    strengths: List[str]
    weaknesses: List[str]
    company_template: Optional[str] = None
    industry_module: Optional[str] = None
    notes: str = ""


@dataclass
class SkillProgress:
    """Track progress of a specific skill over time."""
    skill_name: str
    scores: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    sessions: List[str] = field(default_factory=list)
    
    @property
    def latest_score(self) -> float:
        return self.scores[-1] if self.scores else 0.0
    
    @property
    def average_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0
    
    @property
    def trend(self) -> str:
        if len(self.scores) < 2:
            return "insufficient_data"
        recent = self.scores[-3:]
        earlier = self.scores[:-3] if len(self.scores) > 3 else self.scores[:1]
        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)
        if recent_avg > earlier_avg + 0.1:
            return "improving"
        elif recent_avg < earlier_avg - 0.1:
            return "declining"
        return "stable"
    
    @property
    def improvement_rate(self) -> float:
        if len(self.scores) < 2:
            return 0.0
        return (self.scores[-1] - self.scores[0]) / len(self.scores)


@dataclass
class UserGoal:
    """User-defined goal for improvement."""
    goal_id: str
    description: str
    target_score: float
    target_date: str
    current_score: float = 0.0
    progress: float = 0.0
    status: str = "active"  # active, achieved, expired


class SessionTracker:
    """Multi-session tracking and improvement analytics."""

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/sessions")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.sessions_file = self.data_dir / "sessions.json"
        self.skills_file = self.data_dir / "skills.json"
        self.goals_file = self.data_dir / "goals.json"
        
        self.sessions: List[SessionRecord] = []
        self.skill_progress: Dict[str, SkillProgress] = {}
        self.goals: List[UserGoal] = []
        
        self._load_data()

    def _load_data(self):
        """Load persisted data."""
        if self.sessions_file.exists():
            with open(self.sessions_file, "r") as f:
                data = json.load(f)
                self.sessions = [SessionRecord(**s) for s in data]
        
        if self.skills_file.exists():
            with open(self.skills_file, "r") as f:
                data = json.load(f)
                for skill_name, progress_data in data.items():
                    self.skill_progress[skill_name] = SkillProgress(**progress_data)
        
        if self.goals_file.exists():
            with open(self.goals_file, "r") as f:
                data = json.load(f)
                self.goals = [UserGoal(**g) for g in data]

    def _save_data(self):
        """Persist data to disk."""
        with open(self.sessions_file, "w") as f:
            json.dump([{
                "session_id": s.session_id,
                "timestamp": s.timestamp,
                "duration_minutes": s.duration_minutes,
                "overall_score": s.overall_score,
                "recommendation": s.recommendation,
                "dimension_scores": s.dimension_scores,
                "phase_scores": s.phase_scores,
                "skills_assessed": s.skills_assessed,
                "strengths": s.strengths,
                "weaknesses": s.weaknesses,
                "company_template": s.company_template,
                "industry_module": s.industry_module,
                "notes": s.notes,
            } for s in self.sessions], f, indent=2)
        
        with open(self.skills_file, "w") as f:
            json.dump({
                name: {
                    "skill_name": p.skill_name,
                    "scores": p.scores,
                    "timestamps": p.timestamps,
                    "sessions": p.sessions,
                }
                for name, p in self.skill_progress.items()
            }, f, indent=2)
        
        with open(self.goals_file, "w") as f:
            json.dump([{
                "goal_id": g.goal_id,
                "description": g.description,
                "target_score": g.target_score,
                "target_date": g.target_date,
                "current_score": g.current_score,
                "progress": g.progress,
                "status": g.status,
            } for g in self.goals], f, indent=2)

    def record_session(self, session_data: Dict) -> str:
        """Record a completed interview session."""
        import uuid
        
        session_id = str(uuid.uuid4())[:8]
        
        record = SessionRecord(
            session_id=session_id,
            timestamp=time.time(),
            duration_minutes=session_data.get("duration_minutes", 0),
            overall_score=session_data.get("overall_score", 0),
            recommendation=session_data.get("recommendation", "NO HIRE"),
            dimension_scores=session_data.get("dimension_scores", {}),
            phase_scores=session_data.get("phase_scores", {}),
            skills_assessed=session_data.get("skills_assessed", []),
            strengths=session_data.get("strengths", []),
            weaknesses=session_data.get("weaknesses", []),
            company_template=session_data.get("company_template"),
            industry_module=session_data.get("industry_module"),
            notes=session_data.get("notes", ""),
        )
        
        self.sessions.append(record)
        
        # Update skill progress
        for skill in record.skills_assessed:
            if skill not in self.skill_progress:
                self.skill_progress[skill] = SkillProgress(skill_name=skill)
            self.skill_progress[skill].scores.append(record.overall_score)
            self.skill_progress[skill].timestamps.append(record.timestamp)
            self.skill_progress[skill].sessions.append(session_id)
        
        # Update goals
        self._update_goals(record)
        
        self._save_data()
        return session_id

    def get_session_history(self, limit: int = 10) -> List[Dict]:
        """Get recent session history."""
        sorted_sessions = sorted(self.sessions, key=lambda s: s.timestamp, reverse=True)
        return [{
            "session_id": s.session_id,
            "timestamp": s.timestamp,
            "date": datetime.fromtimestamp(s.timestamp).strftime("%Y-%m-%d %H:%M"),
            "duration_minutes": s.duration_minutes,
            "overall_score": s.overall_score,
            "recommendation": s.recommendation,
            "company_template": s.company_template,
            "industry_module": s.industry_module,
        } for s in sorted_sessions[:limit]]

    def get_improvement_trends(self) -> Dict:
        """Analyze improvement trends across sessions."""
        if len(self.sessions) < 2:
            return {
                "status": "insufficient_data",
                "message": "Need at least 2 sessions for trend analysis",
            }
        
        sorted_sessions = sorted(self.sessions, key=lambda s: s.timestamp)
        scores = [s.overall_score for s in sorted_sessions]
        timestamps = [s.timestamp for s in sorted_sessions]
        
        # Split into first half and second half for comparison
        mid = len(scores) // 2
        first_half = scores[:mid] if mid > 0 else scores[:1]
        second_half = scores[mid:] if mid < len(scores) else scores[-1:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        # Session-over-session change
        session_changes = []
        for i in range(1, len(scores)):
            session_changes.append(scores[i] - scores[i-1])
        
        # Determine overall trend
        if second_avg > first_avg + 0.05:
            overall_trend = "improving"
        elif second_avg < first_avg - 0.05:
            overall_trend = "declining"
        else:
            overall_trend = "stable"
        
        return {
            "total_sessions": len(self.sessions),
            "overall_trend": overall_trend,
            "first_session_score": scores[0],
            "latest_session_score": scores[-1],
            "improvement": scores[-1] - scores[0],
            "average_score": sum(scores) / len(scores),
            "best_score": max(scores),
            "worst_score": min(scores),
            "recent_average": second_avg,
            "first_average": first_avg,
            "session_changes": session_changes,
            "consistency": self._calculate_consistency(scores),
            "estimated_sessions_to_target": self._estimate_sessions_to_target(scores),
        }

    def get_skill_progress(self) -> Dict[str, Dict]:
        """Get progress for each skill."""
        progress = {}
        for skill_name, skill_data in self.skill_progress.items():
            progress[skill_name] = {
                "latest_score": skill_data.latest_score,
                "average_score": skill_data.average_score,
                "trend": skill_data.trend,
                "improvement_rate": skill_data.improvement_rate,
                "sessions_count": len(skill_data.scores),
                "scores_history": skill_data.scores,
            }
        return progress

    def get_weakness_analysis(self) -> List[Dict]:
        """Identify persistent weaknesses across sessions."""
        weakness_counts = {}
        weakness_scores = {}
        
        for session in self.sessions:
            for weakness in session.weaknesses:
                if weakness not in weakness_counts:
                    weakness_counts[weakness] = 0
                    weakness_scores[weakness] = []
                weakness_counts[weakness] += 1
                weakness_scores[weakness].append(session.overall_score)
        
        # Sort by frequency and severity
        weaknesses = []
        for weakness, count in weakness_counts.items():
            avg_score = sum(weakness_scores[weakness]) / len(weakness_scores[weakness])
            weaknesses.append({
                "weakness": weakness,
                "frequency": count,
                "average_score_when_present": avg_score,
                "severity": "high" if count >= 3 else "medium" if count >= 2 else "low",
            })
        
        return sorted(weaknesses, key=lambda x: (-x["frequency"], x["average_score_when_present"]))

    def get_comparative_analysis(self, session_id_1: str, session_id_2: str) -> Dict:
        """Compare two sessions."""
        s1 = next((s for s in self.sessions if s.session_id == session_id_1), None)
        s2 = next((s for s in self.sessions if s.session_id == session_id_2), None)
        
        if not s1 or not s2:
            return {"error": "Session not found"}
        
        # Compare dimensions
        dimension_comparison = {}
        all_dims = set(s1.dimension_scores.keys()) | set(s2.dimension_scores.keys())
        for dim in all_dims:
            score1 = s1.dimension_scores.get(dim, 0)
            score2 = s2.dimension_scores.get(dim, 0)
            dimension_comparison[dim] = {
                "session_1": score1,
                "session_2": score2,
                "change": score2 - score1,
                "improved": score2 > score1,
            }
        
        return {
            "session_1": {
                "id": s1.session_id,
                "date": datetime.fromtimestamp(s1.timestamp).strftime("%Y-%m-%d"),
                "overall_score": s1.overall_score,
                "recommendation": s1.recommendation,
            },
            "session_2": {
                "id": s2.session_id,
                "date": datetime.fromtimestamp(s2.timestamp).strftime("%Y-%m-%d"),
                "overall_score": s2.overall_score,
                "recommendation": s2.recommendation,
            },
            "overall_change": s2.overall_score - s1.overall_score,
            "dimension_comparison": dimension_comparison,
        }

    def set_goal(self, description: str, target_score: float, target_date: str) -> str:
        """Set a new improvement goal."""
        import uuid
        goal_id = str(uuid.uuid4())[:8]
        
        goal = UserGoal(
            goal_id=goal_id,
            description=description,
            target_score=target_score,
            target_date=target_date,
        )
        
        self.goals.append(goal)
        self._save_data()
        return goal_id

    def get_goals(self) -> List[Dict]:
        """Get all goals with progress."""
        return [{
            "goal_id": g.goal_id,
            "description": g.description,
            "target_score": g.target_score,
            "target_date": g.target_date,
            "current_score": g.current_score,
            "progress": g.progress,
            "status": g.status,
        } for g in self.goals]

    def _update_goals(self, session: SessionRecord):
        """Update goal progress based on new session."""
        for goal in self.goals:
            if goal.status == "active":
                goal.current_score = session.overall_score
                goal.progress = min(1.0, session.overall_score / goal.target_score)
                if session.overall_score >= goal.target_score:
                    goal.status = "achieved"
                elif datetime.now().strftime("%Y-%m-%d") > goal.target_date:
                    goal.status = "expired"

    def _calculate_consistency(self, scores: List[float]) -> float:
        """Calculate score consistency (0-1, higher = more consistent)."""
        if len(scores) < 2:
            return 1.0
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        return max(0, 1 - variance)

    def _estimate_sessions_to_target(self, scores: List[float], target: float = 0.8) -> int:
        """Estimate sessions needed to reach target score."""
        if len(scores) < 2:
            return -1
        
        # Calculate improvement rate
        improvement_per_session = (scores[-1] - scores[0]) / len(scores)
        
        if improvement_per_session <= 0:
            return -1  # Not improving
        
        current = scores[-1]
        if current >= target:
            return 0
        
        sessions_needed = (target - current) / improvement_per_session
        return max(1, int(sessions_needed) + 1)

    def get_overall_stats(self) -> Dict:
        """Get overall user statistics."""
        if not self.sessions:
            return {
                "total_sessions": 0,
                "message": "No sessions recorded yet",
            }
        
        scores = [s.overall_score for s in self.sessions]
        
        return {
            "total_sessions": len(self.sessions),
            "total_practice_hours": sum(s.duration_minutes for s in self.sessions) / 60,
            "average_score": sum(scores) / len(scores),
            "best_score": max(scores),
            "latest_score": scores[-1],
            "improvement": scores[-1] - scores[0],
            "active_goals": len([g for g in self.goals if g.status == "active"]),
            "achieved_goals": len([g for g in self.goals if g.status == "achieved"]),
            "skills_practiced": len(self.skill_progress),
            "companies_practiced": list(set(
                s.company_template for s in self.sessions if s.company_template
            )),
            "recommendation_history": {
                "HIRE": len([s for s in self.sessions if s.recommendation == "HIRE"]),
                "MAYBE": len([s for s in self.sessions if s.recommendation == "MAYBE"]),
                "NO HIRE": len([s for s in self.sessions if s.recommendation == "NO HIRE"]),
            },
        }


# Convenience function
def create_session_tracker(data_dir: str = None) -> SessionTracker:
    """Create a session tracker instance."""
    return SessionTracker(data_dir=data_dir)
