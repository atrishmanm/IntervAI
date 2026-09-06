"""
orchestrator/interview_analytics.py
====================================
Real-time analytics for interview performance tracking.

Features:
1. Live performance metrics
2. Skill proficiency radar chart data
3. Response quality trends
4. Time management analytics
5. Comparison with benchmarks
6. Predictive success scoring
7. Detailed breakdown by dimension
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics."""
    total_questions: int = 0
    questions_answered: int = 0
    average_score: float = 0.0
    best_score: float = 0.0
    worst_score: float = 1.0
    current_streak: int = 0
    best_streak: int = 0
    time_per_question: List[float] = field(default_factory=list)
    scores_over_time: List[float] = field(default_factory=list)
    skill_scores: Dict[str, List[float]] = field(default_factory=dict)
    phase_scores: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class AnalyticsSnapshot:
    """Point-in-time analytics snapshot."""
    timestamp: float
    question_number: int
    current_score: float
    cumulative_score: float
    skill_scores: Dict[str, float]
    phase_scores: Dict[str, float]
    time_taken: float
    difficulty: str
    phase: str


class InterviewAnalytics:
    """Real-time interview analytics engine."""

    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.snapshots: List[AnalyticsSnapshot] = []
        self.session_start_time = time.time()
        self.question_start_time = None
        self.current_skill = None
        self.current_phase = None

    def start_question(self, skill: str = None, phase: str = None):
        """Mark start of a new question."""
        self.question_start_time = time.time()
        self.current_skill = skill
        self.current_phase = phase

    def record_answer(self, question_number: int, scores: Dict[str, float], difficulty: str):
        """Record answer and update metrics."""
        if self.question_start_time is None:
            return
        
        time_taken = time.time() - self.question_start_time
        
        # Calculate average score for this question
        avg_score = sum(scores.values()) / max(len(scores), 1)
        
        # Update metrics
        self.metrics.questions_answered += 1
        self.metrics.time_per_question.append(time_taken)
        self.metrics.scores_over_time.append(avg_score)
        
        # Update best/worst
        if avg_score > self.metrics.best_score:
            self.metrics.best_score = avg_score
        if avg_score < self.metrics.worst_score:
            self.metrics.worst_score = avg_score
        
        # Update streak
        if avg_score >= 0.7:
            self.metrics.current_streak += 1
            if self.metrics.current_streak > self.metrics.best_streak:
                self.metrics.best_streak = self.metrics.current_streak
        else:
            self.metrics.current_streak = 0
        
        # Update skill scores
        if self.current_skill:
            if self.current_skill not in self.metrics.skill_scores:
                self.metrics.skill_scores[self.current_skill] = []
            self.metrics.skill_scores[self.current_skill].append(avg_score)
        
        # Update phase scores
        if self.current_phase:
            if self.current_phase not in self.metrics.phase_scores:
                self.metrics.phase_scores[self.current_phase] = []
            self.metrics.phase_scores[self.current_phase].append(avg_score)
        
        # Calculate cumulative average
        self.metrics.average_score = sum(self.metrics.scores_over_time) / len(self.metrics.scores_over_time)
        
        # Create snapshot
        snapshot = AnalyticsSnapshot(
            timestamp=time.time(),
            question_number=question_number,
            current_score=avg_score,
            cumulative_score=self.metrics.average_score,
            skill_scores=self._get_current_skill_scores(),
            phase_scores=self._get_current_phase_scores(),
            time_taken=time_taken,
            difficulty=difficulty,
            phase=self.current_phase or "unknown"
        )
        self.snapshots.append(snapshot)
        
        # Reset question timer
        self.question_start_time = None

    def _get_current_skill_scores(self) -> Dict[str, float]:
        """Get current average scores per skill."""
        result = {}
        for skill, scores in self.metrics.skill_scores.items():
            result[skill] = sum(scores) / len(scores)
        return result

    def _get_current_phase_scores(self) -> Dict[str, float]:
        """Get current average scores per phase."""
        result = {}
        for phase, scores in self.metrics.phase_scores.items():
            result[phase] = sum(scores) / len(scores)
        return result

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        elapsed_time = time.time() - self.session_start_time
        
        return {
            "elapsed_time_seconds": elapsed_time,
            "elapsed_time_formatted": self._format_time(elapsed_time),
            "questions_answered": self.metrics.questions_answered,
            "average_score": self.metrics.average_score,
            "best_score": self.metrics.best_score,
            "worst_score": self.metrics.worst_score,
            "current_streak": self.metrics.current_streak,
            "best_streak": self.metrics.best_streak,
            "avg_time_per_question": sum(self.metrics.time_per_question) / max(len(self.metrics.time_per_question), 1),
            "skill_scores": self._get_current_skill_scores(),
            "phase_scores": self._get_current_phase_scores(),
            "score_trend": self._calculate_trend(),
            "performance_level": self._get_performance_level(),
        }

    def _calculate_trend(self) -> str:
        """Calculate score trend (improving/declining/stable)."""
        if len(self.metrics.scores_over_time) < 3:
            return "insufficient_data"
        
        recent = self.metrics.scores_over_time[-3:]
        earlier = self.metrics.scores_over_time[:-3] if len(self.metrics.scores_over_time) > 3 else self.metrics.scores_over_time[:1]
        
        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)
        
        if recent_avg > earlier_avg + 0.1:
            return "improving"
        elif recent_avg < earlier_avg - 0.1:
            return "declining"
        else:
            return "stable"

    def _get_performance_level(self) -> str:
        """Get overall performance level."""
        avg = self.metrics.average_score
        if avg >= 0.9:
            return "exceptional"
        elif avg >= 0.8:
            return "excellent"
        elif avg >= 0.7:
            return "good"
        elif avg >= 0.6:
            return "satisfactory"
        elif avg >= 0.5:
            return "needs_improvement"
        else:
            return "poor"

    def _format_time(self, seconds: float) -> str:
        """Format seconds to HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def get_radar_chart_data(self) -> Dict:
        """Get data for radar/spider chart visualization."""
        skill_scores = self._get_current_skill_scores()
        
        # Normalize to 0-100 scale
        labels = list(skill_scores.keys()) if skill_scores else ["No skills assessed"]
        values = [skill_scores[s] * 100 for s in labels] if skill_scores else [0]
        
        return {
            "labels": labels,
            "values": values,
            "max_value": 100,
        }

    def get_timeline_data(self) -> Dict:
        """Get timeline data for chart visualization."""
        timestamps = [s.timestamp - self.session_start_time for s in self.snapshots]
        scores = [s.current_score for s in self.snapshots]
        cumulative = [s.cumulative_score for s in self.snapshots]
        
        return {
            "timestamps": timestamps,
            "scores": scores,
            "cumulative_scores": cumulative,
            "labels": [f"Q{s.question_number}" for s in self.snapshots],
        }

    def get_phase_breakdown(self) -> Dict:
        """Get detailed phase breakdown."""
        phase_data = {}
        
        for phase, scores in self.metrics.phase_scores.items():
            phase_data[phase] = {
                "average": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "count": len(scores),
                "trend": self._calculate_phase_trend(scores),
            }
        
        return phase_data

    def _calculate_phase_trend(self, scores: List[float]) -> str:
        """Calculate trend for a specific phase."""
        if len(scores) < 2:
            return "insufficient_data"
        
        recent = scores[-2:]
        earlier = scores[:-2] if len(scores) > 2 else scores[:1]
        
        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)
        
        if recent_avg > earlier_avg + 0.1:
            return "improving"
        elif recent_avg < earlier_avg - 0.1:
            return "declining"
        else:
            return "stable"

    def predict_success(self) -> Dict:
        """Predict interview success probability."""
        avg_score = self.metrics.average_score
        trend = self._calculate_trend()
        
        # Simple prediction model
        base_probability = avg_score * 100
        
        # Adjust for trend
        if trend == "improving":
            base_probability += 5
        elif trend == "declining":
            base_probability -= 5
        
        # Adjust for streak
        if self.metrics.best_streak >= 3:
            base_probability += 3
        
        # Clamp to 0-100
        probability = max(0, min(100, base_probability))
        
        return {
            "success_probability": probability,
            "confidence_level": "high" if self.metrics.questions_answered >= 5 else "medium",
            "factors": {
                "average_score": avg_score,
                "trend": trend,
                "best_streak": self.metrics.best_streak,
                "consistency": self._calculate_consistency(),
            }
        }

    def _calculate_consistency(self) -> float:
        """Calculate score consistency (lower variance = more consistent)."""
        if len(self.metrics.scores_over_time) < 2:
            return 0.0
        
        mean = self.metrics.average_score
        variance = sum((x - mean) ** 2 for x in self.metrics.scores_over_time) / len(self.metrics.scores_over_time)
        
        # Convert to 0-1 scale (lower variance = higher consistency)
        consistency = max(0, 1 - variance)
        return consistency

    def get_benchmarks(self) -> Dict:
        """Get comparison benchmarks."""
        return {
            "industry_average": 0.65,
            "top_performer_threshold": 0.85,
            "minimum_passing": 0.60,
            "your_score": self.metrics.average_score,
            "percentile_estimate": self._estimate_percentile(),
        }

    def _estimate_percentile(self) -> float:
        """Estimate percentile based on score."""
        # Simple percentile estimation
        score = self.metrics.average_score
        if score >= 0.9:
            return 95
        elif score >= 0.8:
            return 85
        elif score >= 0.7:
            return 70
        elif score >= 0.6:
            return 55
        elif score >= 0.5:
            return 40
        else:
            return 20

    def generate_report(self) -> Dict:
        """Generate comprehensive analytics report."""
        return {
            "performance_summary": self.get_performance_summary(),
            "radar_chart": self.get_radar_chart_data(),
            "timeline": self.get_timeline_data(),
            "phase_breakdown": self.get_phase_breakdown(),
            "success_prediction": self.predict_success(),
            "benchmarks": self.get_benchmarks(),
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        avg = self.metrics.average_score
        
        if avg < 0.6:
            recommendations.append("Focus on providing more detailed, structured answers")
            recommendations.append("Practice using the STAR method for behavioral questions")
        
        if self.metrics.best_streak < 2:
            recommendations.append("Work on maintaining consistent performance across questions")
        
        phase_scores = self._get_current_phase_scores()
        for phase, score in phase_scores.items():
            if score < 0.6:
                recommendations.append(f"Improve performance in {phase.replace('_', ' ')} phase")
        
        if not recommendations:
            recommendations.append("Continue maintaining your strong performance!")
        
        return recommendations


# Convenience function
def create_analytics() -> InterviewAnalytics:
    """Create an analytics instance."""
    return InterviewAnalytics()
