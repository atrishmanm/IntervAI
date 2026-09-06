"""
orchestrator/mock_interview.py
==============================
Complete mock interview simulator with all features.

Features:
1. Full interview lifecycle management
2. Real-time performance tracking
3. Adaptive difficulty
4. Multi-format output (text, JSON, report)
5. Session persistence
6. Comparison with previous interviews
7. Exportable reports
"""

import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from .resume_parser import ResumeParser, ResumeProfile
from .adaptive_interview import AdaptiveInterviewEngine, InterviewSession
from .interview_analytics import InterviewAnalytics


@dataclass
class InterviewConfig:
    """Interview configuration."""
    max_questions: int = 15
    time_limit_minutes: int = 45
    phases: List[str] = field(default_factory=lambda: ["warmup", "technical", "behavioral", "coding", "culture_fit"])
    difficulty_start: str = "medium"
    enable_follow_ups: bool = True
    enable_analytics: bool = True
    export_format: str = "json"  # json, markdown, html


class MockInterviewSimulator:
    """Complete mock interview simulator."""

    def __init__(self, config: InterviewConfig = None):
        self.config = config or InterviewConfig()
        self.engine = AdaptiveInterviewEngine()
        self.analytics = InterviewAnalytics()
        self.parser = ResumeParser()
        self.session_active = False
        self.start_time = None
        self.question_times = []

    def start_interview(self, resume_text: str) -> Dict:
        """Start a new interview session."""
        # Parse resume
        profile = self.parser.parse_resume(resume_text)
        
        # Start interview with engine
        result = self.engine.start_interview(resume_text)
        
        # Initialize analytics
        self.analytics = InterviewAnalytics()
        self.session_active = True
        self.start_time = time.time()
        
        # Store profile for later use
        self.current_profile = profile
        
        return {
            "status": "started",
            "interview_id": result["interview_id"],
            "candidate": {
                "name": result["candidate_name"],
                "skills": result["skills_detected"],
                "experience_years": result["experience_years"],
                "resume_quality": result["resume_quality"],
            },
            "structure": {
                "total_questions": result["total_questions"],
                "phases": result["phases"],
                "estimated_duration": f"{self.config.time_limit_minutes} minutes",
            },
            "first_question": result["first_question"],
            "instructions": (
                "Welcome to your mock interview! "
                "Answer each question thoroughly. "
                "You'll receive feedback after each answer. "
                "The interview will adapt based on your performance."
            ),
        }

    def submit_answer(self, response: str, time_taken: float = 0) -> Dict:
        """Submit an answer and get next question."""
        if not self.session_active:
            return {"error": "No active interview session"}
        
        # Start timing for analytics
        current_question_idx = self.engine.session.current_question_idx
        current_question = self.engine.session.questions[current_question_idx]
        
        self.analytics.start_question(
            skill=current_question.skill_tags[0] if current_question.skill_tags else None,
            phase=current_question.phase.value
        )
        
        # Submit to engine
        result = self.engine.submit_answer(response, time_taken)
        
        # Record in analytics
        if "score" in result:
            self.analytics.record_answer(
                question_number=current_question_idx + 1,
                scores=result["score"],
                difficulty=current_question.difficulty.name
            )
        
        # Check if interview is complete
        if result.get("is_complete"):
            self.session_active = False
            result["final_report"] = self._generate_final_report()
        
        # Add analytics to response
        result["analytics"] = self.analytics.get_performance_summary()
        
        return result

    def get_analytics(self) -> Dict:
        """Get current analytics."""
        return self.analytics.get_performance_summary()

    def get_radar_chart(self) -> Dict:
        """Get radar chart data."""
        return self.analytics.get_radar_chart_data()

    def get_timeline(self) -> Dict:
        """Get timeline data."""
        return self.analytics.get_timeline_data()

    def get_phase_breakdown(self) -> Dict:
        """Get phase breakdown."""
        return self.analytics.get_phase_breakdown()

    def get_prediction(self) -> Dict:
        """Get success prediction."""
        return self.analytics.predict_success()

    def _generate_final_report(self) -> Dict:
        """Generate comprehensive final report."""
        if not self.engine.session:
            return {}
        
        # Get engine report
        engine_report = self.engine._generate_final_report()
        
        # Get analytics report
        analytics_report = self.analytics.generate_report()
        
        # Combine reports
        final_report = {
            "interview_summary": {
                "candidate_name": self.current_profile.name if self.current_profile else "Unknown",
                "total_questions": len(self.engine.session.questions),
                "questions_answered": len(self.engine.session.answers),
                "duration_minutes": (time.time() - self.start_time) / 60 if self.start_time else 0,
                "average_score": engine_report.get("overall_score", 0),
                "recommendation": engine_report.get("recommendation", "NO HIRE"),
            },
            "performance_scores": {
                "overall": engine_report.get("overall_score", 0),
                "by_dimension": engine_report.get("dimension_scores", {}),
                "by_phase": engine_report.get("phase_scores", {}),
            },
            "strengths": engine_report.get("strengths", []),
            "weaknesses": engine_report.get("weaknesses", []),
            "detailed_feedback": engine_report.get("detailed_feedback", ""),
            "analytics_summary": analytics_report.get("performance_summary", {}),
            "success_prediction": analytics_report.get("success_prediction", {}),
            "benchmarks": analytics_report.get("benchmarks", {}),
            "recommendations": analytics_report.get("recommendations", []),
            "question_details": self._get_question_details(),
        }
        
        return final_report

    def _get_question_details(self) -> List[Dict]:
        """Get detailed breakdown of each question."""
        details = []
        
        for i, answer in enumerate(self.engine.session.answers):
            detail = {
                "question_number": i + 1,
                "question": answer.question.text,
                "phase": answer.question.phase.value,
                "difficulty": answer.question.difficulty.name,
                "response": answer.response[:200] + "..." if len(answer.response) > 200 else answer.response,
                "scores": answer.scores,
                "average_score": sum(answer.scores.values()) / max(len(answer.scores), 1),
                "feedback": answer.feedback,
                "time_taken": answer.time_taken_sec,
            }
            
            # Add STAR analysis for behavioral questions
            if answer.star_completeness:
                detail["star_analysis"] = answer.star_completeness
            
            details.append(detail)
        
        return details

    def export_report(self, format: str = "json") -> str:
        """Export interview report."""
        report = self._generate_final_report()
        
        if format == "json":
            return json.dumps(report, indent=2, default=str)
        elif format == "markdown":
            return self._convert_to_markdown(report)
        elif format == "html":
            return self._convert_to_html(report)
        else:
            return json.dumps(report, indent=2, default=str)

    def _convert_to_markdown(self, report: Dict) -> str:
        """Convert report to markdown format."""
        md = []
        
        md.append("# Interview Report")
        md.append("")
        
        # Summary
        summary = report.get("interview_summary", {})
        md.append("## Summary")
        md.append(f"- **Candidate**: {summary.get('candidate_name', 'N/A')}")
        md.append(f"- **Questions Answered**: {summary.get('questions_answered', 0)}/{summary.get('total_questions', 0)}")
        md.append(f"- **Duration**: {summary.get('duration_minutes', 0):.1f} minutes")
        md.append(f"- **Overall Score**: {summary.get('average_score', 0):.2%}")
        md.append(f"- **Recommendation**: {summary.get('recommendation', 'N/A')}")
        md.append("")
        
        # Performance Scores
        scores = report.get("performance_scores", {})
        md.append("## Performance Scores")
        md.append(f"- **Overall**: {scores.get('overall', 0):.2%}")
        md.append("")
        
        by_dimension = scores.get("by_dimension", {})
        if by_dimension:
            md.append("### By Dimension")
            for dim, score in by_dimension.items():
                md.append(f"- **{dim.replace('_', ' ').title()}**: {score:.2%}")
            md.append("")
        
        by_phase = scores.get("by_phase", {})
        if by_phase:
            md.append("### By Phase")
            for phase, score in by_phase.items():
                md.append(f"- **{phase.replace('_', ' ').title()}**: {score:.2%}")
            md.append("")
        
        # Strengths & Weaknesses
        strengths = report.get("strengths", [])
        weaknesses = report.get("weaknesses", [])
        
        if strengths:
            md.append("## Strengths")
            for s in strengths:
                md.append(f"- {s.replace('_', ' ').title()}")
            md.append("")
        
        if weaknesses:
            md.append("## Areas for Improvement")
            for w in weaknesses:
                md.append(f"- {w.replace('_', ' ').title()}")
            md.append("")
        
        # Recommendations
        recommendations = report.get("recommendations", [])
        if recommendations:
            md.append("## Recommendations")
            for r in recommendations:
                md.append(f"- {r}")
            md.append("")
        
        return "\n".join(md)

    def _convert_to_html(self, report: Dict) -> str:
        """Convert report to HTML format."""
        html = []
        
        html.append("<!DOCTYPE html>")
        html.append("<html><head><title>Interview Report</title></head><body>")
        html.append("<h1>Interview Report</h1>")
        
        # Summary
        summary = report.get("interview_summary", {})
        html.append("<h2>Summary</h2>")
        html.append("<ul>")
        html.append(f"<li><strong>Candidate</strong>: {summary.get('candidate_name', 'N/A')}</li>")
        html.append(f"<li><strong>Questions Answered</strong>: {summary.get('questions_answered', 0)}/{summary.get('total_questions', 0)}</li>")
        html.append(f"<li><strong>Duration</strong>: {summary.get('duration_minutes', 0):.1f} minutes</li>")
        html.append(f"<li><strong>Overall Score</strong>: {summary.get('average_score', 0):.2%}</li>")
        html.append(f"<li><strong>Recommendation</strong>: {summary.get('recommendation', 'N/A')}</li>")
        html.append("</ul>")
        
        html.append("</body></html>")
        
        return "\n".join(html)


# Convenience function
def create_mock_interview(config: InterviewConfig = None) -> MockInterviewSimulator:
    """Create a mock interview simulator instance."""
    return MockInterviewSimulator(config=config)
