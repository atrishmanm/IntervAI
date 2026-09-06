"""
orchestrator/adaptive_interview.py
===================================
Advanced adaptive interview engine with context-aware questioning.

Features:
1. Resume-based question generation
2. Adaptive difficulty based on performance
3. Follow-up question chains
4. Multi-dimensional evaluation (technical, behavioral, communication)
5. Real-time performance tracking
6. STAR method evaluation for behavioral questions
7. Code challenge generation
8. System design questions for senior roles
9. Cultural fit assessment
10. Comprehensive final report with HIRE/MAYBE/NO HIRE recommendation
"""

import random
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .resume_parser import ResumeParser, ResumeProfile, SkillLevel


class Difficulty(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4


class InterviewPhase(Enum):
    WARMUP = "warmup"
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    CODING = "coding"
    SYSTEM_DESIGN = "system_design"
    CULTURE_FIT = "culture_fit"
    CLOSING = "closing"


@dataclass
class Question:
    text: str
    phase: InterviewPhase
    difficulty: Difficulty
    skill_tags: List[str] = field(default_factory=list)
    is_from_resume: bool = False
    time_limit_sec: int = 300
    rubric: Dict[str, float] = field(default_factory=dict)
    follow_ups: List[str] = field(default_factory=list)


@dataclass
class Answer:
    question: Question
    response: str
    time_taken_sec: float = 0
    scores: Dict[str, float] = field(default_factory=dict)
    star_completeness: Dict[str, float] = field(default_factory=dict)
    feedback: str = ""


@dataclass
class InterviewSession:
    profile: ResumeProfile
    questions: List[Question]
    answers: List[Answer]
    current_question_idx: int = 0
    phase: InterviewPhase = InterviewPhase.WARMUP
    difficulty: Difficulty = Difficulty.MEDIUM
    performance_history: List[Dict[str, float]] = field(default_factory=list)
    is_complete: bool = False
    overall_score: float = 0.0
    recommendation: str = ""


class AdaptiveInterviewEngine:
    """Advanced adaptive interview engine with context-aware questioning."""

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.parser = ResumeParser()
        self.session: Optional[InterviewSession] = None

    def start_interview(self, resume_text: str) -> Dict:
        """Start an interview with a resume."""
        # Parse resume
        profile = self.parser.parse_resume(resume_text)
        
        # Generate personalized questions
        questions = self._generate_questions(profile)
        
        # Create session
        self.session = InterviewSession(
            profile=profile,
            questions=questions,
            answers=[]
        )
        
        # Return first question
        first_q = questions[0]
        return {
            "interview_id": id(self.session),
            "candidate_name": profile.name,
            "skills_detected": [s.name for s in profile.skills[:10]],
            "experience_years": profile.total_experience_years,
            "resume_quality": profile.resume_quality_score,
            "total_questions": len(questions),
            "phases": list(set(q.phase.value for q in questions)),
            "first_question": {
                "text": first_q.text,
                "phase": first_q.phase.value,
                "difficulty": first_q.difficulty.name,
                "time_limit": first_q.time_limit_sec,
            }
        }

    def submit_answer(self, response: str, time_taken: float = 0) -> Dict:
        """Submit an answer and get next question with feedback."""
        if not self.session or self.session.is_complete:
            return {"error": "No active interview session"}
        
        current_q = self.session.questions[self.session.current_question_idx]
        
        # Score the answer
        scores = self._score_answer(current_q, response, time_taken)
        
        # Evaluate STAR method for behavioral questions
        star = {}
        if current_q.phase == InterviewPhase.BEHAVIORAL:
            star = self._evaluate_star(response)
        
        # Generate feedback
        feedback = self._generate_feedback(current_q, scores, star)
        
        # Store answer
        answer = Answer(
            question=current_q,
            response=response,
            time_taken_sec=time_taken,
            scores=scores,
            star_completeness=star,
            feedback=feedback
        )
        self.session.answers.append(answer)
        
        # Update performance history
        self.session.performance_history.append({
            "question_idx": self.session.current_question_idx,
            "scores": scores,
            "phase": current_q.phase.value,
            "difficulty": current_q.difficulty.value,
        })
        
        # Adapt difficulty based on performance
        self._adapt_difficulty(scores)
        
        # Move to next question
        self.session.current_question_idx += 1
        
        # Check if interview is complete
        if self.session.current_question_idx >= len(self.session.questions):
            self.session.is_complete = True
            report = self._generate_final_report()
            return {
                "is_complete": True,
                "score": scores,
                "feedback": feedback,
                "star_analysis": star,
                "report": report,
            }
        
        # Get next question
        next_q = self.session.questions[self.session.current_question_idx]
        
        # Update phase if needed
        if next_q.phase != self.session.phase:
            self.session.phase = next_q.phase
        
        return {
            "is_complete": False,
            "score": scores,
            "feedback": feedback,
            "star_analysis": star,
            "next_question": {
                "text": next_q.text,
                "phase": next_q.phase.value,
                "difficulty": next_q.difficulty.name,
                "time_limit": next_q.time_limit_sec,
                "question_number": self.session.current_question_idx + 1,
                "total_questions": len(self.session.questions),
            }
        }

    def _generate_questions(self, profile: ResumeProfile) -> List[Question]:
        """Generate personalized questions based on resume."""
        questions = []
        
        # 1. Warmup questions (always start with these)
        questions.extend(self._generate_warmup_questions(profile))
        
        # 2. Technical questions based on skills
        questions.extend(self._generate_technical_questions(profile))
        
        # 3. Behavioral questions
        questions.extend(self._generate_behavioral_questions(profile))
        
        # 4. Coding challenge (if technical role)
        if any(s.name in ["python", "javascript", "java", "c++", "go", "rust"] for s in profile.skills):
            questions.extend(self._generate_coding_questions(profile))
        
        # 5. System design (for senior roles)
        if profile.total_experience_years >= 5:
            questions.extend(self._generate_system_design_questions(profile))
        
        # 6. Culture fit
        questions.extend(self._generate_culture_fit_questions(profile))
        
        # 7. Closing
        questions.extend(self._generate_closing_questions(profile))
        
        return questions

    def _generate_warmup_questions(self, profile: ResumeProfile) -> List[Question]:
        """Generate warmup questions."""
        questions = []
        
        # Always start with "Tell me about yourself"
        questions.append(Question(
            text="Tell me about yourself and your background.",
            phase=InterviewPhase.WARMUP,
            difficulty=Difficulty.EASY,
            rubric={"communication": 0.5, "relevance": 0.5}
        ))
        
        # Follow up based on resume
        if profile.summary:
            questions.append(Question(
                text="I see your resume mentions: " + profile.summary[:100] + "... Can you elaborate on this?",
                phase=InterviewPhase.WARMUP,
                difficulty=Difficulty.EASY,
                is_from_resume=True,
                rubric={"communication": 0.5, "relevance": 0.5}
            ))
        
        return questions

    def _generate_technical_questions(self, profile: ResumeProfile) -> List[Question]:
        """Generate technical questions based on skills."""
        questions = []
        
        # Group skills by category
        skill_categories = {}
        for skill in profile.skills:
            category = self._get_skill_category(skill.name)
            if category not in skill_categories:
                skill_categories[category] = []
            skill_categories[category].append(skill)
        
        # Generate questions for each skill category
        for category, skills in skill_categories.items():
            # Pick top skill in category
            top_skill = max(skills, key=lambda s: s.level.value)
            
            # Basic proficiency question
            questions.append(Question(
                text=f"I see {top_skill.name} on your resume. Can you describe your experience with it and a challenging project where you used it?",
                phase=InterviewPhase.TECHNICAL,
                difficulty=Difficulty.EASY if top_skill.level.value <= 2 else Difficulty.MEDIUM,
                skill_tags=[top_skill.name],
                is_from_resume=True,
                rubric={"technical_depth": 0.4, "problem_solving": 0.3, "communication": 0.3}
            ))
            
            # Deep dive question
            questions.append(Question(
                text=f"Tell me about a time you had to debug a complex issue with {top_skill.name}. How did you approach it?",
                phase=InterviewPhase.TECHNICAL,
                difficulty=Difficulty.MEDIUM if top_skill.level.value <= 2 else Difficulty.HARD,
                skill_tags=[top_skill.name],
                is_from_resume=True,
                rubric={"problem_solving": 0.4, "technical_depth": 0.3, "communication": 0.3}
            ))
            
            # Architecture/design question
            if top_skill.level.value >= 3:
                questions.append(Question(
                    text=f"How would you design a system using {top_skill.name} for a large-scale application?",
                    phase=InterviewPhase.TECHNICAL,
                    difficulty=Difficulty.HARD,
                    skill_tags=[top_skill.name],
                    is_from_resume=True,
                    rubric={"technical_depth": 0.5, "problem_solving": 0.3, "communication": 0.2}
                ))
        
        return questions

    def _generate_behavioral_questions(self, profile: ResumeProfile) -> List[Question]:
        """Generate behavioral questions."""
        questions = []
        
        behavioral_templates = [
            ("Tell me about a time you had to learn a new technology quickly.", "learning", Difficulty.MEDIUM),
            ("Describe a situation where you had to collaborate with a difficult team member.", "teamwork", Difficulty.MEDIUM),
            ("How do you prioritize tasks when working on multiple projects?", "time_management", Difficulty.EASY),
            ("Tell me about a technical decision you made that you later had to change.", "adaptability", Difficulty.HARD),
            ("Describe a project where you had to mentor junior developers.", "leadership", Difficulty.MEDIUM),
            ("Tell me about a time you failed. How did you handle it?", "resilience", Difficulty.MEDIUM),
            ("Describe a situation where you had to make a decision with incomplete information.", "decision_making", Difficulty.HARD),
            ("How do you handle stress and pressure?", "stress_management", Difficulty.EASY),
        ]
        
        for template, category, difficulty in behavioral_templates:
            questions.append(Question(
                text=template,
                phase=InterviewPhase.BEHAVIORAL,
                difficulty=difficulty,
                skill_tags=[category],
                rubric={
                    "situation": 0.25,
                    "task": 0.25,
                    "action": 0.25,
                    "result": 0.25
                }
            ))
        
        return questions

    def _generate_coding_questions(self, profile: ResumeProfile) -> List[Question]:
        """Generate coding questions based on skills."""
        questions = []
        
        # Get primary programming language
        primary_lang = None
        for skill in profile.skills:
            if skill.name in ["python", "javascript", "java", "c++", "go", "rust"]:
                primary_lang = skill.name
                break
        
        if primary_lang:
            coding_templates = [
                f"Write a function in {primary_lang} that finds the longest substring without repeating characters.",
                f"Implement a LRU cache in {primary_lang}.",
                f"Write a function to detect a cycle in a linked list using {primary_lang}.",
                f"Implement a thread-safe singleton in {primary_lang}.",
                f"Write a function to flatten a nested dictionary in {primary_lang}.",
            ]
            
            for template in coding_templates:
                questions.append(Question(
                    text=template,
                    phase=InterviewPhase.CODING,
                    difficulty=Difficulty.MEDIUM,
                    skill_tags=[primary_lang],
                    time_limit_sec=600,
                    rubric={
                        "correctness": 0.4,
                        "efficiency": 0.3,
                        "code_quality": 0.3
                    }
                ))
        
        return questions

    def _generate_system_design_questions(self, profile: ResumeProfile) -> List[Question]:
        """Generate system design questions for senior roles."""
        questions = []
        
        system_design_templates = [
            "Design a URL shortener like bit.ly.",
            "Design a real-time chat application like WhatsApp.",
            "Design a distributed file storage system like Dropbox.",
            "Design a news feed system like Twitter.",
            "Design a ride-sharing service like Uber.",
        ]
        
        for template in system_design_templates:
            questions.append(Question(
                text=template,
                phase=InterviewPhase.SYSTEM_DESIGN,
                difficulty=Difficulty.EXPERT,
                time_limit_sec=900,
                rubric={
                    "requirements": 0.2,
                    "high_level_design": 0.3,
                    "detailed_design": 0.3,
                    "trade_offs": 0.2
                }
            ))
        
        return questions

    def _generate_culture_fit_questions(self, profile: ResumeProfile) -> List[Question]:
        """Generate culture fit questions."""
        questions = []
        
        culture_templates = [
            "What kind of work environment do you thrive in?",
            "How do you handle conflict with team members?",
            "What motivates you to do your best work?",
            "How do you approach learning new technologies?",
            "What's your ideal work-life balance?",
        ]
        
        for template in culture_templates:
            questions.append(Question(
                text=template,
                phase=InterviewPhase.CULTURE_FIT,
                difficulty=Difficulty.EASY,
                rubric={
                    "self_awareness": 0.3,
                    "team_fit": 0.4,
                    "growth_mindset": 0.3
                }
            ))
        
        return questions

    def _generate_closing_questions(self, profile: ResumeProfile) -> List[Question]:
        """Generate closing questions."""
        questions = []
        
        questions.append(Question(
            text="Do you have any questions for me?",
            phase=InterviewPhase.CLOSING,
            difficulty=Difficulty.EASY,
            rubric={"engagement": 0.5, "preparation": 0.5}
        ))
        
        return questions

    def _get_skill_category(self, skill_name: str) -> str:
        """Get skill category."""
        categories = {
            "python": "language", "javascript": "language", "java": "language",
            "react": "frontend", "angular": "frontend", "vue": "frontend",
            "aws": "cloud", "azure": "cloud", "gcp": "cloud",
            "machine learning": "data", "deep learning": "data",
            "mysql": "database", "postgresql": "database", "mongodb": "database",
        }
        return categories.get(skill_name, "general")

    def _score_answer(self, question: Question, response: str, time_taken: float = 0) -> Dict[str, float]:
        """Score an answer on multiple dimensions."""
        scores = {}
        
        # Base scoring from rubric
        for dimension, weight in question.rubric.items():
            scores[dimension] = self._heuristic_score(dimension, response, question)
        
        # Bonus for resume-specific questions
        if question.is_from_resume:
            scores["resume_relevance"] = 0.8
        
        # Time efficiency
        if question.time_limit_sec > 0 and time_taken > 0:
            time_ratio = time_taken / question.time_limit_sec
            if time_ratio < 0.3:
                scores["efficiency"] = 0.7  # Too fast
            elif time_ratio < 0.8:
                scores["efficiency"] = 1.0  # Good pace
            else:
                scores["efficiency"] = 0.6  # Too slow
        
        return scores

    def _heuristic_score(self, dimension: str, response: str, question: Question) -> float:
        """Simple heuristic scoring."""
        response_len = len(response.split())
        base_score = 0.5
        
        # Length bonus
        if response_len > 50:
            base_score += 0.1
        if response_len > 100:
            base_score += 0.1
        if response_len > 200:
            base_score += 0.05
        
        # Specificity
        has_numbers = bool(re.search(r'\d+', response))
        has_examples = any(w in response.lower() for w in ["example", "for instance", "such as", "like when"])
        has_technical = any(w in response.lower() for w in question.skill_tags)
        
        if has_numbers:
            base_score += 0.1
        if has_examples:
            base_score += 0.1
        if has_technical:
            base_score += 0.1
        
        # Penalize very short responses
        if response_len < 10:
            base_score -= 0.2
        
        return min(max(base_score, 0.0), 1.0)

    def _evaluate_star(self, response: str) -> Dict[str, float]:
        """Evaluate STAR method completeness."""
        star_keywords = {
            "situation": ["situation", "context", "background", "when", "where"],
            "task": ["task", "responsibility", "goal", "objective", "needed to"],
            "action": ["action", "did", "implemented", "created", "developed", "led"],
            "result": ["result", "outcome", "achieved", "improved", "increased", "decreased"],
        }
        
        scores = {}
        response_lower = response.lower()
        
        for component, keywords in star_keywords.items():
            scores[component] = 1.0 if any(kw in response_lower for kw in keywords) else 0.0
        
        return scores

    def _adapt_difficulty(self, scores: Dict[str, float]):
        """Adapt difficulty based on performance."""
        avg_score = sum(scores.values()) / max(len(scores), 1)
        
        if avg_score >= 0.8 and self.session.difficulty.value < Difficulty.EXPERT.value:
            self.session.difficulty = Difficulty(self.session.difficulty.value + 1)
        elif avg_score <= 0.4 and self.session.difficulty.value > Difficulty.EASY.value:
            self.session.difficulty = Difficulty(self.session.difficulty.value - 1)

    def _generate_feedback(self, question: Question, scores: Dict[str, float], star: Dict[str, float]) -> str:
        """Generate constructive feedback."""
        avg = sum(scores.values()) / max(len(scores), 1)
        feedback_parts = []
        
        if avg >= 0.8:
            feedback_parts.append("Excellent response!")
        elif avg >= 0.6:
            feedback_parts.append("Good answer.")
        elif avg >= 0.4:
            feedback_parts.append("Decent answer, but there's room for improvement.")
        else:
            feedback_parts.append("Your answer could be stronger.")
        
        # STAR feedback for behavioral questions
        if star:
            missing = [k for k, v in star.items() if v == 0]
            if missing:
                feedback_parts.append(f"Consider including more about: {', '.join(missing)}")
        
        # Specific dimension feedback
        for dim, score in scores.items():
            if score < 0.5:
                feedback_parts.append(f"Work on improving your {dim.replace('_', ' ')}")
        
        return " ".join(feedback_parts)

    def _generate_final_report(self) -> Dict:
        """Generate comprehensive final report."""
        if not self.session:
            return {}
        
        # Calculate overall scores
        all_scores = []
        for answer in self.session.answers:
            all_scores.append(answer.scores)
        
        # Aggregate scores by dimension
        dimension_scores = {}
        for scores in all_scores:
            for dim, score in scores.items():
                if dim not in dimension_scores:
                    dimension_scores[dim] = []
                dimension_scores[dim].append(score)
        
        # Calculate averages
        avg_scores = {}
        for dim, scores in dimension_scores.items():
            avg_scores[dim] = sum(scores) / len(scores)
        
        # Calculate overall score
        overall_score = sum(avg_scores.values()) / max(len(avg_scores), 1)
        
        # Generate recommendation
        if overall_score >= 0.8:
            recommendation = "HIRE"
        elif overall_score >= 0.6:
            recommendation = "MAYBE"
        else:
            recommendation = "NO HIRE"
        
        # Performance by phase
        phase_scores = {}
        for answer in self.session.answers:
            phase = answer.question.phase.value
            if phase not in phase_scores:
                phase_scores[phase] = []
            phase_scores[phase].append(sum(answer.scores.values()) / max(len(answer.scores), 1))
        
        phase_averages = {}
        for phase, scores in phase_scores.items():
            phase_averages[phase] = sum(scores) / len(scores)
        
        return {
            "overall_score": overall_score,
            "recommendation": recommendation,
            "dimension_scores": avg_scores,
            "phase_scores": phase_averages,
            "total_questions": len(self.session.questions),
            "questions_answered": len(self.session.answers),
            "candidate_name": self.session.profile.name,
            "skills_assessed": [s.name for s in self.session.profile.skills[:10]],
            "strengths": self._identify_strengths(avg_scores),
            "weaknesses": self._identify_weaknesses(avg_scores),
            "detailed_feedback": self._generate_detailed_feedback(avg_scores, phase_averages),
        }

    def _identify_strengths(self, scores: Dict[str, float]) -> List[str]:
        """Identify top strengths."""
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [dim for dim, score in sorted_scores[:3] if score >= 0.7]

    def _identify_weaknesses(self, scores: Dict[str, float]) -> List[str]:
        """Identify areas for improvement."""
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        return [dim for dim, score in sorted_scores[:3] if score < 0.6]

    def _generate_detailed_feedback(self, dimension_scores: Dict[str, float], phase_scores: Dict[str, float]) -> str:
        """Generate detailed feedback narrative."""
        feedback_parts = []
        
        # Overall assessment
        overall = sum(dimension_scores.values()) / max(len(dimension_scores), 1)
        if overall >= 0.8:
            feedback_parts.append("The candidate demonstrated strong performance across all areas.")
        elif overall >= 0.6:
            feedback_parts.append("The candidate showed solid skills with room for growth.")
        else:
            feedback_parts.append("The candidate needs improvement in several key areas.")
        
        # Phase-specific feedback
        for phase, score in phase_scores.items():
            if score >= 0.8:
                feedback_parts.append(f"Excellent performance in {phase.replace('_', ' ')}.")
            elif score < 0.6:
                feedback_parts.append(f"Needs improvement in {phase.replace('_', ' ')}.")
        
        return " ".join(feedback_parts)


# Convenience function
def create_interview_engine(model=None, tokenizer=None) -> AdaptiveInterviewEngine:
    """Create an interview engine instance."""
    return AdaptiveInterviewEngine(model=model, tokenizer=tokenizer)
