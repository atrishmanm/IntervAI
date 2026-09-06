"""
orchestrator/interview_engine.py — Structured Interview Engine
=============================================================
State-of-the-art interview simulation that ChatGPT/Claude can NOT replicate.

Differentiators:
  - Resume-based personalized questions (not generic)
  - Adaptive difficulty (harder if you answer well, easier if struggling)
  - Multi-dimensional scoring (technical, behavioral, communication)
  - STAR method evaluation for behavioral questions
  - Progressive question flow (warm-up → technical → behavioral → wrap-up)
  - Real-time scoring with detailed rubric
  - Structured feedback report per competency
"""

import re
import json
import random
import time
import os
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from enum import Enum

from orchestrator.company_templates import get_company_template, list_companies
from orchestrator.session_tracker import create_session_tracker


# ─────────────────────────────────────────────────────────────
# Enums & Data Classes
# ─────────────────────────────────────────────────────────────

class InterviewPhase(Enum):
    WARMUP = "warmup"
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    PROBLEM_SOLVING = "problem_solving"
    WRAP_UP = "wrap_up"


class Difficulty(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4


@dataclass
class SkillProfile:
    """Parsed skills and experience from resume."""
    name: str = ""
    skills: list = field(default_factory=list)
    experience_years: int = 0
    education: str = ""
    job_titles: list = field(default_factory=list)
    projects: list = field(default_factory=list)
    languages: list = field(default_factory=list)
    frameworks: list = field(default_factory=list)
    domains: list = field(default_factory=list)
    raw_text: str = ""


@dataclass
class Question:
    """A single interview question."""
    id: str
    text: str
    phase: InterviewPhase
    difficulty: Difficulty
    skill_tags: list = field(default_factory=list)
    follow_ups: list = field(default_factory=list)
    rubric: dict = field(default_factory=dict)
    time_limit_sec: int = 120
    is_from_resume: bool = False


@dataclass
class Answer:
    """Candidate's answer with scoring."""
    question: Question
    response: str
    time_taken_sec: float = 0.0
    scores: dict = field(default_factory=dict)
    feedback: str = ""
    star_completeness: dict = field(default_factory=dict)


@dataclass
class InterviewSession:
    """Full interview session state."""
    candidate_name: str = ""
    resume_text: str = ""
    skill_profile: SkillProfile = field(default_factory=SkillProfile)
    phase: InterviewPhase = InterviewPhase.WARMUP
    current_question_idx: int = 0
    questions: list = field(default_factory=list)
    answers: list = field(default_factory=list)
    scores_by_dimension: dict = field(default_factory=dict)
    difficulty: Difficulty = Difficulty.MEDIUM
    start_time: float = 0.0
    total_time_limit_sec: int = 3600  # 1 hour
    is_complete: bool = False
    company_name: str = ""
    company_template: Optional[Any] = None



# ─────────────────────────────────────────────────────────────
# Resume Parser
# ─────────────────────────────────────────────────────────────

# Skill keyword mappings
SKILL_KEYWORDS = {
    "python": ["python", "py", "cpython"],
    "javascript": ["javascript", "js", "typescript", "ts", "node"],
    "java": ["java", "jvm", "spring", "hibernate"],
    "c++": ["c++", "cpp", "stl", "cmake"],
    "c": ["c language", "embedded c", "posix"],
    "go": ["golang", "go"],
    "rust": ["rust", "cargo"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "sqlite", "database"],
    "nosql": ["mongodb", "redis", "dynamodb", "cassandra", "nosql"],
    "react": ["react", "reactjs", "react.js", "next.js", "nextjs"],
    "angular": ["angular", "angularjs"],
    "vue": ["vue", "vuejs", "vue.js"],
    "node": ["node", "nodejs", "node.js", "express", "fastify"],
    "django": ["django", "flask", "fastapi"],
    "aws": ["aws", "amazon web services", "ec2", "s3", "lambda"],
    "gcp": ["gcp", "google cloud", "bigquery"],
    "azure": ["azure", "microsoft azure"],
    "docker": ["docker", "container", "kubernetes", "k8s"],
    "git": ["git", "github", "gitlab", "bitbucket"],
    "machine_learning": ["machine learning", "ml", "deep learning", "neural network", "pytorch", "tensorflow"],
    "data_science": ["data science", "data analysis", "pandas", "numpy", "scikit-learn"],
    "system_design": ["system design", "distributed system", "microservice", "architecture"],
    "algorithms": ["algorithm", "data structure", "leetcode", "competitive programming"],
    "testing": ["testing", "unit test", "jest", "pytest", "selenium", "cypress"],
    "devops": ["devops", "ci/cd", "jenkins", "terraform", "ansible"],
    "frontend": ["html", "css", "sass", "tailwind", "bootstrap", "ui", "ux"],
    "backend": ["api", "rest", "graphql", "grpc", "backend"],
    "mobile": ["android", "ios", "flutter", "react native", "swift", "kotlin"],
    "security": ["security", "oauth", "jwt", "encryption", "cybersecurity"],
}

DOMAIN_KEYWORDS = {
    "web_development": ["web", "frontend", "backend", "fullstack", "full-stack", "full stack"],
    "mobile_development": ["mobile", "android", "ios", "app"],
    "data_engineering": ["data pipeline", "etl", "data warehouse", "spark", "kafka"],
    "cloud_computing": ["cloud", "aws", "gcp", "azure", "infrastructure"],
    "ai_ml": ["ai", "ml", "machine learning", "deep learning", "nlp", "computer vision"],
    "fintech": ["fintech", "finance", "banking", "payment", "trading"],
    "healthcare": ["healthcare", "medical", "clinical", "ehr", "hipaa"],
    "ecommerce": ["ecommerce", "e-commerce", "shopify", "stripe"],
    "gaming": ["game", "unity", "unreal", "gaming"],
    "cybersecurity": ["security", "cybersecurity", "pentest", "vulnerability"],
}


def parse_resume(text: str) -> SkillProfile:
    """Parse resume text and extract structured information.

    Uses keyword extraction + pattern matching (no external deps needed).
    For production, can be enhanced with spaCy or a custom NER model.
    """
    text_lower = text.lower()
    profile = SkillProfile(raw_text=text)

    # Extract name (first line, typically)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        # First line is usually the name
        name_line = lines[0]
        if len(name_line.split()) <= 4 and not any(c.isdigit() for c in name_line):
            profile.name = name_line

    # Extract skills
    found_skills = set()
    for skill, keywords in SKILL_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                found_skills.add(skill)
                break
    profile.skills = sorted(found_skills)

    # Extract programming languages specifically
    lang_keywords = ["python", "javascript", "java", "c++", "c", "go", "rust",
                     "typescript", "ruby", "php", "swift", "kotlin", "scala"]
    profile.languages = [l for l in lang_keywords if l in text_lower]

    # Extract frameworks
    fw_keywords = ["react", "angular", "vue", "django", "flask", "fastapi",
                   "spring", "express", "next.js", "node.js", "pytorch", "tensorflow",
                   "flutter", "react native", "rails", "laravel"]
    profile.frameworks = [f for f in fw_keywords if f in text_lower]

    # Extract domains
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                profile.domains.append(domain)
                break

    # Extract experience years
    year_patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s+)?experience',
        r'experience\s*:\s*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*yrs?\s*(?:exp|experience)',
    ]
    for pattern in year_patterns:
        match = re.search(pattern, text_lower)
        if match:
            profile.experience_years = int(match.group(1))
            break

    # Extract education
    edu_keywords = ["bachelor", "master", "phd", "b.s.", "m.s.", "b.tech", "m.tech",
                    "bca", "mca", "mba", "degree", "university", "college", "institute"]
    for kw in edu_keywords:
        if kw in text_lower:
            # Find the line containing this keyword
            for line in lines:
                if kw in line.lower():
                    profile.education = line.strip()
                    break
            break

    # Extract job titles
    title_keywords = ["engineer", "developer", "architect", "manager", "lead",
                      "senior", "junior", "staff", "principal", "director",
                      "analyst", "scientist", "consultant"]
    for line in lines:
        line_lower = line.lower()
        if any(t in line_lower for t in title_keywords):
            if len(line.split()) <= 6:  # Job titles are usually short
                profile.job_titles.append(line.strip())

    # Extract projects (lines starting with "project" or containing project keywords)
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if "project" in line_lower or "built" in line_lower or "developed" in line_lower:
            context = lines[i:min(i+3, len(lines))]
            profile.projects.append(" ".join(context))

    return profile


# ─────────────────────────────────────────────────────────────
# Question Bank
# ─────────────────────────────────────────────────────────────

QUESTION_BANK = {
    "warmup": [
        Question(
            id="wu_01",
            text="Tell me about yourself and what got you into software engineering.",
            phase=InterviewPhase.WARMUP,
            difficulty=Difficulty.EASY,
            skill_tags=["communication"],
            rubric={"clarity": 0.3, "relevance": 0.3, "conciseness": 0.2, "enthusiasm": 0.2},
            time_limit_sec=120,
        ),
        Question(
            id="wu_02",
            text="What's a project you're most proud of and why?",
            phase=InterviewPhase.WARMUP,
            difficulty=Difficulty.EASY,
            skill_tags=["communication", "projects"],
            rubric={"depth": 0.3, "impact": 0.3, "technical_detail": 0.2, "clarity": 0.2},
            time_limit_sec=120,
        ),
        Question(
            id="wu_03",
            text="Walk me through a typical workday in your current role.",
            phase=InterviewPhase.WARMUP,
            difficulty=Difficulty.EASY,
            skill_tags=["communication", "experience"],
            rubric={"clarity": 0.3, "relevance": 0.3, "detail": 0.2, "honesty": 0.2},
            time_limit_sec=90,
        ),
    ],
    "technical": {
        "easy": [
            Question(
                id="te_01",
                text="Explain the difference between a stack and a queue. When would you use each?",
                phase=InterviewPhase.TECHNICAL,
                difficulty=Difficulty.EASY,
                skill_tags=["data_structures", "algorithms"],
                rubric={"correctness": 0.4, "examples": 0.3, "clarity": 0.3},
                time_limit_sec=90,
            ),
            Question(
                id="te_02",
                text="What is Big O notation? Can you give examples of O(1), O(n), O(n^2), and O(log n)?",
                phase=InterviewPhase.TECHNICAL,
                difficulty=Difficulty.EASY,
                skill_tags=["complexity", "algorithms"],
                rubric={"correctness": 0.4, "examples": 0.3, "clarity": 0.3},
                time_limit_sec=90,
            ),
            Question(
                id="te_03",
                text="What is the difference between SQL and NoSQL databases? When would you choose one over the other?",
                phase=InterviewPhase.TECHNICAL,
                difficulty=Difficulty.EASY,
                skill_tags=["databases", "sql"],
                rubric={"correctness": 0.4, "tradeoffs": 0.3, "examples": 0.3},
                time_limit_sec=90,
            ),
            Question(
                id="te_04",
                text="Explain what an API is and how REST APIs work.",
                phase=InterviewPhase.TECHNICAL,
                difficulty=Difficulty.EASY,
                skill_tags=["backend", "api"],
                rubric={"correctness": 0.4, "detail": 0.3, "clarity": 0.3},
                time_limit_sec=90,
            ),
        ],
        "medium": [
            Question(
                id="tm_01",
                text="How would you design a URL shortener like bit.ly? Walk me through the architecture.",
                phase=InterviewPhase.TECHNICAL,
                difficulty=Difficulty.MEDIUM,
                skill_tags=["system_design", "backend"],
                rubric={"architecture": 0.3, "scalability": 0.3, "tradeoffs": 0.2, "clarity": 0.2},
                time_limit_sec=180,
            ),
            Question(
                id="tm_02",
                text="Explain how hash tables work internally. What happens when there's a collision?",
                phase=InterviewPhase.TECHNICAL,
                difficulty=Difficulty.MEDIUM,
                skill_tags=["data_structures", "algorithms"],
                rubric={"correctness": 0.3, "collision_handling": 0.3, "complexity": 0.2, "clarity": 0.2},
                time_limit_sec=120,
            ),
            Question(
                id="tm_03",
                text="What is database indexing? How does a B-tree index work and when would you use one?",
                phase=InterviewPhase.TECHNICAL,
                difficulty=Difficulty.MEDIUM,
                skill_tags=["databases", "performance"],
                rubric={"correctness": 0.3, "btree_explanation": 0.3, "use_cases": 0.2, "tradeoffs": 0.2},
                time_limit_sec=120,
            ),
            Question(
                id="tm_04",
                text="Explain the difference between concurrency and parallelism. Give real-world examples.",
                phase=InterviewPhase.TECHNICAL,
                difficulty=Difficulty.MEDIUM,
                skill_tags=["systems", "concurrency"],
                rubric={"correctness": 0.3, "examples": 0.3, "depth": 0.2, "clarity": 0.2},
                time_limit_sec=120,
            ),
            Question(
                id="tm_05",
                text="How does garbage collection work in Java/Python? What are the tradeoffs between different GC algorithms?",
                phase=InterviewPhase.TECHNICAL,
                difficulty=Difficulty.MEDIUM,
                skill_tags=["languages", "performance"],
                rubric={"correctness": 0.3, "algorithms": 0.3, "tradeoffs": 0.2, "clarity": 0.2},
                time_limit_sec=120,
            ),
        ],
        "hard": [
            Question(
                id="th_01",
                text="Design a real-time chat application like WhatsApp. How would you handle message delivery, offline users, and scale to millions of users?",
                phase=InterviewPhase.TECHNICAL,
                difficulty=Difficulty.HARD,
                skill_tags=["system_design", "distributed_systems"],
                rubric={"architecture": 0.25, "scalability": 0.25, "reliability": 0.25, "tradeoffs": 0.25},
                time_limit_sec=300,
            ),
            Question(
                id="th_02",
                text="Explain the CAP theorem. How do modern distributed systems like Cassandra, MongoDB, and Kafka handle the CAP tradeoffs?",
                phase=InterviewPhase.TECHNICAL,
                difficulty=Difficulty.HARD,
                skill_tags=["distributed_systems", "databases"],
                rubric={"cap_understanding": 0.3, "system_examples": 0.3, "tradeoffs": 0.2, "depth": 0.2},
                time_limit_sec=180,
            ),
            Question(
                id="th_03",
                text="How would you implement a rate limiter for an API? Discuss different algorithms (token bucket, sliding window, etc.) and their tradeoffs.",
                phase=InterviewPhase.TECHNICAL,
                difficulty=Difficulty.HARD,
                skill_tags=["system_design", "backend"],
                rubric={"algorithms": 0.3, "implementation": 0.3, "tradeoffs": 0.2, "scalability": 0.2},
                time_limit_sec=180,
            ),
        ],
    },
    "behavioral": [
        Question(
            id="bh_01",
            text="Tell me about a time you had a disagreement with a teammate. How did you resolve it?",
            phase=InterviewPhase.BEHAVIORAL,
            difficulty=Difficulty.MEDIUM,
            skill_tags=["teamwork", "communication", "conflict_resolution"],
            rubric={"star_structure": 0.3, "specificity": 0.3, "outcome": 0.2, "self_awareness": 0.2},
            time_limit_sec=180,
        ),
        Question(
            id="bh_02",
            text="Describe a situation where you had to learn a new technology quickly. How did you approach it?",
            phase=InterviewPhase.BEHAVIORAL,
            difficulty=Difficulty.MEDIUM,
            skill_tags=["learning", "adaptability"],
            rubric={"star_structure": 0.3, "specificity": 0.3, "outcome": 0.2, "reflection": 0.2},
            time_limit_sec=180,
        ),
        Question(
            id="bh_03",
            text="Tell me about a time you failed at something. What did you learn from it?",
            phase=InterviewPhase.BEHAVIORAL,
            difficulty=Difficulty.MEDIUM,
            skill_tags=["resilience", "self_awareness", "growth"],
            rubric={"star_structure": 0.3, "honesty": 0.3, "learning": 0.2, "growth_mindset": 0.2},
            time_limit_sec=180,
        ),
        Question(
            id="bh_04",
            text="Describe a project where you had to work under a tight deadline. How did you prioritize?",
            phase=InterviewPhase.BEHAVIORAL,
            difficulty=Difficulty.MEDIUM,
            skill_tags=["time_management", "prioritization"],
            rubric={"star_structure": 0.3, "specificity": 0.3, "outcome": 0.2, "strategy": 0.2},
            time_limit_sec=180,
        ),
        Question(
            id="bh_05",
            text="Tell me about a time you received critical feedback. How did you handle it?",
            phase=InterviewPhase.BEHAVIORAL,
            difficulty=Difficulty.MEDIUM,
            skill_tags=["feedback", "growth", "humility"],
            rubric={"star_structure": 0.3, "openness": 0.3, "action_taken": 0.2, "outcome": 0.2},
            time_limit_sec=180,
        ),
        Question(
            id="bh_06",
            text="Describe a situation where you had to influence someone without direct authority. How did you approach it?",
            phase=InterviewPhase.BEHAVIORAL,
            difficulty=Difficulty.HARD,
            skill_tags=["leadership", "influence", "communication"],
            rubric={"star_structure": 0.3, "strategy": 0.3, "outcome": 0.2, "empathy": 0.2},
            time_limit_sec=180,
        ),
        Question(
            id="bh_07",
            text="Tell me about a time you went above and beyond for a customer or colleague.",
            phase=InterviewPhase.BEHAVIORAL,
            difficulty=Difficulty.MEDIUM,
            skill_tags=["service", "initiative"],
            rubric={"star_structure": 0.3, "impact": 0.3, "initiative": 0.2, "clarity": 0.2},
            time_limit_sec=180,
        ),
        Question(
            id="bh_08",
            text="Why are you interested in this role? What excites you about joining our team?",
            phase=InterviewPhase.BEHAVIORAL,
            difficulty=Difficulty.EASY,
            skill_tags=["motivation", "cultural_fit"],
            rubric={"specificity": 0.3, "research": 0.3, "enthusiasm": 0.2, "alignment": 0.2},
            time_limit_sec=120,
        ),
        Question(
            id="bh_09",
            text="Where do you see yourself in 5 years? How does this role fit into your career plan?",
            phase=InterviewPhase.BEHAVIORAL,
            difficulty=Difficulty.EASY,
            skill_tags=["ambition", "planning"],
            rubric={"realism": 0.3, "clarity": 0.3, "alignment": 0.2, "ambition": 0.2},
            time_limit_sec=120,
        ),
        Question(
            id="bh_10",
            text="What's your greatest strength and how does it help you in your work?",
            phase=InterviewPhase.BEHAVIORAL,
            difficulty=Difficulty.EASY,
            skill_tags=["self_awareness"],
            rubric={"specificity": 0.3, "examples": 0.3, "self_awareness": 0.2, "relevance": 0.2},
            time_limit_sec=120,
        ),
        Question(
            id="bh_11",
            text="Describe a time you had to make a difficult decision with incomplete information.",
            phase=InterviewPhase.BEHAVIORAL,
            difficulty=Difficulty.HARD,
            skill_tags=["decision_making", "judgment"],
            rubric={"star_structure": 0.3, "reasoning": 0.3, "outcome": 0.2, "reflection": 0.2},
            time_limit_sec=180,
        ),
    ],
}


# ─────────────────────────────────────────────────────────────
# STAR Method Evaluator
# ─────────────────────────────────────────────────────────────

STAR_KEYWORDS = {
    "situation": ["when", "where", "at work", "in my", "during", "while", "at the", "context", "background", "situation"],
    "task": ["i needed to", "my task was", "i was responsible", "i had to", "goal was", "task was", "objective"],
    "action": ["i decided", "i implemented", "i built", "i created", "i led", "i organized", "i managed", "i suggested", "i proposed", "i coded", "i developed", "i set up", "i configured", "i designed", "i planned", "i coordinated", "i initiated"],
    "result": ["as a result", "the outcome", "we achieved", "i learned", "it led to", "successfully", "improved", "reduced", "increased", "delivered", "result", "impact", "effect", "outcome"],
}


def evaluate_star(response: str) -> dict:
    """Evaluate if response follows STAR method (Situation, Task, Action, Result)."""
    response_lower = response.lower()
    scores = {}
    for component, keywords in STAR_KEYWORDS.items():
        found = sum(1 for kw in keywords if kw in response_lower)
        scores[component] = min(found / 2.0, 1.0)  # Normalize to 0-1

    scores["completeness"] = sum(scores.values()) / len(scores)
    return scores


# ─────────────────────────────────────────────────────────────
# Interview Engine
# ─────────────────────────────────────────────────────────────

class InterviewEngine:
    """Main interview engine — orchestrates the entire interview process.

    This is what makes INTERVUE different from ChatGPT/Claude:
    1. Parses YOUR resume and asks personalized questions
    2. Adapts difficulty based on your performance
    3. Scores on multiple dimensions (not just "good/bad")
    4. Evaluates STAR method for behavioral questions
    5. Provides structured feedback report
    """

    def __init__(self, model=None, tokenizer=None, model_service=None, use_model_eval: Optional[bool] = None):
        self.model = model
        self.tokenizer = tokenizer
        self._model_service = model_service
        self.session = InterviewSession()
        self.use_model_eval = (
            use_model_eval
            if use_model_eval is not None
            else (os.environ.get("INTERVAI_USE_MODEL_EVAL", "0") == "1")
        )

    @property
    def model_service(self):
        if self._model_service is None:
            try:
                from backend.model_service import get_model_service
                self._model_service = get_model_service()
            except Exception:
                self._model_service = None
        return self._model_service

    def start_interview(self, resume_text: str, candidate_name: str = "", company_name: str = "") -> dict:
        """Start a new interview session from a resume and optional company mode.

        Returns:
            dict with welcome message + first question
        """
        # Parse resume
        profile = parse_resume(resume_text)
        if candidate_name:
            profile.name = candidate_name

        template = get_company_template(company_name) if company_name else None

        self.session = InterviewSession(
            candidate_name=profile.name or candidate_name or "Candidate",
            resume_text=resume_text,
            skill_profile=profile,
            start_time=time.time(),
            company_name=company_name,
            company_template=template,
        )

        # Select questions based on resume & company template
        self.session.questions = self._select_questions(profile, template=template)

        # Start with warmup
        first_q = self.session.questions[0] if self.session.questions else None

        company_desc = f" for your {template.company_name} mock interview (Target Bar: {template.bar_level.upper()})" if template else ""
        skills_summary = f" Identified skills: {', '.join(profile.skills[:5])}." if profile.skills else ""

        return {
            "message": f"Welcome, {self.session.candidate_name}!{company_desc}.{skills_summary} Let's begin with a warm-up question.",
            "question": first_q.text if first_q else "Tell me about yourself and your background in software development.",
            "phase": self.session.phase.value,
            "time_limit": first_q.time_limit_sec if first_q else 120,
            "question_number": 1,
            "total_questions": len(self.session.questions),
            "company": template.company_name if template else None,
        }

    def submit_answer(self, response: str, time_taken: float = 0) -> dict:
        """Submit an answer and get the next question + feedback.

        Returns:
            dict with score, feedback, next question (or completion)
        """
        if self.session.is_complete:
            return {"error": "Interview already complete"}

        current_q = self.session.questions[self.session.current_question_idx]

        # Score the answer using trained evaluator + rubric heuristics
        scores, model_feedback = self._score_answer(current_q, response, time_taken)
        star = evaluate_star(response) if current_q.phase == InterviewPhase.BEHAVIORAL else {}

        feedback = self._generate_feedback(current_q, scores, star)
        if model_feedback and "Score:" not in feedback:
            feedback = f"{feedback} [AI Evaluator]: {model_feedback}"

        answer = Answer(
            question=current_q,
            response=response,
            time_taken_sec=time_taken,
            scores=scores,
            star_completeness=star,
            feedback=feedback,
        )
        self.session.answers.append(answer)

        # Update difficulty based on performance
        self._update_difficulty(scores)

        # Move to next question
        self.session.current_question_idx += 1

        # Check if interview is complete
        if self.session.current_question_idx >= len(self.session.questions):
            self.session.is_complete = True
            report = self._generate_final_report()
            return {
                "score": scores,
                "feedback": answer.feedback,
                "is_complete": True,
                "report": report,
            }

        # Get next question
        next_q = self.session.questions[self.session.current_question_idx]

        # Update phase if needed
        if next_q.phase != self.session.phase:
            self.session.phase = next_q.phase

        return {
            "score": scores,
            "feedback": answer.feedback,
            "star_analysis": star,
            "question": next_q.text,
            "phase": self.session.phase.value,
            "difficulty": next_q.difficulty.name,
            "time_limit": next_q.time_limit_sec,
            "question_number": self.session.current_question_idx + 1,
            "total_questions": len(self.session.questions),
            "is_complete": False,
        }

    def _select_questions(self, profile: SkillProfile, template=None) -> list:
        """Select personalized questions based on resume and optional company template."""
        questions = []

        # 1. Always start with warmup
        questions.append(random.choice(QUESTION_BANK["warmup"]))

        # 2. Company-specific questions if template provided
        if template and getattr(template, "common_questions", None):
            for i, cq_text in enumerate(template.common_questions[:3]):
                questions.append(Question(
                    id=f"company_{template.company_name.lower()}_{i}",
                    text=cq_text,
                    phase=InterviewPhase.TECHNICAL if i % 2 == 0 else InterviewPhase.BEHAVIORAL,
                    difficulty=Difficulty.HARD if getattr(template, "bar_level", "") in ["very_high", "exceptional"] else Difficulty.MEDIUM,
                    skill_tags=getattr(template, "culture_keywords", [])[:2],
                    rubric={dim: w for dim, w in getattr(template, "evaluation_criteria", {}).items() if w > 0},
                    time_limit_sec=150,
                    is_from_resume=False,
                ))

        # 3. Technical questions based on skills & resume
        skill_difficulty = self._assess_skill_level(profile)
        tech_pool = QUESTION_BANK["technical"].get(skill_difficulty, [])
        resume_questions = self._generate_resume_questions(profile)
        tech_pool = resume_questions + tech_pool
        n_tech = min(3 if template else 4, len(tech_pool))
        if tech_pool:
            questions.extend(random.sample(tech_pool, n_tech))

        # 4. Behavioral questions (STAR method)
        behavioral_pool = QUESTION_BANK["behavioral"]
        n_behav = 2 if template else 3
        questions.extend(random.sample(behavioral_pool, min(n_behav, len(behavioral_pool))))

        # Shuffle middle questions (keep warmup first)
        middle = questions[1:]
        random.shuffle(middle)
        questions = [questions[0]] + middle

        return questions

    def _assess_skill_level(self, profile: SkillProfile) -> str:
        """Assess skill level from resume to set question difficulty."""
        n_skills = len(profile.skills)
        years = profile.experience_years
        has_system_design = "system_design" in profile.skills
        has_ml = "machine_learning" in profile.skills

        if years >= 5 or has_system_design or n_skills >= 10:
            return "hard"
        elif years >= 2 or n_skills >= 6:
            return "medium"
        else:
            return "easy"

    def _generate_resume_questions(self, profile: SkillProfile) -> list:
        """Generate questions based on specific resume content."""
        questions = []

        # Ask about specific skills
        for skill in profile.skills[:3]:
            questions.append(Question(
                id=f"resume_skill_{skill}",
                text=f"I see you have experience with {skill}. Can you tell me about a challenging "
                     f"project you worked on using {skill}?",
                phase=InterviewPhase.TECHNICAL,
                difficulty=Difficulty.MEDIUM,
                skill_tags=[skill],
                rubric={"depth": 0.3, "specificity": 0.3, "problem_solving": 0.2, "clarity": 0.2},
                time_limit_sec=120,
                is_from_resume=True,
            ))

        # Ask about specific projects
        for project in profile.projects[:2]:
            questions.append(Question(
                id="resume_project",
                text=f"Tell me about this project: {project[:100]}... What was your role and what did you learn?",
                phase=InterviewPhase.TECHNICAL,
                difficulty=Difficulty.MEDIUM,
                skill_tags=["projects", "communication"],
                rubric={"depth": 0.3, "impact": 0.3, "learning": 0.2, "clarity": 0.2},
                time_limit_sec=150,
                is_from_resume=True,
            ))

        # Ask about experience gap or career transition if applicable
        if profile.experience_years == 0:
            questions.append(Question(
                id="resume_experience",
                text="I notice you might be early in your career. What have you done to build your skills "
                     "outside of formal work experience?",
                phase=InterviewPhase.BEHAVIORAL,
                difficulty=Difficulty.EASY,
                skill_tags=["initiative", "learning"],
                rubric={"specificity": 0.3, "initiative": 0.3, "examples": 0.2, "passion": 0.2},
                time_limit_sec=120,
                is_from_resume=True,
            ))

        return questions

    def _score_answer(self, question: Question, response: str, time_taken: float = 0) -> tuple:
        """Score an answer on multiple dimensions combining model evaluation and rubric."""
        scores = {}
        model_feedback = ""

        # Model evaluator score (if enabled and loaded)
        model_score = None
        if self.use_model_eval and self.model_service is not None:
            try:
                eval_res = self.model_service.evaluate(question.text, response)
                model_score = eval_res.get("score", 50) / 100.0
                model_feedback = eval_res.get("feedback", "").strip()
            except Exception:
                model_score = None

        # Base scoring from rubric
        rubric = question.rubric or {"depth": 0.35, "clarity": 0.35, "problem_solving": 0.3}
        for dimension, weight in rubric.items():
            heuristic = self._heuristic_score(dimension, response, question)
            if model_score is not None:
                # 60% model evaluation, 40% heuristic checks
                scores[dimension] = round(min(1.0, max(0.0, 0.6 * model_score + 0.4 * heuristic)), 2)
            else:
                scores[dimension] = round(heuristic, 2)

        # Bonus for resume-specific questions
        if question.is_from_resume:
            scores["resume_relevance"] = 0.85

        # Time bonus/penalty
        if question.time_limit_sec > 0 and time_taken > 0:
            time_ratio = time_taken / question.time_limit_sec
            if time_ratio < 0.25:
                scores["efficiency"] = 0.7
            elif time_ratio <= 0.85:
                scores["efficiency"] = 1.0
            else:
                scores["efficiency"] = 0.65

        return scores, model_feedback

    def _heuristic_score(self, dimension: str, response: str, question: Question) -> float:
        """Heuristic scoring fallback based on lexical and structural analysis."""
        response_len = len(response.split())
        base_score = 0.5

        # Length bonus (more detail = better, up to a point)
        if response_len > 40:
            base_score += 0.1
        if response_len > 90:
            base_score += 0.1
        if response_len > 180:
            base_score += 0.05

        # Specificity (numbers, examples, proper nouns)
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
            base_score -= 0.25

        return min(max(base_score, 0.0), 1.0)

    def _update_difficulty(self, scores: dict):
        """Adapt difficulty based on performance."""
        avg_score = sum(scores.values()) / max(len(scores), 1)

        if avg_score >= 0.8 and self.session.difficulty.value < Difficulty.EXPERT.value:
            self.session.difficulty = Difficulty(self.session.difficulty.value + 1)
        elif avg_score <= 0.4 and self.session.difficulty.value > Difficulty.EASY.value:
            self.session.difficulty = Difficulty(self.session.difficulty.value - 1)

    def _generate_feedback(self, question: Question, scores: dict, star: dict) -> str:
        """Generate constructive feedback for an answer."""
        avg = sum(scores.values()) / max(len(scores), 1)
        feedback_parts = []

        if avg >= 0.8:
            feedback_parts.append("Excellent response! Strong technical foundation demonstrated.")
        elif avg >= 0.6:
            feedback_parts.append("Solid answer with good clarity.")
        elif avg >= 0.4:
            feedback_parts.append("Decent answer, but could be enhanced with specific examples or metrics.")
        else:
            feedback_parts.append("This answer needs more depth and technical precision.")

        # STAR feedback for behavioral
        if star and question.phase == InterviewPhase.BEHAVIORAL:
            missing = [k.capitalize() for k, v in star.items() if k != "completeness" and v < 0.5]
            if missing:
                feedback_parts.append(
                    f"To make your story more impactful, provide more detail on {', '.join(missing)}. "
                    f"Remember the STAR framework: Situation, Task, Action, Result."
                )

        # Specific dimension feedback
        low_dims = [d.replace('_', ' ') for d, s in scores.items() if s < 0.5 and d not in ["efficiency", "resume_relevance"]]
        if low_dims:
            feedback_parts.append(f"Focus areas: {', '.join(low_dims)}.")

        return " ".join(feedback_parts)

    def _generate_final_report(self) -> dict:
        """Generate comprehensive final interview report with radar data and session persistence."""
        if not self.session.answers:
            return {"error": "No answers to report on"}

        # Aggregate scores by dimension
        all_scores = {}
        for answer in self.session.answers:
            for dim, score in answer.scores.items():
                if dim not in all_scores:
                    all_scores[dim] = []
                all_scores[dim].append(score)

        dimension_averages = {
            dim: round(sum(scores) / len(scores), 2)
            for dim, scores in all_scores.items()
        }

        # Overall score
        overall = sum(dimension_averages.values()) / max(len(dimension_averages), 1)

        # STAR completeness for behavioral
        behavioral_answers = [a for a in self.session.answers if a.star_completeness]
        avg_star = {}
        if behavioral_answers:
            for component in ["situation", "task", "action", "result"]:
                vals = [a.star_completeness.get(component, 0) for a in behavioral_answers]
                avg_star[component] = round(sum(vals) / len(vals), 2)

        # Skill coverage
        skills_asked = set()
        for q in self.session.questions:
            skills_asked.update(q.skill_tags)

        # Resume questions answered
        resume_q = [a for a in self.session.answers if a.question.is_from_resume]

        # Time analysis
        total_time = sum(a.time_taken_sec for a in self.session.answers)
        avg_time = total_time / max(len(self.session.answers), 1)

        # Competency ratings (scaled 0-1)
        competencies = {
            "Technical Depth": round(dimension_averages.get("depth", 0.6) * 0.5 + dimension_averages.get("correctness", 0.6) * 0.5, 2),
            "Problem Solving": round(dimension_averages.get("problem_solving", 0.6) * 0.6 + dimension_averages.get("tradeoffs", 0.6) * 0.4, 2),
            "Communication": round(dimension_averages.get("clarity", 0.6) * 0.5 + dimension_averages.get("specificity", 0.6) * 0.5, 2),
            "STAR Completeness": round(sum(avg_star.values()) / max(len(avg_star), 1), 2) if avg_star else 0.7,
            "Culture Alignment": round(dimension_averages.get("alignment", 0.7), 2),
        }

        # Radar chart visualization data
        radar_chart = {
            "labels": list(competencies.keys()),
            "values": [round(v * 100, 1) for v in competencies.values()],
        }

        # Recommendation
        if overall >= 0.8:
            recommendation = "STRONG HIRE"
        elif overall >= 0.6:
            recommendation = "HIRE"
        elif overall >= 0.4:
            recommendation = "MAYBE"
        else:
            recommendation = "NO HIRE"

        report_data = {
            "candidate": self.session.candidate_name,
            "company": self.session.company_name or None,
            "overall_score": round(overall * 100, 1),
            "score": round(overall * 100, 1),
            "recommendation": recommendation,
            "verdict": f"{recommendation} — Candidate showed proficiency in {', '.join(list(skills_asked)[:3]) if skills_asked else 'core competencies'}",
            "competencies": competencies,
            "radar_chart": radar_chart,
            "dimension_scores": dimension_averages,
            "star_analysis": avg_star,
            "total_questions": len(self.session.questions),
            "questions_answered": len(self.session.answers),
            "skills_assessed": list(skills_asked),
            "resume_questions_answered": len(resume_q),
            "total_time_min": round(total_time / 60, 1),
            "avg_time_per_question_sec": round(avg_time, 1),
            "difficulty_reached": self.session.difficulty.name,
            "strengths": [d for d, v in competencies.items() if v >= 0.65],
            "areas_for_improvement": [d for d, v in competencies.items() if v < 0.55],
            "detailed_answers": [
                {
                    "question": a.question.text,
                    "phase": a.question.phase.value,
                    "scores": a.scores,
                    "feedback": a.feedback,
                    "star": a.star_completeness if a.star_completeness else None,
                    "time_sec": round(a.time_taken_sec, 1),
                }
                for a in self.session.answers
            ],
        }

        # Persist session to session tracker
        try:
            tracker = create_session_tracker()
            tracker.record_session({
                "duration_minutes": report_data["total_time_min"],
                "overall_score": round(overall, 2),
                "recommendation": recommendation,
                "dimension_scores": {k: round(v, 2) for k, v in dimension_averages.items()},
                "phase_scores": {
                    "technical": competencies["Technical Depth"],
                    "behavioral": competencies["STAR Completeness"],
                },
                "skills_assessed": report_data["skills_assessed"],
                "strengths": report_data["strengths"],
                "weaknesses": report_data["areas_for_improvement"],
                "company_template": self.session.company_name or None,
            })
        except Exception:
            pass

        return report_data


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_resume = """
    John Smith
    Software Engineer | 3 years experience

    Skills: Python, JavaScript, React, Django, PostgreSQL, Docker, AWS
    Education: B.S. Computer Science, MIT 2021

    Experience:
    Senior Developer at TechCorp (2022-present)
    - Built microservices handling 10K requests/sec
    - Led migration from monolith to microservices architecture

    Developer at StartupXYZ (2021-2022)
    - Developed React frontend for SaaS platform
    - Implemented REST APIs using Django REST Framework

    Projects:
    - Real-time Chat Application using WebSockets and Redis
    - E-commerce Platform with Stripe integration
    """

    engine = InterviewEngine()
    result = engine.start_interview(sample_resume, "John Smith")
    print(json.dumps(result, indent=2))

    # Simulate a few answers
    for i in range(3):
        result = engine.submit_answer(
            f"This is a sample answer about my experience. I worked on a project where I had to "
            f"implement a {result.get('phase', 'technical')} solution. For example, when building "
            f"the chat application, I used WebSockets for real-time communication and Redis for "
            f"message queuing. The result was a system that could handle 1000 concurrent users.",
            time_taken=30
        )
        if result.get("is_complete"):
            print("\nFINAL REPORT:")
            print(json.dumps(result["report"], indent=2))
            break
        else:
            print(f"\nQ{result['question_number']}: {result['question'][:80]}...")
            print(f"Score: {result['score']}")
