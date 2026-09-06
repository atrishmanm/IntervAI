"""
orchestrator/company_templates.py
=================================
Company-specific interview patterns for FAANG, startups, and more.

Each company template defines:
1. Interview structure (rounds, durations, focus areas)
2. Question style (what they prioritize)
3. Evaluation criteria (what they value)
4. Difficulty calibration (their bar level)
5. Common question banks
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class CompanyTier(Enum):
    FAANG = "faang"
    BIG_TECH = "big_tech"
    UNICORN = "unicorn"
    STARTUP = "startup"
    ENTERPRISE = "enterprise"
    CONSULTING = "consulting"


@dataclass
class InterviewRound:
    """A single interview round."""
    name: str
    duration_minutes: int
    focus_areas: List[str]
    question_types: List[str]
    difficulty_modifier: float  # 1.0 = normal, 1.2 = harder, 0.8 = easier
    max_questions: int


@dataclass
class CompanyTemplate:
    """Complete interview template for a company."""
    company_name: str
    tier: CompanyTier
    description: str
    rounds: List[InterviewRound]
    evaluation_criteria: Dict[str, float]  # dimension -> weight
    question_style: str  # "behavioral_heavy", "technical_heavy", "balanced"
    culture_keywords: List[str]
    bar_level: str  # "high", "very_high", "exceptional"
    common_questions: List[str]
    what_they_value: List[str]


# ─────────────────────────────────────────────────────────────
# FAANG Templates
# ─────────────────────────────────────────────────────────────

GOOGLE_TEMPLATE = CompanyTemplate(
    company_name="Google",
    tier=CompanyTier.FAANG,
    description="Google is known for its rigorous technical interviews with heavy emphasis on algorithms, data structures, system design, and Googleyness (culture fit).",
    rounds=[
        InterviewRound(
            name="Phone Screen",
            duration_minutes=45,
            focus_areas=["algorithms", "data_structures"],
            question_types=["coding", "problem_solving"],
            difficulty_modifier=1.0,
            max_questions=2,
        ),
        InterviewRound(
            name="Onsite Technical 1",
            duration_minutes=60,
            focus_areas=["algorithms", "data_structures", "optimization"],
            question_types=["coding", "system_design"],
            difficulty_modifier=1.2,
            max_questions=2,
        ),
        InterviewRound(
            name="Onsite Technical 2",
            duration_minutes=60,
            focus_areas=["system_design", "scalability"],
            question_types=["system_design", "architecture"],
            difficulty_modifier=1.3,
            max_questions=1,
        ),
        InterviewRound(
            name="Googleyness & Leadership",
            duration_minutes=45,
            focus_areas=["leadership", "ambiguity", "collaboration"],
            question_types=["behavioral", "situational"],
            difficulty_modifier=1.0,
            max_questions=4,
        ),
    ],
    evaluation_criteria={
        "technical_depth": 0.35,
        "problem_solving": 0.25,
        "communication": 0.15,
        "leadership": 0.15,
        "culture_fit": 0.10,
    },
    question_style="technical_heavy",
    culture_keywords=["googleyness", "intellectual humility", "bias to action", "user focus"],
    bar_level="very_high",
    common_questions=[
        "Tell me about a time you had to make a decision with incomplete information.",
        "Describe a project where you had to navigate ambiguity.",
        "How do you handle disagreements with senior engineers?",
        "Tell me about a time you failed and what you learned.",
        "Design a system like Google Maps.",
    ],
    what_they_value=[
        "Algorithmic thinking and optimization",
        "Ability to handle ambiguity",
        "Collaboration across teams",
        "Bias to action with intellectual humility",
        "User-focused mindset",
    ],
)

META_TEMPLATE = CompanyTemplate(
    company_name="Meta (Facebook)",
    tier=CompanyTier.FAANG,
    description="Meta focuses on building social experiences at scale. Interviews emphasize coding, system design, and move-fast culture.",
    rounds=[
        InterviewRound(
            name="Phone Screen",
            duration_minutes=45,
            focus_areas=["coding", "algorithms"],
            question_types=["coding"],
            difficulty_modifier=1.0,
            max_questions=2,
        ),
        InterviewRound(
            name="Coding Round",
            duration_minutes=60,
            focus_areas=["coding", "data_structures"],
            question_types=["coding", "problem_solving"],
            difficulty_modifier=1.1,
            max_questions=2,
        ),
        InterviewRound(
            name="System Design",
            duration_minutes=60,
            focus_areas=["system_design", "scalability", "real_time"],
            question_types=["system_design"],
            difficulty_modifier=1.2,
            max_questions=1,
        ),
        InterviewRound(
            name="Behavioral",
            duration_minutes=45,
            focus_areas=["move_fast", "impact", "collaboration"],
            question_types=["behavioral"],
            difficulty_modifier=1.0,
            max_questions=4,
        ),
    ],
    evaluation_criteria={
        "technical_depth": 0.30,
        "problem_solving": 0.25,
        "communication": 0.15,
        "impact": 0.20,
        "culture_fit": 0.10,
    },
    question_style="balanced",
    culture_keywords=["move fast", "impact", "boldness", "openness", "build social value"],
    bar_level="very_high",
    common_questions=[
        "Tell me about a time you moved fast and broke things.",
        "Describe your most impactful project.",
        "How do you prioritize features when resources are limited?",
        "Design a news feed system like Facebook.",
        "Tell me about a time you had to make a quick decision.",
    ],
    what_they_value=[
        "Move-fast mentality with quality",
        "Impact-driven development",
        "Bold technical decisions",
        "Collaboration in fast-paced environments",
        "Building at scale",
    ],
)

AMAZON_TEMPLATE = CompanyTemplate(
    company_name="Amazon",
    tier=CompanyTier.FAANG,
    description="Amazon is famous for Leadership Principles. Every question maps to one or more LPs. Technical skills matter but leadership is paramount.",
    rounds=[
        InterviewRound(
            name="Phone Screen",
            duration_minutes=45,
            focus_areas=["coding", "leadership"],
            question_types=["coding", "behavioral"],
            difficulty_modifier=1.0,
            max_questions=2,
        ),
        InterviewRound(
            name="Online Assessment",
            duration_minutes=120,
            focus_areas=["coding", "problem_solving"],
            question_types=["coding"],
            difficulty_modifier=1.0,
            max_questions=2,
        ),
        InterviewRound(
            name="Onsite Loop (4-5 rounds)",
            duration_minutes=60,
            focus_areas=["leadership_principles", "system_design", "coding"],
            question_types=["behavioral", "system_design", "coding"],
            difficulty_modifier=1.2,
            max_questions=3,
        ),
        InterviewRound(
            name="Bar Raiser",
            duration_minutes=60,
            focus_areas=["leadership", "culture", "long_term"],
            question_types=["behavioral", "situational"],
            difficulty_modifier=1.3,
            max_questions=4,
        ),
    ],
    evaluation_criteria={
        "leadership_principles": 0.35,
        "technical_depth": 0.25,
        "problem_solving": 0.20,
        "communication": 0.10,
        "culture_fit": 0.10,
    },
    question_style="behavioral_heavy",
    culture_keywords=[
        "customer obsession", "ownership", "invent and simplify", "are right a lot",
        "learn and be curious", "hire and develop the best", "insist on the highest standards",
        "think big", "bias for action", "frugality", "earn trust", "dive deep",
        "have backbone; disagree and commit", "deliver results",
    ],
    bar_level="very_high",
    common_questions=[
        "Tell me about a time you went above and beyond for a customer.",
        "Describe a time you had to make a decision without enough data.",
        "Tell me about a time you disagreed with your manager.",
        "Give me an example of when you took ownership of something.",
        "Tell me about a time you simplified a complex process.",
    ],
    what_they_value=[
        "Customer obsession above all",
        "Ownership and accountability",
        "Bias for action with good judgment",
        "Insisting on the highest standards",
        "Diving deep into details",
    ],
)

APPLE_TEMPLATE = CompanyTemplate(
    company_name="Apple",
    tier=CompanyTier.FAANG,
    description="Apple values attention to detail, product sense, and deep technical expertise. Interviews are collaborative and product-focused.",
    rounds=[
        InterviewRound(
            name="Phone Screen",
            duration_minutes=45,
            focus_areas=["coding", "problem_solving"],
            question_types=["coding"],
            difficulty_modifier=1.0,
            max_questions=2,
        ),
        InterviewRound(
            name="Technical Deep Dive",
            duration_minutes=60,
            focus_areas=["deep_technical", "problem_solving", "optimization"],
            question_types=["coding", "technical_deep_dive"],
            difficulty_modifier=1.2,
            max_questions=2,
        ),
        InterviewRound(
            name="Design & Architecture",
            duration_minutes=60,
            focus_areas=["system_design", "product_sense", "user_experience"],
            question_types=["system_design", "product_design"],
            difficulty_modifier=1.2,
            max_questions=1,
        ),
        InterviewRound(
            name="Culture & Collaboration",
            duration_minutes=45,
            focus_areas=["attention_to_detail", "collaboration", "innovation"],
            question_types=["behavioral"],
            difficulty_modifier=1.0,
            max_questions=4,
        ),
    ],
    evaluation_criteria={
        "technical_depth": 0.30,
        "product_sense": 0.20,
        "problem_solving": 0.20,
        "attention_to_detail": 0.15,
        "culture_fit": 0.15,
    },
    question_style="technical_heavy",
    culture_keywords=["attention to detail", "simplicity", "innovation", "collaboration", "excellence"],
    bar_level="very_high",
    common_questions=[
        "Tell me about a product you helped build and your specific contributions.",
        "How do you ensure quality in your code?",
        "Describe a time you simplified a complex user experience.",
        "Tell me about a technical challenge you overcame.",
        "How do you approach code reviews?",
    ],
    what_they_value=[
        "Deep technical expertise",
        "Product sense and user empathy",
        "Attention to detail in everything",
        "Collaborative problem solving",
        "Simplicity and elegance in design",
    ],
)

NETFLIX_TEMPLATE = CompanyTemplate(
    company_name="Netflix",
    tier=CompanyTier.FAANG,
    description="Netflix values freedom and responsibility. They hire senior engineers who can operate independently and make good decisions.",
    rounds=[
        InterviewRound(
            name="Phone Screen",
            duration_minutes=45,
            focus_areas=["coding", "experience"],
            question_types=["coding", "behavioral"],
            difficulty_modifier=1.1,
            max_questions=2,
        ),
        InterviewRound(
            name="Technical Deep Dive",
            duration_minutes=60,
            focus_areas=["deep_technical", "system_design", "trade_offs"],
            question_types=["coding", "system_design"],
            difficulty_modifier=1.2,
            max_questions=2,
        ),
        InterviewRound(
            name="Leadership & Culture",
            duration_minutes=60,
            focus_areas=["judgment", "communication", "impact"],
            question_types=["behavioral", "situational"],
            difficulty_modifier=1.1,
            max_questions=4,
        ),
    ],
    evaluation_criteria={
        "technical_depth": 0.30,
        "judgment": 0.25,
        "communication": 0.20,
        "impact": 0.15,
        "culture_fit": 0.10,
    },
    question_style="balanced",
    culture_keywords=["freedom and responsibility", "judgment", "context not control", "highly aligned loosely coupled"],
    bar_level="very_high",
    common_questions=[
        "Tell me about a time you had to make a decision with significant trade-offs.",
        "How do you determine what to work on when there are many priorities?",
        "Describe a time you had to push back on a product decision.",
        "Tell me about your most impactful technical contribution.",
        "How do you handle disagreement on technical direction?",
    ],
    what_they_value=[
        "Senior-level judgment and decision-making",
        "Freedom with accountability",
        "Context-based decision making",
        "High impact with loose coupling",
        "Strong communication and influence",
    ],
)

# ─────────────────────────────────────────────────────────────
# Big Tech Templates
# ─────────────────────────────────────────────────────────────

MICROSOFT_TEMPLATE = CompanyTemplate(
    company_name="Microsoft",
    tier=CompanyTier.BIG_TECH,
    description="Microsoft values growth mindset, collaboration, and technical breadth. Interviews are structured and comprehensive.",
    rounds=[
        InterviewRound(
            name="Phone Screen",
            duration_minutes=45,
            focus_areas=["coding", "problem_solving"],
            question_types=["coding"],
            difficulty_modifier=1.0,
            max_questions=2,
        ),
        InterviewRound(
            name="Onsite (3-4 rounds)",
            duration_minutes=60,
            focus_areas=["coding", "system_design", "behavioral"],
            question_types=["coding", "system_design", "behavioral"],
            difficulty_modifier=1.1,
            max_questions=2,
        ),
    ],
    evaluation_criteria={
        "technical_depth": 0.30,
        "problem_solving": 0.25,
        "communication": 0.20,
        "growth_mindset": 0.15,
        "culture_fit": 0.10,
    },
    question_style="balanced",
    culture_keywords=["growth mindset", "learn-it-all", "collaboration", "inclusive"],
    bar_level="high",
    common_questions=[
        "Tell me about a time you learned something new to solve a problem.",
        "Describe a project where you collaborated across teams.",
        "How do you approach debugging complex issues?",
        "Tell me about a time you received critical feedback.",
    ],
    what_they_value=[
        "Growth mindset and continuous learning",
        "Collaborative problem solving",
        "Technical breadth across domains",
        "Inclusive and empathetic leadership",
    ],
)

# ─────────────────────────────────────────────────────────────
# Unicorn/Startup Templates
# ─────────────────────────────────────────────────────────────

STRIPE_TEMPLATE = CompanyTemplate(
    company_name="Stripe",
    tier=CompanyTier.UNICORN,
    description="Stripe values clean code, systems thinking, and pragmatic engineering. They test for both technical depth and product intuition.",
    rounds=[
        InterviewRound(
            name="Phone Screen",
            duration_minutes=45,
            focus_areas=["coding", "problem_solving"],
            question_types=["coding"],
            difficulty_modifier=1.1,
            max_questions=2,
        ),
        InterviewRound(
            name="Technical Interviews (3 rounds)",
            duration_minutes=60,
            focus_areas=["coding", "system_design", "product_engineering"],
            question_types=["coding", "system_design"],
            difficulty_modifier=1.2,
            max_questions=2,
        ),
        InterviewRound(
            name="Hiring Manager",
            duration_minutes=45,
            focus_areas=["culture", "growth", "collaboration"],
            question_types=["behavioral"],
            difficulty_modifier=1.0,
            max_questions=4,
        ),
    ],
    evaluation_criteria={
        "technical_depth": 0.30,
        "code_quality": 0.20,
        "problem_solving": 0.20,
        "product_sense": 0.15,
        "culture_fit": 0.15,
    },
    question_style="technical_heavy",
    culture_keywords=["user focus", "move with urgency", "think from first principles", "mechanical sympathy"],
    bar_level="high",
    common_questions=[
        "Design a payment processing system.",
        "How do you approach writing reliable distributed systems?",
        "Tell me about a time you simplified a complex system.",
        "How do you balance speed with code quality?",
    ],
    what_they_value=[
        "Clean, well-tested code",
        "Systems thinking at scale",
        "Product intuition and user empathy",
        "Pragmatic engineering decisions",
    ],
)

# ─────────────────────────────────────────────────────────────
# Template Registry
# ─────────────────────────────────────────────────────────────

COMPANY_TEMPLATES = {
    "google": GOOGLE_TEMPLATE,
    "meta": META_TEMPLATE,
    "facebook": META_TEMPLATE,
    "amazon": AMAZON_TEMPLATE,
    "apple": APPLE_TEMPLATE,
    "netflix": NETFLIX_TEMPLATE,
    "microsoft": MICROSOFT_TEMPLATE,
    "stripe": STRIPE_TEMPLATE,
}


def get_company_template(company_name: str) -> Optional[CompanyTemplate]:
    """Get interview template by company name."""
    return COMPANY_TEMPLATES.get(company_name.lower())


def list_companies(tier: CompanyTier = None) -> List[Dict]:
    """List all available company templates."""
    companies = []
    for key, template in COMPANY_TEMPLATES.items():
        if tier is None or template.tier == tier:
            companies.append({
                "key": key,
                "name": template.company_name,
                "tier": template.tier.value,
                "bar_level": template.bar_level,
                "question_style": template.question_style,
            })
    return companies


def get_questions_for_company(company_name: str, phase: str = None) -> List[str]:
    """Get common questions for a specific company."""
    template = get_company_template(company_name)
    if not template:
        return []
    
    return template.common_questions


def get_evaluation_criteria(company_name: str) -> Dict[str, float]:
    """Get evaluation criteria weights for a company."""
    template = get_company_template(company_name)
    if not template:
        return {}
    
    return template.evaluation_criteria
