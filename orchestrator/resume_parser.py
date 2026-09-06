"""
orchestrator/resume_parser.py
=============================
Advanced resume parser with skill extraction, experience parsing, and project detection.

Features:
1. Multi-format resume parsing (PDF, DOCX, plain text)
2. Skill extraction with confidence scores
3. Experience timeline parsing
4. Project detection and categorization
5. Education and certification extraction
6. Contact information extraction
7. Resume quality scoring
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class SkillLevel(Enum):
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4


@dataclass
class Skill:
    name: str
    level: SkillLevel
    confidence: float
    years_experience: Optional[float] = None
    context: str = ""


@dataclass
class Experience:
    title: str
    company: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_months: Optional[int] = None
    description: str = ""
    skills_used: List[str] = field(default_factory=list)


@dataclass
class Project:
    name: str
    description: str
    technologies: List[str] = field(default_factory=list)
    role: str = ""
    impact: str = ""


@dataclass
class Education:
    degree: str
    institution: str
    field_of_study: str = ""
    graduation_year: Optional[int] = None


@dataclass
class ResumeProfile:
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    summary: str = ""
    skills: List[Skill] = field(default_factory=list)
    experience: List[Experience] = field(default_factory=list)
    projects: List[Project] = field(default_factory=list)
    education: List[Education] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    total_experience_years: float = 0.0
    resume_quality_score: float = 0.0


class ResumeParser:
    """Advanced resume parser with skill extraction and experience parsing."""

    # Comprehensive skill database with levels
    SKILL_DATABASE = {
        # Programming Languages
        "python": {"category": "language", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "javascript": {"category": "language", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "java": {"category": "language", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "c++": {"category": "language", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "c#": {"category": "language", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "go": {"category": "language", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "rust": {"category": "language", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "ruby": {"category": "language", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "php": {"category": "language", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "swift": {"category": "language", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "kotlin": {"category": "language", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "typescript": {"category": "language", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "scala": {"category": "language", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "r": {"category": "language", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "matlab": {"category": "language", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "sql": {"category": "language", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        
        # Frameworks & Libraries
        "react": {"category": "framework", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "angular": {"category": "framework", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "vue": {"category": "framework", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "node.js": {"category": "framework", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "express": {"category": "framework", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "django": {"category": "framework", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "flask": {"category": "framework", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "spring": {"category": "framework", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "rails": {"category": "framework", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "laravel": {"category": "framework", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "next.js": {"category": "framework", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "fastapi": {"category": "framework", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "spring boot": {"category": "framework", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        
        # Cloud & DevOps
        "aws": {"category": "cloud", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "azure": {"category": "cloud", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "gcp": {"category": "cloud", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "docker": {"category": "devops", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "kubernetes": {"category": "devops", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "terraform": {"category": "devops", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "jenkins": {"category": "devops", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "ci/cd": {"category": "devops", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        
        # Data & ML
        "machine learning": {"category": "data", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "deep learning": {"category": "data", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "tensorflow": {"category": "data", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "pytorch": {"category": "data", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "nlp": {"category": "data", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "computer vision": {"category": "data", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "data analysis": {"category": "data", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "data science": {"category": "data", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "pandas": {"category": "data", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "numpy": {"category": "data", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "scikit-learn": {"category": "data", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "hadoop": {"category": "data", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "spark": {"category": "data", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        
        # Databases
        "mysql": {"category": "database", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "postgresql": {"category": "database", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "mongodb": {"category": "database", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "oracle": {"category": "database", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "sql server": {"category": "database", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "dynamodb": {"category": "database", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "elasticsearch": {"category": "database", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "cassandra": {"category": "database", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "neo4j": {"category": "database", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        
        # Tools & Practices
        "git": {"category": "tool", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "jira": {"category": "tool", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "agile": {"category": "practice", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "scrum": {"category": "practice", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "rest api": {"category": "architecture", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "graphql": {"category": "architecture", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "microservices": {"category": "architecture", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "tdd": {"category": "practice", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "bdd": {"category": "practice", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "oauth": {"category": "security", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
        "jwt": {"category": "security", "level_map": {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}},
    }

    def parse_resume(self, text: str) -> ResumeProfile:
        """Parse a resume and extract structured information."""
        profile = ResumeProfile()
        
        # Extract contact information
        profile.email = self._extract_email(text)
        profile.phone = self._extract_phone(text)
        profile.location = self._extract_location(text)
        profile.name = self._extract_name(text)
        
        # Extract sections
        profile.summary = self._extract_summary(text)
        profile.skills = self._extract_skills(text)
        profile.experience = self._extract_experience(text)
        profile.projects = self._extract_projects(text)
        profile.education = self._extract_education(text)
        profile.certifications = self._extract_certifications(text)
        profile.languages = self._extract_languages(text)
        
        # Calculate total experience
        profile.total_experience_years = self._calculate_total_experience(profile.experience)
        
        # Calculate resume quality score
        profile.resume_quality_score = self._calculate_resume_quality(profile)
        
        return profile

    def _extract_email(self, text: str) -> str:
        """Extract email address from resume."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(email_pattern, text)
        return match.group(0) if match else ""

    def _extract_phone(self, text: str) -> str:
        """Extract phone number from resume."""
        phone_pattern = r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}'
        match = re.search(phone_pattern, text)
        return match.group(0) if match else ""

    def _extract_location(self, text: str) -> str:
        """Extract location from resume."""
        location_patterns = [
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*,\s*[A-Z]{2})',
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*,\s*[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)',
        ]
        for pattern in location_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return ""

    def _extract_name(self, text: str) -> str:
        """Extract name from resume (first line, typically)."""
        lines = text.strip().split("\n")
        if lines:
            first_line = lines[0].strip()
            # Simple heuristic: if first line is short and doesn't contain common resume keywords
            if len(first_line) < 50 and not any(kw in first_line.lower() for kw in ["resume", "cv", "curriculum", "contact", "summary"]):
                return first_line
        return ""

    def _extract_summary(self, text: str) -> str:
        """Extract summary/objective section."""
        summary_patterns = [
            r'(?:summary|objective|profile|about me)[:\s]*(.*?)(?:\n\n|\Z)',
        ]
        for pattern in summary_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()[:500]
        return ""

    def _extract_skills(self, text: str) -> List[Skill]:
        """Extract skills with confidence levels."""
        skills = []
        text_lower = text.lower()
        
        for skill_name, skill_info in self.SKILL_DATABASE.items():
            if skill_name in text_lower:
                # Determine skill level based on context
                level = self._determine_skill_level(text, skill_name)
                confidence = self._calculate_skill_confidence(text, skill_name)
                
                skills.append(Skill(
                    name=skill_name,
                    level=level,
                    confidence=confidence,
                    context=self._get_skill_context(text, skill_name)
                ))
        
        return skills

    def _determine_skill_level(self, text: str, skill: str) -> SkillLevel:
        """Determine skill level based on context."""
        text_lower = text.lower()
        
        # Look for level indicators
        expert_indicators = ["expert", "advanced", "senior", "lead", "architect", "mastery"]
        advanced_indicators = ["experienced", "proficient", "strong", "solid", "extensive"]
        intermediate_indicators = ["intermediate", "familiar", "working knowledge", "comfortable"]
        beginner_indicators = ["beginner", "basic", "learning", "introductory", "foundational"]
        
        for indicator in expert_indicators:
            if indicator in text_lower and skill in text_lower:
                return SkillLevel.EXPERT
        
        for indicator in advanced_indicators:
            if indicator in text_lower and skill in text_lower:
                return SkillLevel.ADVANCED
        
        for indicator in intermediate_indicators:
            if indicator in text_lower and skill in text_lower:
                return SkillLevel.INTERMEDIATE
        
        for indicator in beginner_indicators:
            if indicator in text_lower and skill in text_lower:
                return SkillLevel.BEGINNER
        
        # Default to intermediate if mentioned
        return SkillLevel.INTERMEDIATE

    def _calculate_skill_confidence(self, text: str, skill: str) -> float:
        """Calculate confidence score for skill extraction."""
        confidence = 0.5  # Base confidence
        
        # Boost for explicit mention
        if skill in text.lower():
            confidence += 0.2
        
        # Boost for context (projects, experience)
        if any(kw in text.lower() for kw in ["project", "built", "developed", "implemented"]):
            confidence += 0.1
        
        # Boost for proficiency indicators
        if any(kw in text.lower() for kw in ["expert", "advanced", "proficient", "experienced"]):
            confidence += 0.1
        
        return min(confidence, 1.0)

    def _get_skill_context(self, text: str, skill: str) -> str:
        """Get context around skill mention."""
        lines = text.split("\n")
        for line in lines:
            if skill in line.lower():
                return line.strip()[:200]
        return ""

    def _extract_experience(self, text: str) -> List[Experience]:
        """Extract work experience."""
        experiences = []
        
        # Look for experience section
        exp_section = re.search(r'(?:experience|work history|employment)[:\s]*(.*?)(?:\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
        if exp_section:
            exp_text = exp_section.group(1)
            # Parse individual positions
            positions = re.split(r'\n(?=[A-Z])', exp_text)
            for pos in positions:
                if len(pos.strip()) > 20:
                    exp = Experience(
                        title=self._extract_job_title(pos),
                        company=self._extract_company(pos),
                        description=pos.strip()[:500]
                    )
                    experiences.append(exp)
        
        return experiences

    def _extract_job_title(self, text: str) -> str:
        """Extract job title from position text."""
        title_patterns = [
            r'((?:Senior|Junior|Lead|Principal|Staff|Chief)?\s*(?:Software|Data|DevOps|Cloud|Security|Frontend|Backend|Full Stack|Mobile|QA|Test|ML|AI)?\s*(?:Engineer|Developer|Architect|Manager|Director|Analyst|Scientist|Specialist|Consultant))',
        ]
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        return ""

    def _extract_company(self, text: str) -> str:
        """Extract company name from position text."""
        # Simple heuristic: look for capitalized words after title
        lines = text.split("\n")
        for line in lines:
            words = line.split()
            if len(words) >= 2:
                # Check if first word is likely a company name
                if words[0][0].isupper() and not any(kw in words[0].lower() for kw in ["senior", "junior", "lead", "principal", "staff", "chief"]):
                    return words[0]
        return ""

    def _extract_projects(self, text: str) -> List[Project]:
        """Extract projects from resume."""
        projects = []
        
        # Look for projects section
        proj_section = re.search(r'(?:projects|key projects|notable projects)[:\s]*(.*?)(?:\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
        if proj_section:
            proj_text = proj_section.group(1)
            # Parse individual projects
            project_blocks = re.split(r'\n(?=[A-Z])', proj_text)
            for block in project_blocks:
                if len(block.strip()) > 20:
                    proj = Project(
                        name=self._extract_project_name(block),
                        description=block.strip()[:500],
                        technologies=self._extract_technologies(block)
                    )
                    projects.append(proj)
        
        return projects

    def _extract_project_name(self, text: str) -> str:
        """Extract project name."""
        # First line or first capitalized phrase
        lines = text.split("\n")
        if lines:
            return lines[0].strip()[:100]
        return ""

    def _extract_technologies(self, text: str) -> List[str]:
        """Extract technologies used in project."""
        techs = []
        text_lower = text.lower()
        
        tech_keywords = [
            "python", "javascript", "java", "c++", "react", "angular", "vue",
            "node.js", "django", "flask", "spring", "aws", "azure", "gcp",
            "docker", "kubernetes", "postgresql", "mongodb", "redis",
            "tensorflow", "pytorch", "machine learning", "deep learning"
        ]
        
        for tech in tech_keywords:
            if tech in text_lower:
                techs.append(tech)
        
        return techs

    def _extract_education(self, text: str) -> List[Education]:
        """Extract education information."""
        education = []
        
        # Look for education section
        edu_section = re.search(r'(?:education|academic background)[:\s]*(.*?)(?:\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
        if edu_section:
            edu_text = edu_section.group(1)
            # Parse individual degrees
            degree_blocks = re.split(r'\n(?=[A-Z])', edu_text)
            for block in degree_blocks:
                if len(block.strip()) > 10:
                    edu = Education(
                        degree=self._extract_degree(block),
                        institution=self._extract_institution(block),
                        field_of_study=self._extract_field_of_study(block)
                    )
                    education.append(edu)
        
        return education

    def _extract_degree(self, text: str) -> str:
        """Extract degree type."""
        degree_patterns = [
            r'(Bachelor(?:\'s)?|Master(?:\'s)?|Ph\.?D\.?|MBA|B\.?S\.?|M\.?S\.?|B\.?A\.?|M\.?A\.?)\s*(?:of|in)?\s*([A-Za-z\s]+)?',
        ]
        for pattern in degree_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        return ""

    def _extract_institution(self, text: str) -> str:
        """Extract institution name."""
        # Look for university/college names
        institution_patterns = [
            r'((?:University|College|Institute|School)\s+(?:of\s+)?[A-Za-z\s]+)',
        ]
        for pattern in institution_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        return ""

    def _extract_field_of_study(self, text: str) -> str:
        """Extract field of study."""
        field_patterns = [
            r'(?:in|of)\s+([A-Za-z\s]+(?:Engineering|Science|Arts|Business|Computer|Data|Information|Technology|Mathematics|Physics|Chemistry|Biology|Economics|Finance|Marketing|Management))',
        ]
        for pattern in field_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    def _extract_certifications(self, text: str) -> List[str]:
        """Extract certifications."""
        certifications = []
        
        # Look for certifications section
        cert_section = re.search(r'(?:certifications|certificates|licenses)[:\s]*(.*?)(?:\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
        if cert_section:
            cert_text = cert_section.group(1)
            # Split by newlines
            certs = [line.strip() for line in cert_text.split("\n") if line.strip()]
            certifications = certs[:10]  # Limit
        
        return certifications

    def _extract_languages(self, text: str) -> List[str]:
        """Extract programming languages mentioned."""
        languages = []
        
        language_names = [
            "Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", "Ruby", "PHP",
            "Swift", "Kotlin", "TypeScript", "Scala", "R", "MATLAB", "SQL"
        ]
        
        for lang in language_names:
            if lang.lower() in text.lower():
                languages.append(lang)
        
        return languages

    def _calculate_total_experience(self, experiences: List[Experience]) -> float:
        """Calculate total years of experience."""
        total_months = 0
        for exp in experiences:
            if exp.duration_months:
                total_months += exp.duration_months
        return total_months / 12.0 if total_months > 0 else 0.0

    def _calculate_resume_quality(self, profile: ResumeProfile) -> float:
        """Calculate resume quality score (0-1)."""
        score = 0.0
        
        # Contact information (20%)
        if profile.email:
            score += 0.05
        if profile.phone:
            score += 0.05
        if profile.location:
            score += 0.05
        if profile.name:
            score += 0.05
        
        # Summary (15%)
        if profile.summary and len(profile.summary) > 50:
            score += 0.15
        
        # Skills (25%)
        if profile.skills:
            skill_score = min(len(profile.skills) / 10, 1.0) * 0.25
            score += skill_score
        
        # Experience (25%)
        if profile.experience:
            exp_score = min(len(profile.experience) / 3, 1.0) * 0.25
            score += exp_score
        
        # Projects (10%)
        if profile.projects:
            proj_score = min(len(profile.projects) / 3, 1.0) * 0.10
            score += proj_score
        
        # Education (5%)
        if profile.education:
            score += 0.05
        
        return min(score, 1.0)


# Convenience function
def parse_resume(text: str) -> ResumeProfile:
    """Parse a resume and return structured profile."""
    parser = ResumeParser()
    return parser.parse_resume(text)
