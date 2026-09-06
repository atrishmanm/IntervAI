"""
scripts/generate_resume_training_data.py
========================================
Generate synthetic resume-based Q&A training data.

Creates training examples like:
  - "I see you have experience with Python. Tell me about a challenging project."
  - "Your resume mentions microservices. How would you design a distributed system?"
  - "You worked at TechCorp. What was your biggest achievement there?"

These teach the model to ask personalized questions based on resume content.
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"

# Resume skill templates
SKILLS = [
    "Python", "JavaScript", "TypeScript", "Java", "C++", "Go", "Rust",
    "React", "Angular", "Vue", "Node.js", "Django", "Flask", "FastAPI",
    "PostgreSQL", "MongoDB", "Redis", "Docker", "Kubernetes", "AWS",
    "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
    "System Design", "Microservices", "REST APIs", "GraphQL",
    "Git", "CI/CD", "Terraform", "Linux",
]

DOMAINS = [
    "web development", "mobile development", "data engineering",
    "cloud infrastructure", "machine learning", "fintech",
    "healthcare", "ecommerce", "gaming", "cybersecurity",
]

JOB_TITLES = [
    "Software Engineer", "Senior Developer", "Full Stack Developer",
    "Backend Engineer", "Frontend Developer", "DevOps Engineer",
    "Data Engineer", "ML Engineer", "Platform Engineer",
    "Tech Lead", "Engineering Manager",
]

COMPANIES = [
    "Google", "Meta", "Amazon", "Microsoft", "Apple", "Netflix",
    "Stripe", "Airbnb", "Uber", "Shopify", "Datadog", "Snowflake",
    "StartupXYZ", "TechCorp", "InnovateLab", "CloudScale",
]

PROJECTS = [
    "real-time chat application", "e-commerce platform", "data pipeline",
    "microservices migration", "CI/CD pipeline", "machine learning model",
    "REST API", "mobile app", "dashboard", "monitoring system",
    "payment integration", "search engine", "recommendation system",
]

# Question templates based on resume content
SKILL_QUESTIONS = [
    "I see you have experience with {skill}. Can you tell me about a challenging project where you used {skill}?",
    "Your resume mentions {skill}. How would you explain {skill} to someone unfamiliar with it?",
    "You listed {skill} as a skill. What's the most complex problem you've solved using {skill}?",
    "I notice {skill} is one of your skills. What are the common pitfalls when working with {skill}?",
    "Your background includes {skill}. How do you stay updated with {skill} best practices?",
]

EXPERIENCE_QUESTIONS = [
    "You worked at {company} as a {title}. What was your biggest technical achievement there?",
    "Tell me about your time at {company}. What challenges did you face?",
    "Your resume shows you were a {title} at {company}. How did you contribute to the team?",
    "You mentioned working on {project} at {company}. What was your specific role?",
    "What did you learn from your experience at {company} that you apply today?",
]

PROJECT_QUESTIONS = [
    "Tell me about the {project} you built. What was the architecture?",
    "You worked on a {project}. What would you do differently if you rebuilt it?",
    "Describe the {project} you mentioned. What technologies did you use?",
    "Your resume mentions building a {project}. How did you handle scalability?",
    "What was the most challenging aspect of building the {project}?",
]

# Answer templates (for training the model to generate good answers)
ANSWER_TEMPLATES = [
    "When I worked on {project} at {company}, I used {skill} to {action}. The key challenge was {challenge}, and I solved it by {solution}. As a result, {outcome}.",
    "At {company}, I was responsible for {responsibility}. I implemented {solution} using {skill}, which {outcome}.",
    "During my time as {title} at {company}, I led the development of {project}. I chose {skill} because {reason}. The system {outcome}.",
    "For the {project}, I designed a {architecture} architecture using {skill}. This allowed us to {outcome}.",
]

ACTIONS = [
    "build a scalable backend", "optimize database queries", "implement caching",
    "design the API", "set up monitoring", " automate deployments",
    "improve performance", "reduce latency", "handle high traffic",
    "integrate third-party services", "implement authentication",
]

CHALLENGES = [
    "handling concurrent users", "managing state across services",
    "ensuring data consistency", "optimizing for low latency",
    "scaling to millions of users", "maintaining code quality",
    "meeting tight deadlines", "integrating legacy systems",
]

SOLUTIONS = [
    "implementing a queue-based architecture", "using Redis for caching",
    "applying database sharding", "implementing circuit breakers",
    "using event-driven architecture", "adding comprehensive monitoring",
    "writing thorough tests", "conducting code reviews",
]

OUTCOMES = [
    "we reduced latency by 50%", "the system handled 10x more traffic",
    "we achieved 99.9% uptime", "deployment time dropped from hours to minutes",
    "we cut infrastructure costs by 30%", "the team velocity doubled",
    "we eliminated all critical bugs", "user satisfaction increased by 40%",
]

REASONS = [
    "its excellent ecosystem", "the team's existing expertise",
    "its performance characteristics", "the strong community support",
    "its type safety", "the rapid development cycle",
]

ARCHITECTURES = [
    "microservices", "event-driven", "serverless", "layered",
    "CQRS", "clean architecture", "hexagonal", "pipeline-based",
]


def generate_skill_qa():
    """Generate Q&A pairs based on skills."""
    examples = []
    for skill in random.sample(SKILLS, min(20, len(SKILLS))):
        q_template = random.choice(SKILL_QUESTIONS)
        question = q_template.format(skill=skill)

        # Generate a relevant answer
        project = random.choice(PROJECTS)
        company = random.choice(COMPANIES)
        action = random.choice(ACTIONS)
        challenge = random.choice(CHALLENGES)
        solution = random.choice(SOLUTIONS)
        outcome = random.choice(OUTCOMES)

        answer = random.choice(ANSWER_TEMPLATES).format(
            project=project, company=company, skill=skill,
            action=action, challenge=challenge, solution=solution,
            outcome=outcome, title="Software Engineer",
            responsibility=f"building {project}",
            architecture=random.choice(ARCHITECTURES),
            reason=random.choice(REASONS),
        )

        examples.append({
            "messages": [
                {"role": "system", "content": "You are a technical interviewer conducting a personalized interview based on the candidate's resume."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        })
    return examples


def generate_experience_qa():
    """Generate Q&A pairs based on work experience."""
    examples = []
    for _ in range(15):
        company = random.choice(COMPANIES)
        title = random.choice(JOB_TITLES)
        project = random.choice(PROJECTS)

        q_template = random.choice(EXPERIENCE_QUESTIONS)
        question = q_template.format(company=company, title=title, project=project)

        skill = random.choice(SKILLS)
        solution = random.choice(SOLUTIONS)
        outcome = random.choice(OUTCOMES)

        answer = random.choice(ANSWER_TEMPLATES).format(
            project=project, company=company, skill=skill,
            action=random.choice(ACTIONS), challenge=random.choice(CHALLENGES),
            solution=solution, outcome=outcome, title=title,
            responsibility=f"leading {project} development",
            architecture=random.choice(ARCHITECTURES),
            reason=random.choice(REASONS),
        )

        examples.append({
            "messages": [
                {"role": "system", "content": "You are a technical interviewer conducting a personalized interview based on the candidate's resume."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        })
    return examples


def generate_project_qa():
    """Generate Q&A pairs based on projects."""
    examples = []
    for project in random.sample(PROJECTS, min(12, len(PROJECTS))):
        q_template = random.choice(PROJECT_QUESTIONS)
        question = q_template.format(project=project)

        skill = random.choice(SKILLS)
        company = random.choice(COMPANIES)

        answer = random.choice(ANSWER_TEMPLATES).format(
            project=project, company=company, skill=skill,
            action=random.choice(ACTIONS), challenge=random.choice(CHALLENGES),
            solution=random.choice(SOLUTIONS), outcome=random.choice(OUTCOMES),
            title="Software Engineer",
            responsibility=f"building {project}",
            architecture=random.choice(ARCHITECTURES),
            reason=random.choice(REASONS),
        )

        examples.append({
            "messages": [
                {"role": "system", "content": "You are a technical interviewer conducting a personalized interview based on the candidate's resume."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        })
    return examples


def generate_domain_qa():
    """Generate domain-specific questions."""
    examples = []
    for domain in random.sample(DOMAINS, min(8, len(DOMAINS))):
        question = f"You mentioned interest in {domain}. What draws you to this field, and how have you applied it in your work?"

        answer = f"I'm passionate about {domain} because {random.choice(REASONS)}. " \
                 f"In my previous role, I {random.choice(ACTIONS)} using {random.choice(SKILLS)}, " \
                 f"which {random.choice(OUTCOMES)}."

        examples.append({
            "messages": [
                {"role": "system", "content": "You are a technical interviewer conducting a personalized interview based on the candidate's resume."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        })
    return examples


def main():
    print("=" * 60)
    print("  Generating Resume-Based Training Data")
    print("=" * 60)

    all_examples = []
    all_examples.extend(generate_skill_qa())
    print(f"  Skill Q&A: {len(all_examples)} examples")

    n = len(all_examples)
    all_examples.extend(generate_experience_qa())
    print(f"  Experience Q&A: {len(all_examples) - n} examples")

    n = len(all_examples)
    all_examples.extend(generate_project_qa())
    print(f"  Project Q&A: {len(all_examples) - n} examples")

    n = len(all_examples)
    all_examples.extend(generate_domain_qa())
    print(f"  Domain Q&A: {len(all_examples) - n} examples")

    # Shuffle
    random.shuffle(all_examples)

    # Save
    output_path = DATA_DIR / "resume_interview_qa.jsonl"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n  Total: {len(all_examples)} examples")
    print(f"  Saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
