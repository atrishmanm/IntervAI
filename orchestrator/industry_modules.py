"""
orchestrator/industry_modules.py
================================
Industry-specific interview modules for Backend, Frontend, Data Science, DevOps.

Each module defines:
1. Core skills to assess
2. Technical question patterns
3. System design focus areas
4. Tool-specific questions
5. Evaluation criteria specific to the industry
6. Common project scenarios
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class IndustryType(Enum):
    BACKEND = "backend"
    FRONTEND = "frontend"
    DATA_SCIENCE = "data_science"
    DEVOPS = "devops"
    FULLSTACK = "fullstack"
    MOBILE = "mobile"
    SECURITY = "security"
    EMBEDDED = "embedded"


@dataclass
class SkillAssessment:
    """Defines how to assess a specific skill."""
    skill_name: str
    weight: float  # 0-1, importance in evaluation
    question_types: List[str]
    difficulty_range: List[str]  # ["easy", "hard"]
    evaluation_rubric: Dict[str, float]


@dataclass
class IndustryModule:
    """Complete industry-specific interview module."""
    industry: IndustryType
    name: str
    description: str
    core_skills: List[SkillAssessment]
    system_design_topics: List[str]
    coding_challenges: List[str]
    tool_questions: Dict[str, List[str]]  # tool_name -> questions
    project_scenarios: List[str]
    evaluation_criteria: Dict[str, float]
    typical_questions: List[str]
    what_interviewers_look_for: List[str]
    common_mistakes: List[str]


# ─────────────────────────────────────────────────────────────
# Backend Engineering Module
# ─────────────────────────────────────────────────────────────

BACKEND_MODULE = IndustryModule(
    industry=IndustryType.BACKEND,
    name="Backend Engineering",
    description="Backend engineers build server-side logic, APIs, databases, and distributed systems.",
    core_skills=[
        SkillAssessment(
            skill_name="api_design",
            weight=0.20,
            question_types=["system_design", "coding"],
            difficulty_range=["medium", "hard"],
            evaluation_rubric={"correctness": 0.4, "scalability": 0.3, "documentation": 0.3},
        ),
        SkillAssessment(
            skill_name="database_design",
            weight=0.20,
            question_types=["system_design", "coding"],
            difficulty_range=["medium", "hard"],
            evaluation_rubric={"normalization": 0.3, "indexing": 0.3, "query_optimization": 0.4},
        ),
        SkillAssessment(
            skill_name="distributed_systems",
            weight=0.25,
            question_types=["system_design", "behavioral"],
            difficulty_range=["hard", "expert"],
            evaluation_rubric={"trade_offs": 0.4, "consistency": 0.3, "availability": 0.3},
        ),
        SkillAssessment(
            skill_name="performance_optimization",
            weight=0.15,
            question_types=["coding", "system_design"],
            difficulty_range=["medium", "hard"],
            evaluation_rubric={"profiling": 0.3, "caching": 0.4, "scaling": 0.3},
        ),
        SkillAssessment(
            skill_name="security",
            weight=0.10,
            question_types=["system_design", "behavioral"],
            difficulty_range=["medium", "hard"],
            evaluation_rubric={"authentication": 0.4, "authorization": 0.3, "encryption": 0.3},
        ),
        SkillAssessment(
            skill_name="code_quality",
            weight=0.10,
            question_types=["coding", "behavioral"],
            difficulty_range=["easy", "medium"],
            evaluation_rubric={"readability": 0.3, "testing": 0.4, "maintainability": 0.3},
        ),
    ],
    system_design_topics=[
        "URL shortener (like bit.ly)",
        "Real-time chat system (like WhatsApp)",
        "E-commerce platform (like Amazon)",
        "Payment processing system (like Stripe)",
        "Notification system (like Slack)",
        "Content delivery network (like Cloudflare)",
        "Search engine (like Google)",
        "Social media feed (like Twitter)",
        "Video streaming platform (like YouTube)",
        "Ride-sharing service (like Uber)",
    ],
    coding_challenges=[
        "Implement a rate limiter",
        "Design a connection pool",
        "Build a simple cache with TTL",
        "Implement a task scheduler",
        "Design a distributed lock",
        "Build a message queue",
        "Implement circuit breaker pattern",
        "Design a feature flag system",
    ],
    tool_questions={
        "databases": [
            "Explain database indexing strategies.",
            "When would you use SQL vs NoSQL?",
            "How do you handle database migrations?",
            "Explain ACID properties with examples.",
        ],
        "caching": [
            "Explain cache invalidation strategies.",
            "When would you use Redis vs Memcached?",
            "How do you handle cache stampede?",
            "Explain write-through vs write-behind caching.",
        ],
        "message_queues": [
            "Explain at-least-once vs exactly-once delivery.",
            "When would you use Kafka vs RabbitMQ?",
            "How do you handle message ordering?",
            "Explain dead letter queues.",
        ],
        "apis": [
            "REST vs gRPC: when to use which?",
            "How do you version APIs?",
            "Explain API rate limiting strategies.",
            "How do you handle API authentication?",
        ],
    },
    project_scenarios=[
        "Design a URL shortener that handles 100M URLs/day",
        "Build a real-time notification system for a social media app",
        "Design a payment processing system with fraud detection",
        "Build a content management system for a news website",
        "Design a search autocomplete system",
        "Build a job scheduling system for background tasks",
    ],
    evaluation_criteria={
        "technical_depth": 0.30,
        "system_design": 0.25,
        "problem_solving": 0.20,
        "code_quality": 0.15,
        "communication": 0.10,
    },
    typical_questions=[
        "Design a URL shortener like bit.ly.",
        "How would you handle 1 million concurrent users?",
        "Explain the CAP theorem with real-world examples.",
        "How do you ensure data consistency in a distributed system?",
        "Describe your approach to database schema design.",
        "How do you handle database failures in production?",
        "Explain your monitoring and alerting strategy.",
        "How do you handle security vulnerabilities?",
    ],
    what_interviewers_look_for=[
        "Deep understanding of distributed systems trade-offs",
        "Ability to design scalable and reliable systems",
        "Knowledge of database optimization techniques",
        "Security-conscious mindset",
        "Clear communication of technical decisions",
    ],
    common_mistakes=[
        "Over-engineering simple solutions",
        "Ignoring error handling and edge cases",
        "Not considering scalability from the start",
        "Forgetting about monitoring and observability",
        "Not explaining trade-offs clearly",
    ],
)

# ─────────────────────────────────────────────────────────────
# Frontend Engineering Module
# ─────────────────────────────────────────────────────────────

FRONTEND_MODULE = IndustryModule(
    industry=IndustryType.FRONTEND,
    name="Frontend Engineering",
    description="Frontend engineers build user interfaces, optimize performance, and ensure accessibility.",
    core_skills=[
        SkillAssessment(
            skill_name="component_design",
            weight=0.25,
            question_types=["system_design", "coding"],
            difficulty_range=["medium", "hard"],
            evaluation_rubric={"reusability": 0.4, "composability": 0.3, "maintainability": 0.3},
        ),
        SkillAssessment(
            skill_name="state_management",
            weight=0.20,
            question_types=["coding", "system_design"],
            difficulty_range=["medium", "hard"],
            evaluation_rubric={"predictability": 0.4, "performance": 0.3, "debugging": 0.3},
        ),
        SkillAssessment(
            skill_name="performance_optimization",
            weight=0.20,
            question_types=["coding", "system_design"],
            difficulty_range=["medium", "hard"],
            evaluation_rubric={"bundle_size": 0.3, "rendering": 0.4, "caching": 0.3},
        ),
        SkillAssessment(
            skill_name="accessibility",
            weight=0.15,
            question_types=["coding", "behavioral"],
            difficulty_range=["easy", "medium"],
            evaluation_rubric={"wcag_compliance": 0.4, "screen_readers": 0.3, "keyboard_nav": 0.3},
        ),
        SkillAssessment(
            skill_name="testing",
            weight=0.10,
            question_types=["coding", "behavioral"],
            difficulty_range=["medium", "hard"],
            evaluation_rubric={"unit_tests": 0.3, "integration": 0.4, "e2e": 0.3},
        ),
        SkillAssessment(
            skill_name="responsive_design",
            weight=0.10,
            question_types=["coding", "system_design"],
            difficulty_range=["easy", "medium"],
            evaluation_rubric={"mobile_first": 0.4, "breakpoints": 0.3, "flexibility": 0.3},
        ),
    ],
    system_design_topics=[
        "Design a component library (like Material UI)",
        "Build a real-time collaboration tool (like Figma)",
        "Design a dashboard with drag-and-drop (like Notion)",
        "Build a progressive web app (like Twitter Lite)",
        "Design a design system (like Atlassian Design)",
        "Build a rich text editor (like Google Docs)",
        "Design a notification center",
        "Build a file upload system with preview",
    ],
    coding_challenges=[
        "Implement a debounce/throttle function",
        "Build a virtual scrolling list",
        "Implement a simple state management library",
        "Build a drag-and-drop sortable list",
        "Implement infinite scrolling with cache",
        "Build a modal/dialog system with focus management",
        "Implement a toast notification system",
        "Build a data table with sorting, filtering, pagination",
    ],
    tool_questions={
        "react": [
            "Explain React reconciliation algorithm.",
            "When would you use useEffect vs useMemo?",
            "How do you optimize React re-renders?",
            "Explain React Server Components.",
        ],
        "css": [
            "Explain CSS specificity and cascade.",
            "When would you use CSS-in-JS vs CSS modules?",
            "How do you create responsive layouts?",
            "Explain CSS Grid vs Flexbox use cases.",
        ],
        "performance": [
            "How do you identify and fix performance bottlenecks?",
            "Explain lazy loading strategies.",
            "How do you optimize bundle size?",
            "Explain code splitting techniques.",
        ],
        "accessibility": [
            "How do you test for accessibility?",
            "Explain ARIA roles and attributes.",
            "How do you handle focus management?",
            "Explain WCAG compliance levels.",
        ],
    },
    project_scenarios=[
        "Build a dashboard with real-time data visualization",
        "Design a component library for a design system",
        "Build a file upload system with drag-and-drop",
        "Create a rich text editor with collaborative editing",
        "Build a responsive navigation with mobile menu",
        "Implement a notification system with real-time updates",
    ],
    evaluation_criteria={
        "component_design": 0.25,
        "performance": 0.20,
        "accessibility": 0.15,
        "testing": 0.15,
        "code_quality": 0.15,
        "communication": 0.10,
    },
    typical_questions=[
        "How would you design a reusable component library?",
        "Explain how you would optimize a slow-rendering page.",
        "How do you handle state in a complex application?",
        "Describe your approach to responsive design.",
        "How do you ensure accessibility in your applications?",
        "Explain your testing strategy for frontend code.",
        "How do you handle cross-browser compatibility?",
        "Describe your approach to code organization.",
    ],
    what_interviewers_look_for=[
        "Strong understanding of component architecture",
        "Performance optimization skills",
        "Accessibility awareness",
        "Testing discipline",
        "Clean, maintainable code",
    ],
    common_mistakes=[
        "Overcomplicating component APIs",
        "Ignoring performance implications",
        "Not considering accessibility",
        "Insufficient testing",
        "Poor state management patterns",
    ],
)

# ─────────────────────────────────────────────────────────────
# Data Science Module
# ─────────────────────────────────────────────────────────────

DATA_SCIENCE_MODULE = IndustryModule(
    industry=IndustryType.DATA_SCIENCE,
    name="Data Science",
    description="Data scientists analyze data, build models, and derive insights to drive business decisions.",
    core_skills=[
        SkillAssessment(
            skill_name="statistics",
            weight=0.20,
            question_types=["technical", "coding"],
            difficulty_range=["medium", "hard"],
            evaluation_rubric={"hypothesis_testing": 0.4, "probability": 0.3, "regression": 0.3},
        ),
        SkillAssessment(
            skill_name="machine_learning",
            weight=0.25,
            question_types=["technical", "system_design"],
            difficulty_range=["medium", "hard"],
            evaluation_rubric={"model_selection": 0.3, "feature_engineering": 0.4, "evaluation": 0.3},
        ),
        SkillAssessment(
            skill_name="sql",
            weight=0.20,
            question_types=["coding", "technical"],
            difficulty_range=["medium", "hard"],
            evaluation_rubric={"query_writing": 0.4, "optimization": 0.3, "window_functions": 0.3},
        ),
        SkillAssessment(
            skill_name="data_visualization",
            weight=0.15,
            question_types=["coding", "behavioral"],
            difficulty_range=["easy", "medium"],
            evaluation_rubric={"storytelling": 0.4, "clarity": 0.3, "tool_choice": 0.3},
        ),
        SkillAssessment(
            skill_name="python",
            weight=0.15,
            question_types=["coding", "technical"],
            difficulty_range=["medium", "hard"],
            evaluation_rubric={"pandas": 0.4, "numpy": 0.3, "scikit-learn": 0.3},
        ),
        SkillAssessment(
            skill_name="communication",
            weight=0.05,
            question_types=["behavioral"],
            difficulty_range=["easy", "medium"],
            evaluation_rubric={"clarity": 0.5, "simplification": 0.5},
        ),
    ],
    system_design_topics=[
        "Design an A/B testing framework",
        "Build a recommendation system (like Netflix)",
        "Design a fraud detection pipeline",
        "Build a real-time analytics dashboard",
        "Design a feature store for ML models",
        "Build an anomaly detection system",
        "Design a data pipeline for ETL",
        "Build a churn prediction system",
    ],
    coding_challenges=[
        "Write a SQL query to find top customers by revenue",
        "Implement k-means clustering from scratch",
        "Build a simple linear regression model",
        "Write a function to handle missing data",
        "Implement a basic recommendation system",
        "Write code for A/B test statistical significance",
        "Build a data cleaning pipeline",
        "Implement a confusion matrix and calculate metrics",
    ],
    tool_questions={
        "python": [
            "Explain pandas DataFrame operations.",
            "How do you handle large datasets that don't fit in memory?",
            "Explain vectorization in NumPy.",
            "How do you debug a slow pandas operation?",
        ],
        "sql": [
            "Explain window functions with examples.",
            "How do you optimize slow SQL queries?",
            "Explain CTEs vs subqueries.",
            "How do you handle NULL values in SQL?",
        ],
        "ml": [
            "Explain bias-variance tradeoff.",
            "How do you handle imbalanced datasets?",
            "Explain cross-validation strategies.",
            "How do you select features for a model?",
        ],
        "visualization": [
            "When would you use a bar chart vs line chart?",
            "How do you handle plotting large datasets?",
            "Explain best practices for data visualization.",
            "How do you make visualizations accessible?",
        ],
    },
    project_scenarios=[
        "Build a customer segmentation model",
        "Design an A/B testing framework for a website",
        "Build a churn prediction system",
        "Create a real-time analytics dashboard",
        "Build a recommendation engine for e-commerce",
        "Design a data quality monitoring system",
    ],
    evaluation_criteria={
        "technical_depth": 0.25,
        "statistics": 0.20,
        "machine_learning": 0.20,
        "coding": 0.20,
        "communication": 0.15,
    },
    typical_questions=[
        "Explain a time you used data to drive a business decision.",
        "How do you handle missing or dirty data?",
        "Describe your approach to feature engineering.",
        "How do you communicate technical results to non-technical stakeholders?",
        "Explain your model evaluation strategy.",
        "How do you handle class imbalance?",
        "Describe a project where your model failed. What did you learn?",
        "How do you stay current with new ML techniques?",
    ],
    what_interviewers_look_for=[
        "Strong statistical foundations",
        "Practical ML implementation skills",
        "SQL proficiency",
        "Data storytelling ability",
        "Business acumen and problem framing",
    ],
    common_mistakes=[
        "Jumping to complex models without understanding the data",
        "Not validating assumptions",
        "Overfitting to training data",
        "Poor communication of results",
        "Ignoring data quality issues",
    ],
)

# ─────────────────────────────────────────────────────────────
# DevOps/SRE Module
# ─────────────────────────────────────────────────────────────

DEVOPS_MODULE = IndustryModule(
    industry=IndustryType.DEVOPS,
    name="DevOps/SRE",
    description="DevOps engineers build infrastructure, CI/CD pipelines, and ensure system reliability.",
    core_skills=[
        SkillAssessment(
            skill_name="infrastructure",
            weight=0.25,
            question_types=["system_design", "coding"],
            difficulty_range=["medium", "hard"],
            evaluation_rubric={"iaac": 0.4, "cloud": 0.3, "automation": 0.3},
        ),
        SkillAssessment(
            skill_name="cicd",
            weight=0.20,
            question_types=["system_design", "coding"],
            difficulty_range=["medium", "hard"],
            evaluation_rubric={"pipeline_design": 0.4, "testing": 0.3, "deployment": 0.3},
        ),
        SkillAssessment(
            skill_name="monitoring",
            weight=0.20,
            question_types=["system_design", "behavioral"],
            difficulty_range=["medium", "hard"],
            evaluation_rubric={"metrics": 0.4, "alerting": 0.3, "observability": 0.3},
        ),
        SkillAssessment(
            skill_name="containerization",
            weight=0.15,
            question_types=["coding", "system_design"],
            difficulty_range=["medium", "hard"],
            evaluation_rubric={"docker": 0.4, "kubernetes": 0.4, "orchestration": 0.2},
        ),
        SkillAssessment(
            skill_name="security",
            weight=0.10,
            question_types=["system_design", "behavioral"],
            difficulty_range=["medium", "hard"],
            evaluation_rubric={"hardening": 0.4, "secrets": 0.3, "compliance": 0.3},
        ),
        SkillAssessment(
            skill_name="incident_management",
            weight=0.10,
            question_types=["behavioral"],
            difficulty_range=["medium", "hard"],
            evaluation_rubric={"response": 0.4, "post_mortem": 0.3, "prevention": 0.3},
        ),
    ],
    system_design_topics=[
        "Design a CI/CD pipeline for microservices",
        "Build a monitoring and alerting system",
        "Design a disaster recovery system",
        "Build a secrets management system",
        "Design a multi-region deployment",
        "Build an infrastructure as code framework",
        "Design a cost optimization system",
        "Build a compliance automation system",
    ],
    coding_challenges=[
        "Write a script to automate server provisioning",
        "Build a simple container orchestration system",
        "Write infrastructure as code for a web application",
        "Build a log aggregation system",
        "Write a script to monitor service health",
        "Build a deployment pipeline with rollback",
        "Write code to manage DNS records",
        "Build a certificate management system",
    ],
    tool_questions={
        "docker": [
            "Explain Docker networking modes.",
            "How do you optimize Docker images?",
            "Explain Docker Compose vs Docker Swarm.",
            "How do you handle secrets in Docker?",
        ],
        "kubernetes": [
            "Explain Kubernetes pod lifecycle.",
            "How do you manage Kubernetes secrets?",
            "Explain horizontal vs vertical scaling.",
            "How do you debug a crashing pod?",
        ],
        "terraform": [
            "Explain Terraform state management.",
            "How do you handle Terraform modules?",
            "Explain Terraform workspaces.",
            "How do you manage drift in Terraform?",
        ],
        "monitoring": [
            "Explain the difference between metrics, logs, and traces.",
            "How do you set up effective alerting?",
            "Explain SLIs, SLOs, and SLAs.",
            "How do you handle alert fatigue?",
        ],
    },
    project_scenarios=[
        "Design a CI/CD pipeline for a microservices architecture",
        "Build a monitoring system for a distributed application",
        "Design a disaster recovery plan for a critical service",
        "Build an infrastructure as code module for AWS",
        "Design a cost optimization strategy for cloud resources",
        "Build a compliance automation system",
    ],
    evaluation_criteria={
        "infrastructure": 0.25,
        "cicd": 0.20,
        "monitoring": 0.20,
        "problem_solving": 0.15,
        "communication": 0.10,
        "security": 0.10,
    },
    typical_questions=[
        "How do you ensure high availability in your systems?",
        "Describe your approach to incident management.",
        "How do you handle configuration drift?",
        "Explain your monitoring and alerting strategy.",
        "How do you manage secrets in production?",
        "Describe a time you improved system reliability.",
        "How do you approach capacity planning?",
        "Explain your disaster recovery strategy.",
    ],
    what_interviewers_look_for=[
        "Strong infrastructure automation skills",
        "Reliability engineering mindset",
        "Security awareness",
        "Incident management experience",
        "Cost optimization thinking",
    ],
    common_mistakes=[
        "Overcomplicating infrastructure",
        "Ignoring security considerations",
        "Not having proper monitoring",
        "Poor incident response documentation",
        "Not considering cost implications",
    ],
)


# ─────────────────────────────────────────────────────────────
# Module Registry
# ─────────────────────────────────────────────────────────────

INDUSTRY_MODULES = {
    "backend": BACKEND_MODULE,
    "frontend": FRONTEND_MODULE,
    "data_science": DATA_SCIENCE_MODULE,
    "datascience": DATA_SCIENCE_MODULE,
    "devops": DEVOPS_MODULE,
    "sre": DEVOPS_MODULE,
}


def get_industry_module(industry: str) -> Optional[IndustryModule]:
    """Get industry module by name."""
    return INDUSTRY_MODULES.get(industry.lower())


def list_industries() -> List[Dict]:
    """List all available industry modules."""
    return [{
        "key": key,
        "name": module.name,
        "description": module.description,
        "core_skills": [s.skill_name for s in module.core_skills],
        "system_design_topics": len(module.system_design_topics),
        "coding_challenges": len(module.coding_challenges),
    } for key, module in INDUSTRY_MODULES.items()]


def get_questions_for_industry(industry: str, question_type: str = None) -> List[str]:
    """Get questions for a specific industry."""
    module = get_industry_module(industry)
    if not module:
        return []
    
    if question_type == "system_design":
        return module.system_design_topics
    elif question_type == "coding":
        return module.coding_challenges
    elif question_type == "behavioral":
        return module.typical_questions
    else:
        return module.typical_questions
