"""
analysis/scorer.py
==================
Scores a student's answer against 5 strict analytical dimensions:
  1. Algorithmic Approach
  2. Time Complexity
  3. Space Complexity
  4. Edge Case Awareness
  5. Communication / Depth

Returns a highly structured, data-driven analysis report.
"""

import json
import re
from dataclasses import dataclass, field


@dataclass
class AnalysisReport:
    score: int = 0
    max_score: int = 100
    verdict: str = ""
    checks: list = field(default_factory=list)
    expert_reference: str = ""
    follow_up: str = ""
    
    def to_dict(self):
        return {
            "score": self.score,
            "max_score": self.max_score,
            "verdict": self.verdict,
            "checks": self.checks,
            "expert_reference": self.expert_reference,
            "follow_up": self.follow_up,
        }


def extract_big_o(text: str) -> set:
    """Extract all O(...) notations from text."""
    matches = re.findall(r"[Oo]\s*\(\s*([^\)]+)\s*\)", text)
    # Also look for word equivalents
    words = text.lower()
    implicit = []
    if "constant time" in words or "constant space" in words: implicit.append("1")
    if "linear time" in words or "linear space" in words: implicit.append("n")
    if "logarithmic" in words: implicit.append("log n")
    if "quadratic" in words: implicit.append("n^2")
    
    return set([m.lower().replace(" ", "") for m in matches] + implicit)


def score_answer(
    student_answer: str,
    reference_answer: str,
    key_phrases: list,
    question_type: str,
    expert_text: str = "",
    expert_score: float = 1.0,  # Legacy
) -> AnalysisReport:
    report = AnalysisReport()
    report.expert_reference = expert_text[:600] if expert_text else ""
    
    student_lower = student_answer.lower()
    points = 0
    max_points = 0
    
    # ─── Factor 1: Algorithmic Approach / Output Correctness (30 pts) ───
    max_points += 30
    
    if question_type == "output_prediction":
        # Check if the expected output is present in the student's answer
        # Fuzzy match: strip spaces and quotes
        expected = key_phrases[0] if key_phrases else ""
        expected_clean = re.sub(r"[\s\"']", "", expected.lower())
        student_clean = re.sub(r"[\s\"']", "", student_lower)
        
        if expected_clean and expected_clean in student_clean:
            points += 30
            report.checks.append({"label": "Output Correctness", "status": "pass", "detail": f"Correct output predicted: {expected}"})
        else:
            report.checks.append({"label": "Output Correctness", "status": "fail", "detail": f"Incorrect output. Expected: {expected}"})
    else:
        if key_phrases:
            matched = 0
            missed = []
            for kp in key_phrases:
                if kp.lower() in student_lower:
                    matched += 1
                else:
                    missed.append(kp)
            
            ratio = matched / len(key_phrases)
            points += int(ratio * 30)
            
            if ratio == 1.0:
                report.checks.append({"label": "Algorithm / Concept", "status": "pass", "detail": "Identified correct approach."})
            elif ratio > 0:
                report.checks.append({"label": "Algorithm / Concept", "status": "partial", "detail": f"Missing some elements: {', '.join(missed)}"})
            else:
                report.checks.append({"label": "Algorithm / Concept", "status": "fail", "detail": "Did not identify the expected approach/concept."})
        else:
            # Freebie if no key phrases
            points += 30
            report.checks.append({"label": "Algorithm / Concept", "status": "pass", "detail": "General concept understood."})

    # ─── Factor 2 & 3: Time and Space Complexity (20 + 20 pts) ───
    # We only heavily penalize this for theoretical or concept questions
    if question_type in ("theoretical", "concept"):
        max_points += 40
        student_big_o = extract_big_o(student_answer)
        ref_big_o = extract_big_o(reference_answer + " " + expert_text)
        
        # If the question is specifically about time or space, we can be more lenient if they just provide the answer
        has_time = "time" in student_lower or "O(" in student_answer or "constant" in student_lower or "linear" in student_lower
        has_space = "space" in student_lower or "O(" in student_answer or "constant" in student_lower or "linear" in student_lower
        
        if has_time and has_space:
            points += 40
            report.checks.append({"label": "Complexity Analysis", "status": "pass", "detail": "Mentioned both Time and Space complexity."})
        elif has_time or has_space:
            points += 20
            report.checks.append({"label": "Complexity Analysis", "status": "partial", "detail": "Mentioned either Time or Space, but not both."})
        else:
            report.checks.append({"label": "Complexity Analysis", "status": "fail", "detail": "Missing Big-O complexity analysis."})
    else:
        # Output prediction doesn't strictly need complexity
        report.checks.append({"label": "Complexity Analysis", "status": "pass", "detail": "Not required for code execution questions."})

    # ─── Factor 4: Edge Case Awareness (15 pts) ───
    max_points += 15
    edge_cases = ["empty", "null", "none", "zero", "negative", "duplicate", "bounds", "overflow", "edge case", "single"]
    found_edges = [e for e in edge_cases if e in student_lower]
    
    if found_edges:
        points += 15
        report.checks.append({"label": "Edge Cases", "status": "pass", "detail": f"Considered constraints: {', '.join(found_edges)}"})
    elif question_type == "output_prediction":
        points += 15  # Not strictly required for output prediction
        report.checks.append({"label": "Edge Cases", "status": "pass", "detail": "Not required for code execution questions."})
    else:
        report.checks.append({"label": "Edge Cases", "status": "partial", "detail": "No edge cases or constraints mentioned."})

    # ─── Factor 5: Communication Depth (15 pts) ───
    max_points += 15
    word_count = len(student_answer.split())
    if question_type == "output_prediction":
        if word_count >= 1:
            points += 15
            report.checks.append({"label": "Communication", "status": "pass", "detail": "Answer provided clearly."})
        else:
            report.checks.append({"label": "Communication", "status": "fail", "detail": "No answer provided."})
    else:
        struct_words = ["because", "therefore", "if", "then", "since", "while", "for", "however"]
        struct_count = sum(1 for w in struct_words if w in student_lower)
        
        if word_count >= 20 and struct_count >= 1:
            points += 15
            report.checks.append({"label": "Communication", "status": "pass", "detail": f"Detailed and reasoned ({word_count} words)."})
        elif word_count >= 10:
            points += 10
            report.checks.append({"label": "Communication", "status": "partial", "detail": f"Slightly brief ({word_count} words)."})
        else:
            report.checks.append({"label": "Communication", "status": "fail", "detail": f"Too brief ({word_count} words), explain your reasoning."})
    
    # ─── Final Score ───
    report.score = int((points / max(max_points, 1)) * 100)
    report.max_score = 100
    
    if report.score >= 80:
        report.verdict = "Excellent"
    elif report.score >= 50:
        report.verdict = "Passable"
    else:
        report.verdict = "Needs Improvement"
    
    # ─── Follow-up Question ───
    if report.score >= 80:
        report.follow_up = "Flawless! Let's move on to the next challenge."
    elif report.score >= 50:
        report.follow_up = "You got the core idea, but review the expert reference to see what details you missed (like edge cases or space complexity)."
    else:
        report.follow_up = "Review the expert reference below carefully to understand the required approach and constraints."
    
    return report
