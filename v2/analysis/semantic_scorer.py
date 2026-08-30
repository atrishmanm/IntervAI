"""
analysis/semantic_scorer.py
============================
Contextual answer evaluation — the "ChatGPT-like" part.

GOAL: evaluate a candidate's answer the way a human interviewer would:
  - Does it UNDERSTAND the concept (semantic, not just keyword match)?
  - Is it ACCURATE (doesn't contradict the reference)?
  - Is it COMPLETE (covers required concepts / reasoning steps)?
  - Is it CLEAR / well-structured (depth, reasoning, example)?

It does NOT simply count matched keywords. It uses:
  1. Semantic similarity (sentence embeddings via the ranker model or TF-IDF fallback)
  2. Concept coverage (each expected concept checked semantically)
  3. Accuracy (contradiction check vs reference)
  4. Structure/depth heuristics
  5. Human-like feedback text generation

Returns an AnalysisReport dict that the orchestrator + frontend consume.
"""

import math
import re

# Optional ML imports (fall back to pure heuristics if unavailable)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _norm_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse spaces."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ─────────────────────────────────────────────────────────────
# Embedding provider (plug the ranker bi-encoder here when available)
# ─────────────────────────────────────────────────────────────

class EmbeddingProvider:
    """Provides semantic embeddings for text.
    Uses the trained QuestionRanker's document tower if available;
    falls back to a hashed-ngram TF-IDF-like embedding otherwise.
    """
    def __init__(self):
        self._ranker = None
        self._tokenizer = None

    def _try_load_ranker(self):
        """Try to load the trained ranker for real semantic embeddings."""
        if self._ranker is not None:
            return self._ranker is not None
        try:
            import sys
            from pathlib import Path
            root = Path(__file__).resolve().parent.parent
            sys.path.insert(0, str(root))
            import torch
            from models.ranker.model import QuestionRanker, ENCODER_CONFIG
            ckpt_path = root / "models" / "ranker" / "saved" / "best_ranker.pt"
            if not ckpt_path.exists():
                return False
            # Build ranker matching its training config
            from tokenizers import Tokenizer
            tok_path = root / "tokenizer" / "saved" / "tokenizer.json"
            self._tokenizer = Tokenizer.from_file(str(tok_path))
            vocab_size = self._tokenizer.get_vocab_size()
            self._ranker = QuestionRanker(vocab_size=vocab_size, **ENCODER_CONFIG)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self._ranker.load_state_dict(ckpt["model_state_dict"])
            self._ranker.eval()
            return True
        except Exception:
            return False

    def embed(self, text: str):
        """Return a normalized embedding vector for text."""
        if self._try_load_ranker():
            return self._embed_ranker(text)
        return self._embed_ngram(text)

    def _embed_ranker(self, text):
        import torch
        enc = self._tokenizer.encode(text)
        ids = enc.ids[:256]
        pad = self._tokenizer.token_to_id("[PAD]") or 0
        ids = ids + [pad] * (256 - len(ids))
        t = torch.tensor([ids])
        with torch.no_grad():
            vec = self._ranker.encode_document(t)
        return vec.squeeze(0).numpy()

    def _embed_ngram(self, text, dim=512):
        """Deterministic hashed 3-gram embedding (fallback)."""
        import hashlib
        if not HAS_NUMPY:
            return None
        vec = np.zeros(dim, dtype=np.float32)
        norm = _norm_text(text)
        tokens = norm.split()
        grams = []
        for i in range(len(tokens)):
            grams.append(tokens[i])
            if i > 0:
                grams.append(tokens[i-1] + "_" + tokens[i])
            if i > 1:
                grams.append(tokens[i-2] + "_" + tokens[i-1] + "_" + tokens[i])
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest()[:8], 16) % dim
            vec[h] += 1.0
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        return vec

    def cosine(self, a, b) -> float:
        if a is None or b is None:
            return 0.0
        if HAS_NUMPY:
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        return 0.0


_embedder = EmbeddingProvider()


# ─────────────────────────────────────────────────────────────
# Concept lexicon — canonical phrases per concept
# ─────────────────────────────────────────────────────────────

CONCEPT_PHRASES = {
    "array": ["contiguous memory", "index", "random access", "fixed size"],
    "linked list": ["node", "pointer", "next", "head", "linked"],
    "stack": ["lifo", "last in first out", "push", "pop", "peek"],
    "queue": ["fifo", "first in first out", "enqueue", "dequeue", "front", "rear"],
    "hash table": ["hash function", "key value", "bucket", "collision", "hash map"],
    "binary search": ["sorted array", "divide in half", "midpoint", "log n", "search interval"],
    "time complexity": ["o(", "big o", "log n", "o(n)", "o(n log n)", "o(n^2)", "constant time", "linear"],
    "space complexity": ["o(", "auxiliary", "memory", "space", "in place", "recursion stack"],
    "recursion": ["base case", "calls itself", "recursive", "stack frame", "base condition"],
    "tree": ["root", "node", "child", "leaf", "parent", "binary tree", "subtree"],
    "binary search tree": ["left subtree", "right subtree", "less than", "greater than", "ordered"],
    "graph": ["vertices", "edges", "adjacency", "node", "neighbors", "directed", "undirected"],
    "bfs": ["queue", "level by level", "shortest path", "visited", "explore neighbors"],
    "dfs": ["stack", "recursion", "depth first", "visited", "backtrack"],
    "dynamic programming": ["overlapping subproblems", "memoization", "tabulation", "optimal substructure", "state"],
    "greedy": ["local optimum", "greedy choice", "optimal substructure", "greedily"],
    "sorting": ["bubble", "merge", "quick", "heap", "insertion", "stable", "partition", "compare"],
    "heap": ["complete binary tree", "max heap", "min heap", "heapify", "priority queue"],
    "trie": ["prefix", "character", "word", "autocomplete", "prefix tree"],
    "two pointers": ["two indices", "left pointer", "right pointer", "opposite ends"],
    "sliding window": ["window", "expand", "shrink", "subarray", "substring"],
    "pointer": ["reference", "address", "memory location", "pointer"],
    "complexity analysis": ["o(", "big o", "growth", "asymptotic", "worst case", "average case"],
    "data structure": ["store", "organize", "efficient", "insert", "delete", "search"],
    "algorithm": ["steps", "procedure", "solve", "efficient", "approach"],
    "object oriented": ["class", "object", "encapsulation", "inheritance", "polymorphism", "abstraction"],
    "sql": ["select", "table", "query", "join", "database", "where"],
    "os": ["process", "thread", "memory", "scheduler", "deadlock", "mutex"],
    "networking": ["tcp", "udp", "packet", "protocol", "ip", "http", "socket"],
    "string": ["substring", "character", "concatenate", "manipulation"],
    "bit manipulation": ["bit", "xor", "shift", "mask", "binary"],
    "backtracking": ["try", "undo", "explore", "constraint", "prune", "recurse"],
}


def extract_concepts_from_reference(reference: str) -> list:
    """Guess the concepts a question is testing from the reference answer."""
    ref_lower = reference.lower()
    found = []
    for concept, phrases in CONCEPT_PHRASES.items():
        hit = sum(1 for p in phrases if p in ref_lower)
        if hit >= 1:
            found.append((concept, hit))
    found.sort(key=lambda x: -x[1])
    return [c for c, _ in found[:5]]


# ─────────────────────────────────────────────────────────────
# Core scoring
# ─────────────────────────────────────────────────────────────

def score_answer_contextual(
    student_answer: str,
    reference_answer: str,
    key_phrases: list = None,
    question_type: str = "theoretical",
    expert_text: str = "",
    question: str = "",
) -> dict:
    """
    Evaluate a candidate's answer contextually.

    Returns a dict:
      {
        "score": 0-100,
        "verdict": str,
        "semantic_similarity": 0-1,
        "concepts": [{name, status: pass|partial|miss, detail}],
        "accuracy": {status, detail},
        "completeness": {status, detail},
        "quality": {status, detail},
        "feedback": str,
        "missing_concepts": [str],
        "covered_concepts": [str],
        "suggested_followup": str,
      }
    """
    answer = student_answer.strip()
    reference = (reference_answer or expert_text or "").strip()
    if not reference:
        reference = " ".join(key_phrases or [])

    # ── 1. Semantic similarity ──
    emb_a = _embedder.embed(answer)
    emb_r = _embedder.embed(reference)
    sim = _embedder.cosine(emb_a, emb_r) if emb_a is not None else 0.0

    # ── 2. Concept coverage ──
    concepts = extract_concepts_from_reference(reference)
    concept_results = []
    covered = []
    missing = []

    for concept in concepts:
        phrases = CONCEPT_PHRASES[concept]
        # Semantic check: embed each canonical phrase, take max similarity with answer
        phrase_hits = []
        for phrase in phrases[:4]:
            emb_p = _embedder.embed(phrase)
            s = _embedder.cosine(emb_a, emb_p) if emb_p is not None else 0.0
            phrase_hits.append(s)
        best_sim = max(phrase_hits) if phrase_hits else 0.0

        # Keyword fallback check
        kw_hits = sum(1 for p in phrases if p.lower() in answer.lower())

        if best_sim >= 0.55 or kw_hits >= 1:
            concept_results.append({"name": concept, "status": "pass",
                                    "detail": f"'{concept}' addressed"})
            covered.append(concept)
        elif best_sim >= 0.35:
            concept_results.append({"name": concept, "status": "partial",
                                    "detail": f"'{concept}' partially addressed"})
            missing.append(concept)
        else:
            concept_results.append({"name": concept, "status": "miss",
                                    "detail": f"'{concept}' not addressed"})
            missing.append(concept)

    coverage_ratio = len(covered) / max(len(concepts), 1)

    # ── 3. Accuracy (contradiction vs reference) ──
    # Simple heuristic: if the answer mentions a key term but framed negatively,
    # or if semantic similarity to reference is very low while length is high → suspect.
    accuracy_status = "pass"
    accuracy_detail = "No obvious contradictions with the expected answer."
    neg_words = ["not ", "isn't", "doesn't", "wrong", "incorrectly", "cannot be", "never"]
    if len(covered) == 0 and len(answer) > 30 and sim < 0.15:
        accuracy_status = "fail"
        accuracy_detail = "Answer does not appear to align with the expected response — check for a misunderstanding."
    elif any(w in answer.lower() for w in neg_words) and sim < 0.2:
        accuracy_status = "partial"
        accuracy_detail = "Answer contains negations that may contradict the expected response."

    # ── 4. Completeness ──
    completeness_status = "pass"
    completeness_detail = "Answer covers the expected material."
    if coverage_ratio >= 0.8:
        pass
    elif coverage_ratio >= 0.4:
        completeness_status = "partial"
        completeness_detail = f"Answer covers {len(covered)}/{len(concepts)} expected concepts. Missing: {', '.join(missing)}"
    else:
        completeness_status = "fail"
        completeness_detail = f"Answer covers only {len(covered)}/{len(concepts)} expected concepts. Missing: {', '.join(missing)}"

    # ── 5. Quality / depth ──
    word_count = len(answer.split())
    has_code = bool(re.search(r"```|def |class |function |return ", answer))
    struct_words = ["because", "therefore", "if", "then", "since", "while", "for", "however", "thus"]
    struct_count = sum(1 for w in struct_words if w in answer.lower())

    quality_status = "pass"
    quality_detail = "Well-explained with reasoning."
    if word_count < 10:
        quality_status = "fail"
        quality_detail = f"Answer is too brief ({word_count} words) — explain your reasoning."
    elif word_count < 20 and not has_code:
        quality_status = "partial"
        quality_detail = f"Answer is a bit brief ({word_count} words); add more reasoning or an example."

    # ── 6. Composite score ──
    weights = {"sim": 0.30, "coverage": 0.35, "accuracy": 0.15, "completeness": 0.10, "quality": 0.10}
    sim_score = max(0.0, min(1.0, sim)) * 100
    cov_score = coverage_ratio * 100
    acc_score = {"pass": 100, "partial": 55, "fail": 20}[accuracy_status]
    comp_score = {"pass": 100, "partial": 55, "fail": 20}[completeness_status]
    qual_score = {"pass": 100, "partial": 55, "fail": 20}[quality_status]

    score = int(
        weights["sim"] * sim_score
        + weights["coverage"] * cov_score
        + weights["accuracy"] * acc_score
        + weights["completeness"] * comp_score
        + weights["quality"] * qual_score
    )

    if score >= 80:
        verdict = "Excellent"
    elif score >= 55:
        verdict = "Good"
    elif score >= 35:
        verdict = "Needs Improvement"
    else:
        verdict = "Off Track"

    # ── 7. Human-like feedback ──
    feedback = _generate_feedback(score, verdict, covered, missing, completeness_detail)

    # ── 8. Follow-up suggestion ──
    suggested_followup = _suggest_followup(missing, question_type, question)

    return {
        "score": score,
        "max_score": 100,
        "verdict": verdict,
        "semantic_similarity": round(sim, 3),
        "concepts": concept_results,
        "covered_concepts": covered,
        "missing_concepts": missing,
        "accuracy": {"status": accuracy_status, "detail": accuracy_detail},
        "completeness": {"status": completeness_status, "detail": completeness_detail},
        "quality": {"status": quality_status, "detail": quality_detail,
                    "word_count": word_count, "has_code": has_code},
        "feedback": feedback,
        "suggested_followup": suggested_followup,
    }


# ─────────────────────────────────────────────────────────────
# Feedback + follow-up generation
# ─────────────────────────────────────────────────────────────

def _generate_feedback(score, verdict, covered, missing, completeness_detail):
    if verdict == "Excellent":
        return (
            "That's a strong answer. You correctly addressed the key concepts and "
            "explained your reasoning well. An interviewer would be satisfied with "
            "this level of understanding."
        )
    if verdict == "Good":
        if missing:
            return (
                f"Good effort — you understood the core idea. To strengthen it, "
                f"also touch on: {', '.join(missing)}. Interviewers look for a "
                f"complete picture, not just the main mechanism."
            )
        return (
            "Good answer with solid understanding. Adding a concrete example or "
            "mentioning complexity would make it excellent."
        )
    if verdict == "Needs Improvement":
        return (
            f"You have the general idea but the answer is incomplete. "
            f"{completeness_detail} Try structuring your answer as: concept definition, "
            f"how it works, complexity, and an example."
        )
    return (
        "Your answer doesn't align well with the expected response yet. "
        "Let's work through it: first define what the concept is, then how it works, "
        "then its complexity and edge cases. Give it another try."
    )


def _suggest_followup(missing, question_type, question):
    if not missing:
        return "Let's move on to the next question."
    top = missing[0]
    if question_type in ("theoretical", "concept"):
        return (
            f"You mentioned the main idea — can you elaborate specifically on "
            f"'{top}' and why it matters here?"
        )
    return f"Good start. Now, could you explain the '{top}' aspect in more detail?"


# ─────────────────────────────────────────────────────────────
# Backward-compatible alias (used by the existing inference service)
# ─────────────────────────────────────────────────────────────

def score_answer(
    student_answer, reference_answer, key_phrases=None,
    question_type="theoretical", expert_text="", expert_score=1.0, question=""
):
    """Drop-in that returns the full contextual report."""
    return score_answer_contextual(
        student_answer=student_answer,
        reference_answer=reference_answer,
        key_phrases=key_phrases or [],
        question_type=question_type,
        expert_text=expert_text,
        question=question,
    )


if __name__ == "__main__":
    print("Semantic Scorer — self test")
    print("-" * 60)
    tests = [
        ("What is binary search?",
         "Binary search works by repeatedly dividing the search interval in half. "
         "It compares the target with the middle element and discards half each time. "
         "It requires a sorted array and runs in O(log n) time.",
         "Binary search is a divide-and-conquer algorithm on a sorted array. "
         "Compare target with mid, discard half of the search space each step, "
         "repeat until found. Time complexity O(log n), space O(1)."),
        ("What is a stack?",
         "It's like a queue.",
         "A stack is a LIFO data structure. Elements are pushed on top and popped "
         "from the top. Operations are push, pop, peek, all O(1)."),
    ]
    for q, cand, ref in tests:
        res = score_answer_contextual(cand, ref, question=q)
        print(f"\nQ: {q}")
        print(f"  Score: {res['score']} ({res['verdict']})  sim={res['semantic_similarity']}")
        print(f"  Covered: {res['covered_concepts']}")
        print(f"  Missing: {res['missing_concepts']}")
        print(f"  Feedback: {res['feedback']}")
