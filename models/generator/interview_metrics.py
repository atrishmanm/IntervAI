"""
models/generator/interview_metrics.py — Interview-Specific Accuracy
==================================================================
Specialized metrics for evaluating coding interview model quality.

Metrics:
  - Code correctness (compiles + runs)
  - BLEU/ROUGE for explanation quality
  - Concept accuracy (CS concepts mentioned)
  - Relevance scoring for follow-up questions
"""

import re
import subprocess
import tempfile
import os
from collections import Counter


# ─────────────────────────────────────────────────────────────
# Code Correctness (compile + run check)
# ─────────────────────────────────────────────────────────────

def check_code_correctness(code, test_cases=None, timeout=5):
    """Check if generated code compiles and runs correctly.

    Returns:
        dict with:
            - compiles: bool
            - runs: bool
            - test_passed: int (out of total)
            - error_msg: str or None
    """
    result = {"compiles": False, "runs": False, "test_passed": 0, "total_tests": 0, "error_msg": None}
    tmp_path = None

    # Clean code block markers
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    if code.startswith("python"):
        code = code[6:]

    # Try to compile
    try:
        compile(code, "<generated>", "exec")
        result["compiles"] = True
    except SyntaxError as e:
        result["error_msg"] = f"SyntaxError: {e}"
        return result

    # Try to run (with timeout)
    if test_cases:
        result["total_tests"] = len(test_cases)
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                f.write("\n\n# Test cases\n")
                for i, test in enumerate(test_cases):
                    f.write(f"try:\n")
                    f.write(f"    result = {test}\n")
                    f.write(f"    assert result, f'Test {i+1} failed'\n")
                    f.write(f"    print(f'PASS_{i}')\n")
                    f.write(f"except Exception as e:\n")
                    f.write(f"    print(f'FAIL_{i}: {{e}}')\n")
                tmp_path = f.name

            proc = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            result["runs"] = proc.returncode == 0
            result["test_passed"] = proc.stdout.count("PASS_")
            if not result["runs"]:
                result["error_msg"] = proc.stderr[:500] if proc.stderr else "Runtime error"
        except subprocess.TimeoutExpired:
            result["error_msg"] = "Timeout (>5s)"
        except Exception as e:
            result["error_msg"] = str(e)[:200]
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    else:
        # Just check if it runs without error
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                tmp_path = f.name
            proc = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            result["runs"] = proc.returncode == 0
            if not result["runs"]:
                result["error_msg"] = proc.stderr[:500] if proc.stderr else "Runtime error"
        except subprocess.TimeoutExpired:
            result["error_msg"] = "Timeout (>5s)"
        except Exception as e:
            result["error_msg"] = str(e)[:200]
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    return result


# ─────────────────────────────────────────────────────────────
# BLEU Score (explanation quality)
# ─────────────────────────────────────────────────────────────

def compute_bleu(reference, hypothesis, max_n=4):
    """Compute BLEU score between reference and hypothesis.

    Returns:
        float: BLEU score (0-1)
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not hyp_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, len(hyp_tokens) / max(len(ref_tokens), 1))

    # N-gram precision
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter()
        for i in range(len(ref_tokens) - n + 1):
            ngram = tuple(ref_tokens[i:i+n])
            ref_ngrams[ngram] += 1

        hyp_ngrams = Counter()
        for i in range(len(hyp_tokens) - n + 1):
            ngram = tuple(hyp_tokens[i:i+n])
            hyp_ngrams[ngram] += 1

        # Clip counts
        clipped = 0
        total = 0
        for ngram, count in hyp_ngrams.items():
            clipped += min(count, ref_ngrams.get(ngram, 0))
            total += count

        precisions.append(clipped / max(total, 1))

    # Geometric mean
    if any(p == 0 for p in precisions):
        return 0.0

    import math
    log_avg = sum(math.log(p) for p in precisions) / len(precisions)
    return bp * math.exp(log_avg)


# ─────────────────────────────────────────────────────────────
# ROUGE Score (explanation quality)
# ─────────────────────────────────────────────────────────────

def compute_rouge_l(reference, hypothesis):
    """Compute ROUGE-L F1 score between reference and hypothesis.

    Returns:
        float: ROUGE-L F1 (0-1)
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not ref_tokens or not hyp_tokens:
        return 0.0

    # LCS length
    lcs_len = _lcs_length(ref_tokens, hyp_tokens)

    precision = lcs_len / max(len(hyp_tokens), 1)
    recall = lcs_len / max(len(ref_tokens), 1)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def _lcs_length(x, y):
    """Compute length of longest common subsequence."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0

    # Optimize memory for large sequences
    if m * n > 1000000:
        # Use approximate LCS with window
        return _approximate_lcs(x, y, window=100)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def _approximate_lcs(x, y, window=100):
    """Approximate LCS for large sequences using sliding window."""
    m, n = len(x), len(y)
    matches = 0
    for i in range(m):
        start = max(0, i - window)
        end = min(n, i + window)
        for j in range(start, end):
            if x[i] == y[j]:
                matches += 1
                break
    return matches


# ─────────────────────────────────────────────────────────────
# Concept Accuracy (CS concepts)
# ─────────────────────────────────────────────────────────────

CS_CONCEPTS = {
    # Data structures
    "array": ["array", "list", "vector"],
    "linked_list": ["linked list", "node", "pointer"],
    "stack": ["stack", "push", "pop", "lifo"],
    "queue": ["queue", "enqueue", "dequeue", "fifo"],
    "hash_table": ["hash", "hashmap", "dictionary", "hashtable"],
    "tree": ["tree", "binary tree", "bst", "node"],
    "graph": ["graph", "vertex", "edge", "adjacency"],
    "heap": ["heap", "priority queue", "min-heap", "max-heap"],
    "trie": ["trie", "prefix tree"],

    # Algorithms
    "sorting": ["sort", "quicksort", "mergesort", "heapsort", "bubble sort"],
    "searching": ["binary search", "linear search", "bfs", "dfs"],
    "dynamic_programming": ["dynamic programming", "memoization", "tabulation", "dp"],
    "greedy": ["greedy", "greedy algorithm"],
    "recursion": ["recursion", "recursive", "base case", "call stack"],
    "memory": ["memory", "stack", "heap", "allocation"],
    "pointers": ["pointer", "reference", "address"],
    "oop": ["object", "class", "inheritance", "polymorphism"],
    "functional": ["functional", "pure function", "immutable"],
}


def compute_concept_accuracy(text, expected_concepts=None):
    """Check if text mentions relevant CS concepts.

    Args:
        text: Generated text to evaluate
        expected_concepts: list of concept keys to check (or None for auto-detect)

    Returns:
        dict with:
            - concepts_found: list of found concept keys
            - coverage: float (0-1) fraction of expected concepts found
            - total_concepts: int
    """
    text_lower = text.lower()

    found_concepts = []
    for concept_key, keywords in CS_CONCEPTS.items():
        for keyword in keywords:
            if keyword in text_lower:
                found_concepts.append(concept_key)
                break

    if expected_concepts:
        coverage = len(set(found_concepts) & set(expected_concepts)) / max(len(expected_concepts), 1)
    else:
        coverage = min(len(found_concepts) / 5.0, 1.0)  # Normalize to 0-1

    return {
        "concepts_found": list(set(found_concepts)),
        "coverage": coverage,
        "total_concepts": len(set(found_concepts)),
    }


# ─────────────────────────────────────────────────────────────
# Relevance Scoring (follow-up questions)
# ─────────────────────────────────────────────────────────────

def compute_relevance(answer, question, context=""):
    """Score how relevant a follow-up question/answer is to the context.

    Uses simple keyword overlap + semantic cues.

    Returns:
        float: relevance score (0-1)
    """
    # Extract keywords (remove stopwords)
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "being", "have", "has", "had", "do", "does", "did", "will",
                 "would", "could", "should", "may", "might", "can", "shall",
                 "to", "of", "in", "for", "on", "with", "at", "by", "from",
                 "as", "into", "through", "during", "before", "after", "above",
                 "below", "between", "under", "again", "further", "then", "once",
                 "i", "you", "he", "she", "it", "we", "they", "this", "that",
                 "these", "those", "what", "which", "who", "whom", "how", "when",
                 "where", "why", "not", "no", "nor", "but", "or", "and", "so",
                 "if", "than", "too", "very", "just", "about", "also", "only"}

    def extract_keywords(text):
        words = re.findall(r'\b\w+\b', text.lower())
        return set(w for w in words if w not in stopwords and len(w) > 2)

    q_kw = extract_keywords(question)
    a_kw = extract_keywords(answer)
    ctx_kw = extract_keywords(context) if context else set()

    # Keyword overlap
    if ctx_kw:
        overlap_ctx = len(a_kw & ctx_kw) / max(len(ctx_kw), 1)
    else:
        overlap_ctx = 0.5

    overlap_q = len(a_kw & q_kw) / max(len(q_kw), 1) if q_kw else 0

    # Check for question words in answer (indicates relevance)
    question_words = {"why", "how", "explain", "what", "because", "since", "therefore"}
    has_explanation = bool(a_kw & question_words)

    # Score
    score = 0.4 * overlap_ctx + 0.3 * overlap_q + 0.3 * (1.0 if has_explanation else 0.5)
    return min(score, 1.0)


# ─────────────────────────────────────────────────────────────
# Combined Interview Score
# ─────────────────────────────────────────────────────────────

def compute_interview_score(response, reference="", question="", code=None, test_cases=None):
    """Compute comprehensive interview quality score.

    Returns:
        dict with all individual scores + combined score
    """
    scores = {}

    # Code correctness
    if code:
        code_result = check_code_correctness(code, test_cases)
        scores["code_compiles"] = 1.0 if code_result["compiles"] else 0.0
        scores["code_runs"] = 1.0 if code_result["runs"] else 0.0
        if code_result["total_tests"] > 0:
            scores["code_test_pass"] = code_result["test_passed"] / code_result["total_tests"]
        else:
            scores["code_test_pass"] = scores["code_runs"]
    else:
        scores["code_compiles"] = 0.0
        scores["code_runs"] = 0.0
        scores["code_test_pass"] = 0.0

    # Explanation quality
    if reference:
        scores["bleu"] = compute_bleu(reference, response)
        scores["rouge_l"] = compute_rouge_l(reference, response)
    else:
        scores["bleu"] = 0.0
        scores["rouge_l"] = 0.0

    # Concept accuracy
    concept_result = compute_concept_accuracy(response)
    scores["concept_coverage"] = concept_result["coverage"]
    scores["concepts_found"] = concept_result["total_concepts"]

    # Relevance
    scores["relevance"] = compute_relevance(response, question, reference)

    # Combined score (weighted average)
    weights = {
        "code_test_pass": 0.30,
        "bleu": 0.15,
        "rouge_l": 0.15,
        "concept_coverage": 0.20,
        "relevance": 0.20,
    }
    scores["combined_score"] = sum(scores[k] * w for k, w in weights.items())

    return scores
