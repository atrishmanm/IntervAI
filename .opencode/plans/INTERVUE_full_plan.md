# INTERVUE: ChatGPT-Like Coding Interview Model — Full Plan

## Goal
Build a **10-50M parameter decoder-only Transformer** trained from scratch on coding/CS data, capable of conducting natural technical interviews. Not a general-purpose ChatGPT — a specialized AI interviewer that understands CS concepts, evaluates answers, generates follow-ups, and adapts to candidates.

## Constraints
- **GPU**: GTX 1650 4GB (local) + Colab/Kaggle free tier (T4 16GB)
- **Timeline**: 3+ months (UROP/school project)
- **From scratch**: No pretrained LLM fine-tuning — train our own model
- **Domain**: CS/programming interviews only

---

## Phase 0: Data Engineering (Week 1-2)

### 0.1 — Acquire All Datasets

Replace the current 10K-capped download with a comprehensive data collection:

| Dataset | HuggingFace ID | Size | Purpose |
|---------|---------------|------|---------|
| **StandardLogic** | `stindardlogic/coding-interview-sft-100k` | 100K conversations | Multi-turn interview dialogues |
| **CodeAlpaca** | `sahil2801/CodeAlpaca-20k` | 20K instruction/response | Code instruction following |
| **CodeFeedback** | `m-a-p/CodeFeedback-Filtered-Instruction` | 156K instructions | High-quality code Q&A (filtered for complexity 4-5) |
| **Mohler ASAG** | `nkazi/MohlerASAG` | 2,442 student answers | Real student answers with human grades (0-5) |
| **CRUXEval** | `cruxeval-org/cruxeval` | 800 code tracing | Code reasoning examples |
| **MMLU CS** | Already in project | ~500 questions | Conceptual CS knowledge |
| **OpenCodeInstruct** | `nvidia/OpenCodeInstruct` | 5M samples | Largest code instruction dataset |
| **StarCoder Data** | `bigcode/starcoderdata` | 783GB (streaming) | Code pretraining corpus |

**Action items:**
1. Update `data_pipeline/download_datasets.py` to download ALL of the above
2. Remove the 10K cap on StandardLogic — download full 100K
3. Add streaming for StarCoder (too large to download fully — stream during training)
4. Save all raw data to `data/raw/`

### 0.2 — Transform Raw Data into Training Formats

Create `data_pipeline/transform_training_data.py` — converts raw datasets into structured training examples.

**For each raw dataset, generate multiple training formats:**

#### Format A: Instruction Tuning (for chat capability)
```json
{
  "messages": [
    {"role": "system", "content": "You are an expert technical interviewer..."},
    {"role": "user", "content": "What is a binary search tree?"},
    {"role": "assistant", "content": "A binary search tree (BST) is a binary tree data structure..."}
  ]
}
```

#### Format B: Answer Evaluation (for scoring capability)
```json
{
  "question": "What is binary search?",
  "reference_answer": "Binary search works on sorted arrays...",
  "student_answer": "It divides the array in half...",
  "score": 3.5,
  "grade": "partially_correct",
  "feedback": "You understood the division mechanism but missed the sorted-array requirement and complexity analysis."
}
```

#### Format C: Follow-up Generation (for adaptive questioning)
```json
{
  "question": "What is binary search?",
  "student_answer": "It divides the array in half repeatedly.",
  "concepts_covered": ["divide_search_space", "recursive_idea"],
  "concepts_missing": ["sorted_array", "O(log_n)"],
  "followup": "Why must the array be sorted before applying binary search?"
}
```

#### Format D: Concept Explanation (for knowledge)
```json
{
  "concept": "Binary Search",
  "explanation": "Binary search is a divide-and-conquer algorithm that works on sorted arrays...",
  "prerequisites": ["arrays", "comparison", "divide_and_conquer"],
  "common_mistakes": ["unsorted input", "off-by-one errors", "infinite loops"],
  "difficulty": "medium"
}
```

#### Format E: Interview Dialogue (for conversational flow)
```json
{
  "dialogue": [
    {"role": "interviewer", "content": "Can you explain what a hash table is?"},
    {"role": "candidate", "content": "It's a data structure that maps keys to values using a hash function."},
    {"role": "interviewer", "content": "Good. What happens when two keys hash to the same index?"},
    {"role": "candidate", "content": "That's called a collision. You can handle it with chaining or open addressing."},
    {"role": "interviewer", "content": "Correct. Can you explain the difference between those two collision resolution strategies?"}
  ]
}
```

### 0.3 — Build Concept Graph

Create `data_pipeline/build_concept_graph.py` — builds a knowledge structure from the datasets.

Store as JSON in `data/processed/concept_graph.json`:
```json
{
  "binary_search": {
    "prerequisites": ["arrays", "comparison", "divide_and_conquer"],
    "concepts": ["sorted_array", "midpoint", "search_space"],
    "complexity": {"time": "O(log n)", "space": "O(1)"},
    "common_mistakes": ["unsorted_input", "off_by_one", "infinite_loop"],
    "related": ["linear_search", "interpolation_search"],
    "followups": [
      "Why must the array be sorted?",
      "How do you handle duplicates?",
      "What about rotated sorted arrays?"
    ],
    "difficulty": "medium"
  }
}
```

### 0.4 — Create Candidate Knowledge State Model

Create `orchestrator/candidate_state.py`:

```json
{
  "candidate_id": "session_123",
  "concept_scores": {
    "arrays": 0.85,
    "linked_lists": 0.72,
    "binary_search": 0.45,
    "hash_tables": 0.60,
    "trees": 0.30,
    "graphs": 0.20,
    "sorting": 0.75,
    "dynamic_programming": 0.15
  },
  "weaknesses": ["binary_search", "graphs", "dynamic_programming"],
  "strengths": ["arrays", "sorting"],
  "current_difficulty": "easy",
  "questions_answered": 5,
  "average_score": 65
}
```

**Difficulty adaptation logic:**
- If average_score >= 75 for last 3 questions → increase difficulty
- If average_score <= 40 for last 2 questions → decrease difficulty
- If concept_score < 0.5 → prioritize that concept's prerequisites first

---

## Phase 1: Tokenizer Upgrade (Week 2-3)

### 1.1 — Retrain Tokenizer with Larger Vocab

Current tokenizer: 8K vocab (too small for a generative model).

**New tokenizer config:**
- **Vocab size**: 16,000
- **Training corpus**: ALL datasets combined
- **Pre-tokenization**: ByteLevel (handles code syntax better)
- **Special tokens**:
  - `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`
  - `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`
  - `<|code|>`, `<|/code|>`
- **Max length**: 1024 tokens

### 1.2 — Update Dialogues Format

Change to ChatML-style:
```
<|system|> You are an expert technical interviewer.<|end|>
<|user|> question <|end|>
<|assistant|> answer <|end|>
```

---

## Phase 2: Model Architecture (Week 3-4)

### 2.1 — Decoder-Only Transformer

Create `models/generator/model.py` — a GPT-style decoder-only Transformer.

**Architecture specification (targeting ~20M params):**

| Component | Dimension | Parameters |
|-----------|-----------|------------|
| Token embedding | 16000 × 256 | 4,096,000 |
| Position embedding | 1024 × 256 | 262,144 |
| Transformer layer × 8 | — | ~14M |
| LayerNorm × 16 | 256 | 4,096 |
| LM head (tied) | — | 0 |
| **Total** | | **~18-20M** |

**Key design choices:**
- **Pre-LayerNorm** — more stable training
- **Learned positional encoding**
- **GELU activation** in FFN
- **Weight tying** — LM head shares weights with token embedding

### 2.2 — Alternative Architecture Sizes

| Config | embed_dim | n_layers | ff_dim | n_heads | ~Params |
|--------|-----------|----------|--------|---------|---------|
| **Small** | 192 | 6 | 768 | 4 | ~8M |
| **Medium** | 256 | 8 | 1024 | 4 | ~20M |
| **Large** | 384 | 10 | 1536 | 6 | ~45M |

---

## Phase 3: Training Pipeline (Week 4-10)

### 3.1 — Stage 1: Code Pretraining
- **Data**: Stream from StarCoder (sample ~10M lines)
- **Objective**: Next token prediction on raw code
- **LR**: 3e-4, cosine decay, 1000 warmup steps
- **Batch**: 8 (GTX 1650) or 32 (Colab T4)
- **Epochs**: 2-3

### 3.2 — Stage 2: CS Domain Training
- **Data**: StandardLogic 100K + CodeAlpaca 20K + CodeFeedback 156K + MMLU + CRUXEval
- **Format**: ChatML conversations
- **LR**: 1e-4
- **Epochs**: 3-5

### 3.3 — Stage 3: Instruction Tuning
- **Data**: Instruction-format examples from all datasets
- **LR**: 5e-5
- **Epochs**: 2-3

### 3.4 — Stage 4: Interview Dialogue Tuning (MOST CRITICAL)
- **Data**: Transformed interview dialogues
- **Teaches**: Asking questions, evaluating answers, generating follow-ups
- **LR**: 2e-5
- **Epochs**: 2-3

### 3.5 — Stage 5: Answer Evaluation Tuning
- **Data**: Mohler ASAG (2,442 real student answers) + synthetic evaluations
- **Teaches**: Scoring answers, identifying gaps, providing feedback
- **LR**: 2e-5
- **Epochs**: 3-5

### 3.6 — Stage 6: Follow-up Generation Tuning
- **Data**: Follow-up training examples from question bank
- **Teaches**: Generating intelligent follow-up questions
- **LR**: 2e-5
- **Epochs**: 2-3

---

## Phase 4: Retrieval-Augmented Generation (Week 10-11)

### 4.1 — Build Knowledge Base
- Store CS concepts, questions, solutions
- Use existing QuestionRanker bi-encoder for retrieval

### 4.2 — Context Injection
- Retrieve top-5 relevant concepts for each interaction
- Inject into generator's system prompt

---

## Phase 5: System Integration (Week 11-12)

### 5.1 — Update Inference Service
- Load trained generator model
- Combine with existing ranker for retrieval
- Add concept graph for intelligent question selection

### 5.2 — Update Orchestrator
- Use generator for natural language responses
- Track candidate knowledge state
- Adaptive difficulty based on performance

### 5.3 — Update Frontend
- Display natural language feedback
- Show concept mastery visualization

---

## Phase 6: Evaluation & Iteration (Week 12-14)

### Metrics

| Metric | Target |
|--------|--------|
| Perplexity | < 25 |
| Answer evaluation accuracy (vs Mohler human scores) | Pearson r > 0.6 |
| Follow-up relevance (human eval) | > 3.5/5 |
| Conversation coherence (human eval) | > 3.5/5 |
| Interview naturalness (human eval) | > 3.5/5 |

### Ablation Studies (Research Component)
1. Data ablation (10%, 25%, 50%, 100%)
2. Stage ablation (skip each stage)
3. Size ablation (8M, 20M, 45M)
4. RAG ablation (with vs without retrieval)
5. Tokenizer ablation (8K, 16K, 32K vocab)

---

## Training Time Estimates

| Stage | GTX 1650 (4GB) | Colab T4 (16GB) |
|-------|----------------|-----------------|
| Tokenizer | 10 min | 10 min |
| Stage 1: Pretraining | 3-5 days | 8-12 hours |
| Stage 2: Domain | 1-2 days | 3-5 hours |
| Stage 3: Instruction | 1-2 days | 3-5 hours |
| Stage 4: Interview | 6-12 hours | 1-2 hours |
| Stage 5: Evaluation | 12-24 hours | 2-4 hours |
| Stage 6: Follow-up | 6-12 hours | 1-2 hours |
| **Total** | **~8-15 days** | **~18-30 hours** |

---

## Success Criteria

The model is ready when it can:

1. **Conduct a 10-turn interview** without going off-topic
2. **Evaluate answers** with Pearson r > 0.6 against human grades
3. **Generate relevant follow-ups** addressing specific gaps
4. **Adapt difficulty** based on candidate performance
5. **Feel natural** — rated > 3.5/5 by human evaluators
6. **Handle CS topics**: arrays, linked lists, trees, graphs, hash tables, sorting, searching, DP, OOP, DBMS

---

## Research Question (UROP Report)

> "How much technical interview capability can be achieved with a 20M-parameter language model through domain specialization, multi-stage training, and retrieval augmentation?"
