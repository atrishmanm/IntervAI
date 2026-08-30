<div align="center">

# AI Interview Prep (V3.1)

A rigorous, highly analytical technical interview prep platform trained entirely from scratch on a dataset of 100,000 real coding interviews. 

Unlike ChatGPT, which provides conversational feedback, this system provides a **strict, 5-factor data-driven Report Card** evaluating your algorithmic approach, time/space complexity, edge cases, and communication depth.

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)

</div>

## 🌟 Why this over ChatGPT?

When you practice interviews with general LLMs, they often give you the answer, are too lenient, or fail to enforce strict interview rubrics. This platform is different:

1. **No External APIs:** The logic and evaluation are completely local, built from scratch without calling OpenAI or Anthropic.
2. **Direct Expert Mapping:** Every question is mapped directly to a real, expert transcript from the `stindardlogic/coding-interview-sft-100k` dataset.
3. **The 5-Factor Report Card:** Your answers are rigorously scored out of 100 on:
   - **Algorithmic Approach:** Did you identify the right pattern?
   - **Time Complexity:** Did you state the correct Big-O bound?
   - **Space Complexity:** Did you state the correct memory footprint?
   - **Edge Cases:** Did you identify constraints (empty arrays, negatives, etc.)?
   - **Communication Depth:** Was your reasoning detailed and substantive?

## 🚀 Quick Start (1-Click Run)

For Windows users, getting started takes exactly one click.

1. Open PowerShell in the project directory.
2. Run the startup script:
   ```powershell
   ./run.ps1
   ```

**What this script does automatically:**
- Creates a Python virtual environment (`venv`).
- Installs all requirements.
- Downloads the HuggingFace dataset (if missing).
- Cleans and parses the dataset into 30,000+ targeted interview questions.
- Builds the SQLite Question Bank.
- Starts the FastAPI backend.
- Opens your browser to the sleek, ChatGPT-style interview interface.

*(Note: The very first time you run this, it may take 2-3 minutes to download and process the dataset. Subsequent runs will be instant!)*

## 🧠 Question Types

The system extracts and tests you on three specific question formats:

1. **Theoretical Analysis:** "What is the optimal time/space complexity for solving [Problem]?"
2. **Code Analysis (Output Prediction):** "What does the following snippet output?" (Tests dry-running code in your head).
3. **Concept Explanation:** "Explain the Sliding Window approach for solving [Problem]. Why does it work?"

## 📁 Architecture & File Structure

```text
aiInterview/
├── run.ps1                      # 1-Click Startup Script
├── train_all.py                 # Data Pipeline Orchestrator
├── requirements.txt
├── data/
│   ├── raw/                     # Downloaded HuggingFace data
│   ├── processed/               # Cleaned JSONL files
│   └── question_bank.db         # SQLite database mapping questions -> expert answers
├── data_pipeline/
│   ├── download_datasets.py     # Fetches the 100k interview dataset
│   ├── clean_data.py            # NLP extraction of questions/complexities
│   └── build_question_bank.py   # Ingests clean data into SQLite
├── analysis/
│   └── scorer.py                # 5-Factor Analytical evaluation engine
├── orchestrator/
│   └── state_machine.py         # Manages the flow of the interview session
├── backend/
│   ├── main.py                  # FastAPI server and endpoints
│   └── inference_service.py     # Connects orchestrator to the scorer
└── frontend/
    └── index.html               # Clean, grayscale, ChatGPT-style UI
```

## 🎙️ Voice Support

The UI includes Web Speech API integration. Click the microphone icon to answer questions verbally, just like a real interview!

## 📜 License

MIT License. Feel free to use this for your college projects or personal interview prep!
