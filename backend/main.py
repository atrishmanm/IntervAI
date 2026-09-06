"""
backend/main.py — IntervAI Backend Platform
===========================================
Production-ready FastAPI backend for the IntervAI interview intelligence system.

Features:
- Full support for trained 125M parameter model evaluator & follow-up generator
- Multi-mode interview engine (General Technical, FAANG/Company-Specific, Salary Negotiation)
- Document resume parsing (PDF, DOCX, TXT, MD) with pure-Python fallbacks
- Unified API responses supporting modern and legacy frontend clients
- Persistent session tracking and improvement analytics
"""

import io
import re
import sys
import time
import uuid
import zlib
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator.interview_engine import InterviewEngine
from orchestrator.company_templates import list_companies, get_company_template
from orchestrator.salary_negotiation import SalaryNegotiationPractice, create_negotiation_practice
from orchestrator.session_tracker import create_session_tracker

app = FastAPI(title="IntervAI", version="4.0.0", description="AI Interview Intelligence Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_PATH = ROOT / "frontend" / "index.html"

# In-memory session registries
_interview_sessions: Dict[str, InterviewEngine] = {}
_session_start_times: Dict[str, float] = {}
_negotiation_sessions: Dict[str, SalaryNegotiationPractice] = {}


# ─────────────────────────────────────────────────────────────
# Text Extraction Utilities
# ─────────────────────────────────────────────────────────────

def extract_text_from_upload(filename: str, content: bytes) -> str:
    """Extract clean text from uploaded files (PDF, DOCX, TXT, MD)."""
    lower = filename.lower()

    if lower.endswith(".docx"):
        # Try python-docx
        try:
            import docx
            doc = docx.Document(io.BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            if text:
                return text
        except Exception:
            pass

        # Pure-Python zipfile fallback
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                xml_content = z.read("word/document.xml")
                tree = ET.fromstring(xml_content)
                texts = [node.text for node in tree.iter() if node.text]
                return " ".join(texts)
        except Exception as e:
            return f"[DOCX text extraction notice: {e}]"

    elif lower.endswith(".pdf"):
        # Try pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            pages = [p.extract_text() or "" for p in reader.pages]
            text = "\n".join(pages).strip()
            if text:
                return text
        except Exception:
            pass

        # Try PyPDF2
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            pages = [p.extract_text() or "" for p in reader.pages]
            text = "\n".join(pages).strip()
            if text:
                return text
        except Exception:
            pass

        # Pure-Python PDF stream fallback
        try:
            streams = re.findall(rb'stream[\r\n]+(.*?)[\r\n]+endstream', content, re.DOTALL)
            text_parts = []
            for s in streams:
                try:
                    decomp = zlib.decompress(s)
                    words = re.findall(r'\((.*?)\)', decomp.decode('latin-1', errors='ignore'))
                    if words:
                        text_parts.append(" ".join(words))
                except Exception:
                    continue
            if text_parts:
                return "\n".join(text_parts)
        except Exception:
            pass

        return content.decode("latin-1", errors="ignore")

    # Plain text / markdown
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue

    return content.decode("utf-8", errors="ignore")


# ─────────────────────────────────────────────────────────────
# Pydantic Request Models
# ─────────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    resume_text: str = ""
    candidate_name: str = ""
    company_name: str = ""

class AnswerRequest(BaseModel):
    session_id: str
    answer: str
    time_taken: float = 0.0

class ChatRequest(BaseModel):
    session_id: str
    answer: str
    time_taken: float = 0.0

class NegotiationStartRequest(BaseModel):
    scenario_id: Optional[str] = None

class NegotiationRespondRequest(BaseModel):
    session_id: str
    response: str
    tactic: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# Core & System Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the interactive web frontend."""
    if FRONTEND_PATH.exists():
        return FileResponse(str(FRONTEND_PATH))
    return {"message": "IntervAI API running. Frontend not found at frontend/index.html"}


@app.get("/api/status")
async def get_status():
    """Check system status, model availability, and active session count."""
    saved_models_dir = ROOT / "models" / "generator" / "saved"
    available_ckpts = [p.stem for p in saved_models_dir.glob("*.pt")] if saved_models_dir.exists() else []

    return {
        "status": "ready",
        "ready": True,
        "version": "4.0.0",
        "active_sessions": len(_interview_sessions),
        "available_checkpoints": available_ckpts,
        "features": [
            "adaptive_interview",
            "company_templates",
            "salary_negotiation",
            "session_tracker",
            "multi_format_resume_upload",
            "voice_audio",
            "star_rubric_scoring"
        ]
    }


# ─────────────────────────────────────────────────────────────
# Company Templates Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/api/companies")
async def get_companies():
    """List available company templates (Google, Meta, Amazon, Apple, Netflix, etc.)."""
    return {"companies": list_companies()}


@app.get("/api/companies/{company_name}")
async def get_company_details(company_name: str):
    """Get full structure and evaluation criteria for a target company."""
    template = get_company_template(company_name)
    if not template:
        raise HTTPException(status_code=404, detail=f"Company template '{company_name}' not found.")

    return {
        "company_name": template.company_name,
        "tier": template.tier.value,
        "description": template.description,
        "bar_level": template.bar_level,
        "question_style": template.question_style,
        "culture_keywords": template.culture_keywords,
        "what_they_value": template.what_they_value,
        "evaluation_criteria": template.evaluation_criteria,
        "common_questions": template.common_questions,
        "rounds": [
            {
                "name": r.name,
                "duration_minutes": r.duration_minutes,
                "focus_areas": r.focus_areas,
                "question_types": r.question_types,
                "max_questions": r.max_questions,
            }
            for r in template.rounds
        ]
    }


# ─────────────────────────────────────────────────────────────
# Interview Session Endpoints (Modern + Legacy Compatible)
# ─────────────────────────────────────────────────────────────

@app.post("/api/interview/start")
@app.post("/api/start")
async def start_interview_session(req: StartRequest):
    """Start an interview session with resume text and optional company target."""
    session_id = str(uuid.uuid4())
    engine = InterviewEngine()

    try:
        result = engine.start_interview(
            resume_text=req.resume_text,
            candidate_name=req.candidate_name,
            company_name=req.company_name,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to start interview: {e}")

    _interview_sessions[session_id] = engine
    _session_start_times[session_id] = time.time()

    first_question_text = result["question"]

    return {
        "session_id": session_id,
        "type": "interview_started",
        "message": result["message"],
        "question": first_question_text,
        "phase": result["phase"],
        "time_limit": result["time_limit"],
        "question_number": result["question_number"],
        "total_questions": result["total_questions"],
        "company": result.get("company"),
        "first_question": {
            "question_text": first_question_text,
            "question_type": result.get("phase", "warmup"),
            "topic": result.get("company") or "General Technical",
        }
    }


@app.post("/api/interview/start-file")
async def start_interview_with_file(
    file: UploadFile = File(...),
    candidate_name: str = Form(""),
    company_name: str = Form(""),
):
    """Start interview by uploading a resume document (PDF, DOCX, TXT, MD)."""
    content = await file.read()
    extracted_text = extract_text_from_upload(file.filename or "resume.txt", content)

    if not extracted_text or len(extracted_text.strip()) < 15:
        raise HTTPException(
            status_code=400,
            detail="Could not extract sufficient text from the uploaded file. Please paste your resume text."
        )

    return await start_interview_session(StartRequest(
        resume_text=extracted_text,
        candidate_name=candidate_name,
        company_name=company_name,
    ))


@app.post("/api/interview/answer")
@app.post("/api/chat")
async def submit_candidate_answer(req: AnswerRequest):
    """Submit an answer and get comprehensive score, feedback, and next question."""
    if req.session_id not in _interview_sessions:
        raise HTTPException(status_code=404, detail="Interview session not found. Please start a new session.")

    engine = _interview_sessions[req.session_id]

    try:
        result = engine.submit_answer(req.answer, req.time_taken)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing answer: {e}")

    scores_dict = result.get("score", {})
    avg_score = int(sum(scores_dict.values()) / max(len(scores_dict), 1) * 100) if scores_dict else 75

    concepts = [
        {
            "name": dim.replace("_", " ").title(),
            "status": "pass" if val >= 0.70 else "partial" if val >= 0.45 else "fail",
            "detail": f"Proficiency: {int(val * 100)}%"
        }
        for dim, val in scores_dict.items()
    ]

    if result.get("is_complete"):
        final_report = result.get("report", {})
        # Cleanup active session
        _interview_sessions.pop(req.session_id, None)
        _session_start_times.pop(req.session_id, None)

        return {
            "type": "session_complete",
            "is_complete": True,
            "score": scores_dict,
            "feedback": result.get("feedback", ""),
            "report": final_report,
            "summary": {
                "total_questions": final_report.get("total_questions", 0),
                "average_score": final_report.get("overall_score", avg_score),
                "verdict": final_report.get("recommendation", "COMPLETED"),
            },
            "panel_report": {
                "overall": {
                    "overall_score": final_report.get("overall_score", avg_score),
                    "verdict": final_report.get("verdict", final_report.get("recommendation", "COMPLETED")),
                },
                "concepts": [
                    {
                        "concept": k,
                        "level": "Mastered" if v >= 0.75 else "Developing" if v >= 0.5 else "Beginner",
                        "detail": f"Score: {int(v * 100)}%"
                    }
                    for k, v in final_report.get("competencies", {}).items()
                ],
                "weaknesses": final_report.get("areas_for_improvement", []),
                "suggestions": [
                    "Back technical decisions with trade-off analysis.",
                    "Explicitly quantify business and performance impact.",
                    "Structure behavioral responses with Situation, Task, Action, and Result."
                ],
            }
        }

    # In-progress question response
    next_question_text = result.get("question", "")
    phase_str = result.get("phase", "technical")

    return {
        "type": "analysis_and_next",
        "is_complete": False,
        "score": scores_dict,
        "feedback": result.get("feedback", ""),
        "star_analysis": result.get("star_analysis", {}),
        "question": next_question_text,
        "phase": phase_str,
        "difficulty": result.get("difficulty", "MEDIUM"),
        "time_limit": result.get("time_limit", 120),
        "question_number": result.get("question_number", 1),
        "total_questions": result.get("total_questions", 6),
        "questions_asked": max(1, result.get("question_number", 2) - 1),
        "report": {
            "score": avg_score,
            "overall_score": avg_score,
            "feedback": result.get("feedback", ""),
            "concepts": concepts,
            "suggested_followup": "Deepen your technical trade-offs in the next prompt.",
        },
        "next_question": {
            "question_text": next_question_text,
            "question_type": phase_str,
            "topic": phase_str.title(),
        }
    }


@app.get("/api/interview/report/{session_id}")
async def get_interview_report(session_id: str):
    """Retrieve full final report for an active or completed session."""
    if session_id not in _interview_sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    engine = _interview_sessions[session_id]
    return engine._generate_final_report()


@app.get("/api/interview/report/{session_id}/export")
async def export_interview_report(session_id: str, format: str = Query("markdown", regex="^(markdown|json)$")):
    """Export interview report in Markdown or JSON format."""
    if session_id not in _interview_sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    engine = _interview_sessions[session_id]
    report = engine._generate_final_report()

    if format == "json":
        return JSONResponse(content=report)

    # Markdown export
    md_lines = [
        f"# IntervAI Performance Report — {report.get('candidate', 'Candidate')}",
        f"\n**Overall Score**: {report.get('overall_score', 0)} / 100  |  **Recommendation**: {report.get('recommendation', 'N/A')}",
        f"\n**Company / Track**: {report.get('company') or 'General Technical'}  |  **Difficulty**: {report.get('difficulty_reached', 'MEDIUM')}",
        "\n## Core Competencies",
    ]
    for comp, val in report.get("competencies", {}).items():
        bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
        md_lines.append(f"- **{comp}**: `{bar}` {int(val * 100)}%")

    if report.get("star_analysis"):
        md_lines.append("\n## Behavioral STAR Analysis")
        for k, v in report["star_analysis"].items():
            md_lines.append(f"- **{k.capitalize()}**: {int(v * 100)}%")

    md_lines.append("\n## Detailed Question Breakdown")
    for i, a in enumerate(report.get("detailed_answers", []), 1):
        md_lines.append(f"\n### Question {i}: {a['question']}")
        md_lines.append(f"- **Phase**: {a['phase']}  |  **Time**: {a['time_sec']}s")
        md_lines.append(f"- **Feedback**: {a['feedback']}")

    return PlainTextResponse("\n".join(md_lines), media_type="text/markdown")


@app.get("/api/interview/sessions")
async def list_active_sessions():
    """List all currently active interview sessions."""
    return {
        "sessions": [
            {
                "id": sid,
                "candidate": engine.session.candidate_name,
                "phase": engine.session.phase.value,
                "progress": f"{engine.session.current_question_idx}/{len(engine.session.questions)}",
                "company": engine.session.company_name or "General",
                "elapsed_min": round((time.time() - _session_start_times.get(sid, time.time())) / 60, 1),
            }
            for sid, engine in _interview_sessions.items()
        ]
    }


# ─────────────────────────────────────────────────────────────
# Salary Negotiation Practice Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/api/negotiation/scenarios")
async def get_negotiation_scenarios():
    """List realistic compensation negotiation scenarios."""
    practice = create_negotiation_practice()
    return {"scenarios": practice.list_scenarios()}


@app.post("/api/negotiation/start")
async def start_salary_negotiation(req: NegotiationStartRequest):
    """Start an interactive salary negotiation practice session."""
    session_id = str(uuid.uuid4())
    practice = create_negotiation_practice()
    init_data = practice.start_negotiation(req.scenario_id)

    _negotiation_sessions[session_id] = practice

    scenario = practice.current_scenario
    return {
        "session_id": session_id,
        "scenario_id": scenario.scenario_id,
        "company_type": scenario.company_type,
        "role_level": scenario.role_level,
        "expected_range": scenario.expected_range,
        "initial_offer": {
            "base_salary": scenario.initial_offer.base_salary,
            "signing_bonus": scenario.initial_offer.signing_bonus,
            "equity_annual": scenario.initial_offer.equity_annual,
            "total_first_year": scenario.initial_offer.total_first_year,
            "total_4yr": scenario.initial_offer.total_4yr,
        },
        "market_data": {
            "base_salary": scenario.market_data.base_salary,
            "total_first_year": scenario.market_data.total_first_year,
            "total_4yr": scenario.market_data.total_4yr,
            "signing_bonus": scenario.market_data.signing_bonus,
            "equity_annual": scenario.market_data.equity_annual,
        },
        "phase": practice.current_phase.value,
        "round": practice.round_number,
        "instructions": init_data["instructions"],
        "opening_message": (
            f"Hello! Congratulations on your offer for {scenario.role_level} at our {scenario.company_type} team! "
            f"We are offering a base salary of ${scenario.initial_offer.base_salary:,.0f} with a "
            f"${scenario.initial_offer.signing_bonus:,.0f} signing bonus and ${scenario.initial_offer.equity_annual:,.0f}/year in equity. "
            f"How does this sound to you?"
        )
    }


@app.post("/api/negotiation/respond")
async def respond_salary_negotiation(req: NegotiationRespondRequest):
    """Submit candidate counter-offer/response and receive realistic recruiter reaction."""
    if req.session_id not in _negotiation_sessions:
        raise HTTPException(status_code=404, detail="Negotiation session not found.")

    practice = _negotiation_sessions[req.session_id]

    try:
        result = practice.submit_response(req.response, tactic=req.tactic or "collaborative")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Negotiation evaluation error: {e}")

    fb = result.get("feedback", {})
    emp = result.get("employer_response", {})
    is_closing = result.get("rounds_remaining", 1) <= 0 or practice.round_number >= 4

    summary = practice.get_negotiation_summary() if is_closing else None

    return {
        "round": result.get("round", practice.round_number),
        "phase": result.get("phase", practice.current_phase.value),
        "rounds_remaining": result.get("rounds_remaining", 0),
        "is_complete": is_closing,
        "scores": {
            "assertiveness": round(fb.get("assertiveness", 0.5) * 100, 1),
            "professionalism": round(fb.get("professionalism", 0.8) * 100, 1),
            "data_driven": round(fb.get("data_driven", 0.5) * 100, 1),
            "collaboration": round(fb.get("collaboration", 0.6) * 100, 1),
            "overall": round(fb.get("overall", 0.6) * 100, 1),
        },
        "strengths": fb.get("strengths", []),
        "improvements": fb.get("improvements", []),
        "suggested_response": fb.get("suggested_response", ""),
        "employer_message": emp.get("message", "Thank you for the counter-proposal."),
        "employer_response_type": emp.get("response_type", "discussion"),
        "new_offer": emp.get("new_offer", {}),
        "summary": summary,
    }


# ─────────────────────────────────────────────────────────────
# Performance Analytics Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/api/analytics/history")
async def get_performance_history():
    """Get multi-session tracking analytics, improvement trends, and skill progress."""
    tracker = create_session_tracker()
    history = tracker.get_session_history()
    trends = tracker.get_improvement_trends()
    skill_progress = tracker.get_skill_progress()
    weaknesses = tracker.get_weakness_analysis()

    return {
        "total_sessions": len(history),
        "history": history,
        "trends": trends,
        "skill_progress": skill_progress,
        "weakness_analysis": weaknesses,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
