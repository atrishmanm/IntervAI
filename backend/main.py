"""
backend/main.py
===============
FastAPI backend for AI Interview Prep V3.

Endpoints:
    POST /api/start    - Start a new interview session
    POST /api/chat     - Send a student answer, get structured analysis + next question
    GET  /api/status   - Check system readiness
    GET  /             - Serve the frontend HTML
"""

import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.inference_service import models_ready
from orchestrator.state_machine import InterviewState, orchestrator

app = FastAPI(title="AI Interview Prep V3", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_PATH = ROOT / "frontend" / "index.html"
_sessions: dict[str, InterviewState] = {}


class StartRequest(BaseModel):
    pass

class ChatRequest(BaseModel):
    session_id: str
    answer: str


@app.get("/", include_in_schema=False)
async def serve_frontend():
    if FRONTEND_PATH.exists():
        return FileResponse(str(FRONTEND_PATH), media_type="text/html")
    return JSONResponse({"message": "Frontend not found."})


@app.get("/api/status")
async def get_status():
    status = models_ready()
    return {
        **status,
        "ready": status["question_bank"] and status["tfidf_index"],
    }


@app.post("/api/start")
async def start_session(req: StartRequest = None):
    session_id = str(uuid.uuid4())
    state = InterviewState(session_id=session_id)
    result = orchestrator.start_session(state)
    _sessions[session_id] = state

    return {
        "session_id": session_id,
        **result,
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    state = _sessions.get(req.session_id)
    if state is None:
        raise HTTPException(404, "Session not found. Start a new session first.")

    if state.finished:
        raise HTTPException(400, "Session already complete.")

    if not req.answer.strip():
        raise HTTPException(400, "Answer cannot be empty.")

    result = orchestrator.process_answer(state, req.answer)

    return {
        "session_id": req.session_id,
        "turn_count": state.turn_count,
        "questions_asked": state.questions_asked,
        "finished": state.finished,
        **result,
    }


@app.get("/api/report")
async def get_report(session_id: str):
    """Return the panel-style final report for a finished session."""
    state = _sessions.get(session_id)
    if state is None:
        raise HTTPException(404, "Session not found. Start a new session first.")
    if not state.finished:
        return {
            "session_id": session_id,
            "finished": False,
            "message": "Interview still in progress.",
            "questions_answered": state.questions_asked,
        }

    # Regenerate report from the latest candidate state if not cached
    if not state.report:
        from orchestrator.candidate_state import CandidateStateManager
        cand = CandidateStateManager().get_or_create(session_id)
        from analysis.report import build_panel_report
        state.report = build_panel_report(cand)

    return {
        "session_id": session_id,
        "finished": True,
        "panel_report": state.report,
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting AI Interview Prep V3 on http://localhost:8000")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
