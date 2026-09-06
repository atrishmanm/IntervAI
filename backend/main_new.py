"""
backend/main.py
===============
FastAPI backend with all interview features.

Features:
1. Resume upload and parsing
2. Adaptive interview sessions
3. Real-time analytics
4. Report generation
5. Session management
6. Multiple export formats
"""

import json
import uuid
import time
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator.resume_parser import ResumeParser
from orchestrator.adaptive_interview import AdaptiveInterviewEngine
from orchestrator.interview_analytics import InterviewAnalytics
from orchestrator.mock_interview import MockInterviewSimulator, InterviewConfig


app = FastAPI(
    title="INTERVUE API",
    description="AI-Powered Interview Simulator",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Data models
class ResumeUpload(BaseModel):
    resume_text: str


class AnswerSubmit(BaseModel):
    session_id: str
    response: str
    time_taken: float = 0


class InterviewConfigModel(BaseModel):
    max_questions: int = 15
    time_limit_minutes: int = 45
    phases: List[str] = ["warmup", "technical", "behavioral", "coding", "culture_fit"]
    difficulty_start: str = "medium"
    enable_follow_ups: bool = True
    enable_analytics: bool = True
    export_format: str = "json"


# In-memory session storage
sessions: Dict[str, MockInterviewSimulator] = {}


@app.get("/")
async def root():
    return {
        "name": "INTERVUE API",
        "version": "1.0.0",
        "description": "AI-Powered Interview Simulator",
        "endpoints": {
            "POST /api/interview/start": "Start interview with resume text",
            "POST /api/interview/start-file": "Start interview with uploaded resume",
            "POST /api/interview/answer": "Submit answer",
            "GET /api/interview/analytics/{session_id}": "Get real-time analytics",
            "GET /api/interview/report/{session_id}": "Get final report",
            "GET /api/interview/report/{session_id}/export": "Export report (format: json/markdown/html)",
            "GET /api/interview/sessions": "List all sessions",
        }
    }


@app.post("/api/interview/start")
async def start_interview(resume: ResumeUpload, config: InterviewConfigModel = None):
    """Start a new interview with resume text."""
    try:
        # Create session
        session_id = str(uuid.uuid4())
        
        # Create simulator with config
        interview_config = config or InterviewConfig()
        simulator = MockInterviewSimulator(config=interview_config)
        
        # Start interview
        result = simulator.start_interview(resume.resume_text)
        
        # Store session
        sessions[session_id] = simulator
        
        return {
            "session_id": session_id,
            "status": "started",
            "candidate": result["candidate"],
            "structure": result["structure"],
            "first_question": result["first_question"],
            "instructions": result["instructions"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/interview/start-file")
async def start_interview_file(file: UploadFile = File(...), config: InterviewConfigModel = None):
    """Start interview with uploaded resume file."""
    try:
        # Read file content
        content = await file.read()
        resume_text = content.decode("utf-8")
        
        # Create session
        session_id = str(uuid.uuid4())
        
        # Create simulator with config
        interview_config = config or InterviewConfig()
        simulator = MockInterviewSimulator(config=interview_config)
        
        # Start interview
        result = simulator.start_interview(resume_text)
        
        # Store session
        sessions[session_id] = simulator
        
        return {
            "session_id": session_id,
            "status": "started",
            "candidate": result["candidate"],
            "structure": result["structure"],
            "first_question": result["first_question"],
            "instructions": result["instructions"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/interview/answer")
async def submit_answer(answer: AnswerSubmit):
    """Submit an answer and get next question."""
    if answer.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    simulator = sessions[answer.session_id]
    
    try:
        result = simulator.submit_answer(answer.response, answer.time_taken)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/interview/analytics/{session_id}")
async def get_analytics(session_id: str):
    """Get real-time analytics for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    simulator = sessions[session_id]
    
    return {
        "session_id": session_id,
        "analytics": simulator.get_analytics(),
        "radar_chart": simulator.get_radar_chart(),
        "timeline": simulator.get_timeline(),
        "phase_breakdown": simulator.get_phase_breakdown(),
        "prediction": simulator.get_prediction(),
    }


@app.get("/api/interview/report/{session_id}")
async def get_report(session_id: str):
    """Get final interview report."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    simulator = sessions[session_id]
    
    if simulator.session_active:
        return {"status": "in_progress", "message": "Interview not yet complete"}
    
    report = simulator._generate_final_report()
    return report


@app.get("/api/interview/report/{session_id}/export")
async def export_report(session_id: str, format: str = "json"):
    """Export interview report in specified format."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    simulator = sessions[session_id]
    
    if simulator.session_active:
        raise HTTPException(status_code=400, detail="Interview not yet complete")
    
    try:
        report = simulator.export_report(format)
        return {"format": format, "report": report}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/interview/sessions")
async def list_sessions():
    """List all interview sessions."""
    session_list = []
    for session_id, simulator in sessions.items():
        session_list.append({
            "session_id": session_id,
            "candidate_name": simulator.current_profile.name if simulator.current_profile else "Unknown",
            "status": "active" if simulator.session_active else "completed",
            "questions_answered": len(simulator.engine.session.answers) if simulator.engine.session else 0,
            "start_time": simulator.start_time,
        })
    
    return {"sessions": session_list}


@app.delete("/api/interview/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    return {"status": "deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
