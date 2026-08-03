# AI Interview Prep - Startup Script
# Automatically handles setup, training pipeline, and server start.

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  Starting AI Interview Prep" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

# 1. Check Python installation
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python is not installed or not in PATH." -ForegroundColor Red
    exit 1
}

# 2. Setup Virtual Environment
$VenvDir = Join-Path $ProjectDir "venv"
if (!(Test-Path $VenvDir)) {
    Write-Host "`n[1/4] Creating Python Virtual Environment..." -ForegroundColor Yellow
    python -m venv $VenvDir
}

# 3. Activate Virtual Environment & Install Requirements
Write-Host "`n[2/4] Activating venv and checking dependencies..." -ForegroundColor Yellow
$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
if (Test-Path $ActivateScript) {
    . $ActivateScript
} else {
    Write-Host "ERROR: Could not find venv activation script." -ForegroundColor Red
    exit 1
}

pip install -r requirements.txt --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install requirements." -ForegroundColor Red
    exit 1
}

# 4. Check Database and Run Training Pipeline if needed
$DbPath = Join-Path $ProjectDir "data\question_bank.db"
if (!(Test-Path $DbPath)) {
    Write-Host "`n[3/4] First time setup: Running data pipeline (This may take a few minutes)..." -ForegroundColor Yellow
    $env:PYTHONIOENCODING="utf-8"
    python train_all.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Data pipeline failed." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`n[3/4] Database already exists. Skipping data pipeline." -ForegroundColor Green
}

# 5. Start Backend Server
Write-Host "`n[4/4] Starting FastAPI Server..." -ForegroundColor Yellow
Write-Host "Opening http://localhost:8000 in your browser..." -ForegroundColor Cyan

# Wait 2 seconds, then open browser
Start-Job -ScriptBlock {
    Start-Sleep -Seconds 3
    Start-Process "http://localhost:8000"
} | Out-Null

$env:PYTHONIOENCODING="utf-8"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

