@echo off
echo ==============================================
echo   Starting NutriLens Frontend and Backend
echo ==============================================

echo [1/2] Starting Python ML Backend on port 8000...
start "NutriLens Backend (FastAPI)" cmd /k ".venv\Scripts\python.exe -m uvicorn ml_backend.main:app --reload --host 0.0.0.0 --port 8000 --env-file .env.local"

echo [2/2] Starting Vite Frontend on port 8443...
start "NutriLens Frontend (Vite)" cmd /k "npm run dev"

echo.
echo Both services launched in separate windows!
echo - Frontend: http://localhost:8443
echo - Backend API Docs: http://localhost:8000/docs
echo ==============================================
