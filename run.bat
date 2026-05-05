@echo off
echo ================================
echo KaStack RAG System Setup
echo ================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Please install Python 3.8+
    exit /b 1
)

echo. Python found

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

echo. Virtual environment activated

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt -q

echo. Dependencies installed

REM Run the system
echo.
echo ================================
echo Starting RAG System Backend
echo ================================
echo.
echo API will be available at: http://localhost:5000
echo Frontend will be available at: http://localhost:5000
echo.
echo Press Ctrl+C to stop
echo.

cd backend
python app.py

pause
