@echo off
REM Start the Flask web app for cat/dog breed classification

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment in .\venv...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install/update dependencies
echo Installing/updating dependencies...
pip install -r requirements.txt

REM Run the app
echo Starting Flask app...
python main.py
