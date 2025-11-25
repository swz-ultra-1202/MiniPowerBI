@echo off
REM ============================================================
REM  Make Any Windows System Compatible for  Streamlit Project
REM  Includes: Virtual Environment, Dependencies, Run App
REM  Author: M.Shah Nawaz
REM ============================================================

echo.
echo ===========================================
echo  Setting up Environment For Power BI
echo ===========================================
echo.

REM 1) Create virtual environment
python -m venv .venv
if %errorlevel% neq 0 (
    echo ERROR: Python venv could not be created.
    echo Make sure Python is installed and added to PATH.
    pause
    exit /b
)

REM 2) Activate venv
call .venv\Scripts\activate

echo.
echo Virtual environment activated successfully.
echo.

REM 3) Upgrade pip
python -m pip install --upgrade pip

REM 4) Install required dependencies
echo Installing dependencies...
pip install streamlit pandas numpy matplotlib scikit-learn python-docx

if %errorlevel% neq 0 (
    echo ERROR: One or more dependencies failed to install.
    pause
    exit /b
)

echo.
echo Dependencies installed successfully!
echo.

REM 5) Run Streamlit App
echo Starting Streamlit Application...
streamlit run "%~dp0ui.py"

echo.
echo Application stopped.
pause
