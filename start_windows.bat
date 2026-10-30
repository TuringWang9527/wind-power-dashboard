@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    set "WIND_PYTHON=.venv\Scripts\python.exe"
) else (
    set "WIND_PYTHON=python"
)
"%WIND_PYTHON%" -m streamlit run app.py %*
if errorlevel 1 (
    echo.
    echo Start failed. Install dependencies first:
    echo python -m pip install -r requirements.txt
    pause
)
endlocal
