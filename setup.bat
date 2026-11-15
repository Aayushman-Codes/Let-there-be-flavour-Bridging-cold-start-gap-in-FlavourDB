@echo off
REM ======================================================================
REM  SavorLink - one-click setup, results and frontend (Windows)
REM
REM    setup.bat            full run: install, test, train + evaluate,
REM                         print the comparison, launch the frontend
REM    setup.bat results    only print the comparison of the 4 methods
REM    setup.bat app        only launch the frontend (uses existing results)
REM
REM  If any step fails, see setup.md for the manual steps.
REM ======================================================================
setlocal
cd /d "%~dp0"
set "MODE=%~1"

echo.
echo [1/6] Checking Python...
where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH. Install Python 3.10+ from https://www.python.org/downloads/
    goto :fail
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo Python 3.10 or newer is required.
    python --version
    goto :fail
)
if not exist ".venv\Scripts\activate.bat" (
    echo Creating virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 goto :fail
)
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :fail

echo.
echo [2/6] Installing dependencies (first run downloads PyTorch and Streamlit, a few minutes)...
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
if errorlevel 1 goto :fail

if /i "%MODE%"=="app" goto :app
if /i "%MODE%"=="results" goto :results
if not exist "results\logs" mkdir "results\logs"

echo.
echo [3/6] Running the test suite (full log: results\logs\tests.log)...
python tests\run_tests.py > "results\logs\tests.log" 2>&1
if errorlevel 1 (
    type "results\logs\tests.log"
    goto :fail
)
findstr /c:"Summary:" "results\logs\tests.log"

echo.
echo [4/6] Training and evaluating 4 modalities x 3 seeds (about a minute; log: results\logs\benchmark.log)...
python -m src.run_benchmark > "results\logs\benchmark.log" 2>&1
if errorlevel 1 (
    powershell -NoProfile -Command "Get-Content 'results\logs\benchmark.log' -Tail 40"
    goto :fail
)
echo       done: results\RESULTS.md, results\results.json, results\models\

:results
echo.
echo [5/6] Results
python -m src.compare
if errorlevel 1 goto :fail
if /i "%MODE%"=="results" goto :done

:app
echo.
echo [6/6] Launching the frontend at http://localhost:8501  (press Ctrl+C here to stop it)
if not exist "results\results.json" (
    echo No results yet. Run setup.bat without arguments first.
    goto :fail
)
if not defined SAVORLINK_NO_BROWSER start "" /b cmd /c "timeout /t 5 /nobreak >nul & explorer http://localhost:8501"
python -m streamlit run app\streamlit_app.py
goto :done

:done
endlocal
exit /b 0

:fail
echo.
echo Setup stopped because the step above failed. See setup.md for manual instructions.
endlocal
exit /b 1
