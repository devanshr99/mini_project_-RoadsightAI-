@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo   RoadSight AI - Starting Local Flask Backend (127.0.0.1:5000)
echo ============================================================
echo.

REM --- 1. Locate Python ---
set "PYTHON_EXE=python"
where python >nul 2>nul
if errorlevel 1 (
    if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" (
        set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
    ) else if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
        set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    ) else (
        echo [ERROR] Python was not found in PATH or standard install folders.
        echo Please install Python 3.10+ and run: pip install -r requirements.txt
        pause
        exit /b 1
    )
)
echo [OK] Using Python: %PYTHON_EXE%

REM --- 2. Check required Python packages ---
"%PYTHON_EXE%" -c "import torch, torchvision, flask, PIL, cv2, numpy" 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] One or more required Python packages are missing:
    echo         torch, torchvision, flask, pillow, opencv-python-headless, numpy
    echo.
    echo Fix by running:
    echo   "%PYTHON_EXE%" -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)
echo [OK] Required Python packages verified.

REM --- 3. Check model checkpoints ---
set "MISSING_FILES=0"
for %%F in (
    "checkpoints\custom_cnn_best.pth"
    "checkpoints\signnet_lite_best.pth"
    "checkpoints\mobilenet_v2_best.pth"
    "checkpoints\resnet18_best.pth"
) do (
    if not exist %%F (
        echo [ERROR] Missing model checkpoint: %%~F
        set "MISSING_FILES=1"
    )
)

REM --- 4. Check dataset metadata & plots ---
if not exist "data\class_names.json" (
    echo [ERROR] Missing dataset taxonomy file: data\class_names.json
    set "MISSING_FILES=1"
)
if not exist "data\dataset_summary.json" (
    echo [WARNING] Missing dataset summary: data\dataset_summary.json
)

if "!MISSING_FILES!"=="1" (
    echo.
    echo [ERROR] Critical model or dataset files are missing. Cannot start backend safely.
    pause
    exit /b 1
)
echo [OK] All 4 model checkpoints and dataset files verified.
echo.
echo Starting Flask server at http://127.0.0.1:5000 ...
echo Health endpoint: http://127.0.0.1:5000/api/health
echo Press Ctrl+C to stop.
echo.

"%PYTHON_EXE%" app.py
if errorlevel 1 (
    echo.
    echo [ERROR] Backend exited with an error. Check the output above.
    pause
)
