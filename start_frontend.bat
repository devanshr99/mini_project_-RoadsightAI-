@echo off
cd /d "%~dp0frontend"

echo ============================================================
echo   RoadSight AI - Starting Local Vite Frontend (localhost:5173)
echo ============================================================
echo.

where node >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js was not found in PATH.
    echo Please install Node.js 18+ from https://nodejs.org/
    pause
    exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] npm was not found in PATH.
    pause
    exit /b 1
)

if not exist "node_modules" (
    echo [INFO] node_modules not found. Installing frontend dependencies...
    call npm install
    if errorlevel 1 (
        echo [ERROR] npm install failed.
        pause
        exit /b 1
    )
)

echo [OK] Frontend dependencies ready.
echo Starting Vite dev server at http://localhost:5173 ...
echo API proxy: /api -^> http://127.0.0.1:5000
echo Press Ctrl+C to stop.
echo.

call npm run dev
if errorlevel 1 (
    echo.
    echo [ERROR] Frontend dev server exited with an error.
    pause
)
