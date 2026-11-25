@echo off
cd /d "%~dp0"

echo ============================================================
echo   RoadSight AI - Launching Full Local Stack
echo ============================================================
echo.
echo [1/2] Launching Flask Backend (http://127.0.0.1:5000)...
start "RoadSight AI - Backend (Flask)" cmd /k ""%~dp0start_backend.bat""

timeout /t 3 /nobreak >nul

echo [2/2] Launching Vite Frontend (http://localhost:5173)...
start "RoadSight AI - Frontend (Vite)" cmd /k ""%~dp0start_frontend.bat""

echo.
echo Both servers are starting in separate windows:
echo   - Frontend UI:    http://localhost:5173/
echo   - Backend Health: http://127.0.0.1:5000/api/health
echo.
echo Close the individual terminal windows to stop the servers.
pause
