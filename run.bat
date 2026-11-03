@echo off
cls
echo ===================================================
echo   AI MULTIMODAL DISEASE DETECTION DASHBOARD
echo ===================================================
echo.
cd /d "E:\SEMESTER 3\minor project\AI_Multimodal_Disease_Detection"
echo Starting application with virtual environment...
echo Dashboard URL: http://127.0.0.1:5000/
echo.
".\venv\Scripts\python.exe" app.py
pause
