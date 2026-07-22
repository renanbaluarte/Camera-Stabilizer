@echo off
setlocal
cd /d %~dp0

if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -r requirements.txt

pyinstaller --onefile --windowed --name CameraStabilizer ^
    --add-data "config;config" ^
    --collect-all cv2 ^
    --collect-all pyvirtualcam ^
    app.py

echo.
echo Executavel em dist\CameraStabilizer.exe
endlocal
