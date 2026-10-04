@echo off
chcp 65001 > nul
cd /d "%~dp0"
rem Create .venv and install dependencies (CPU or CUDA torch)
if not exist .venv (
    py -3.11 -m venv .venv || python -m venv .venv || goto :error
)
.venv\Scripts\python -m pip install --upgrade pip || goto :error
where nvidia-smi > nul 2>&1
if %errorlevel%==0 (
    echo NVIDIA GPU detected: installing CUDA build of PyTorch
    .venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 || goto :error
) else (
    echo No NVIDIA GPU: installing CPU build of PyTorch
    .venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || goto :error
)
findstr /v /i "groundingdino" requirements.txt > "%TEMP%\yolo_req.txt"
.venv\Scripts\python -m pip install -r "%TEMP%\yolo_req.txt" || goto :error
echo.
echo Setup finished.
pause
exit /b 0
:error
echo.
echo Setup failed.
pause
exit /b 1
