@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
rem Usage: predict.bat <image or video> [model]   (drag and drop works too)
if "%~1"=="" (
    echo Drag an image or video onto predict.bat
    pause
    exit /b 1
)
set INPUT=%~f1
set NAME=%~n1_pred%~x1
cd /d "%~dp0"
set MODEL=%~2
if "%MODEL%"=="" set MODEL=runs\train\yolo_train\weights\best.pt
if not exist runs\predict mkdir runs\predict
.venv\Scripts\python scripts\inference.py --model "%MODEL%" --input "%INPUT%" --output "runs\predict\%NAME%" --no-display
pause
