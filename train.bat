@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
rem Usage: train.bat [dataset folder] [extra options for scripts\main.py]
rem   train.bat
rem   train.bat datasets\my_dataset --epochs 100 --model yolo26s.pt
rem   train.bat datasets\my_dataset --weights runs\train\yolo_train\weights\best.pt
set DATA=%~1
if "%DATA%"=="" (
    set DATA=datasets\my_dataset
) else (
    shift
)
set ARGS=
:collect
if "%~1"=="" goto run
set ARGS=%ARGS% %1
shift
goto collect
:run
.venv\Scripts\python scripts\main.py --data-dir "%DATA%" --batch 8 --imgsz 640 %ARGS%
pause
