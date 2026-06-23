@echo off
setlocal

set PYTHON=C:\Users\Netrin\Desktop\Python\Facial_Recog\faceenv\Scripts\python.exe

echo ============================================================
echo  FACE RECOGNITION TRAINING + VALIDATION PIPELINE
echo  Python: %PYTHON%
echo ============================================================
echo.

cd /d "%~dp0"

if not exist "%PYTHON%" (
    echo [ERROR] faceenv not found at %PYTHON%
    pause & exit /b 1
)

echo [STEP 0] Extracting datasets and aligning faces ...
echo ============================================================
"%PYTHON%" 00_prepare_data.py
if errorlevel 1 ( echo [ERROR] Step 0 failed. & pause & exit /b 1 )

echo.
echo [STEP 1] Fine-tuning metric adapter ...
echo ============================================================
"%PYTHON%" 01_finetune.py
if errorlevel 1 ( echo [ERROR] Step 1 failed. & pause & exit /b 1 )

echo.
echo [STEP 2] Building reference gallery ...
echo ============================================================
"%PYTHON%" 02_build_gallery.py
if errorlevel 1 ( echo [ERROR] Step 2 failed. & pause & exit /b 1 )

echo.
echo [STEP 3] Running validation ...
echo ============================================================
"%PYTHON%" 03_validate.py
if errorlevel 1 ( echo [ERROR] Step 3 failed. & pause & exit /b 1 )

echo.
echo ============================================================
echo  PIPELINE COMPLETE
echo  Results in: train_val\reports\
echo ============================================================
pause
