@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0\.."

set VENV_DIR=model_venvs\forgery_venv
set PYTHON_EXE=%VENV_DIR%\Scripts\python.exe
set FORGERY_DIR=models\forgery
set TRUFOR_DIR=%FORGERY_DIR%\TruFor
set CHECKPOINT_DIR=%FORGERY_DIR%\checkpoints
set CHECKPOINT_FILE=%CHECKPOINT_DIR%\trufor.pth.tar
set WEIGHTS_ZIP=%FORGERY_DIR%\TruFor_weights.zip
set WEIGHTS_TMP=%FORGERY_DIR%\weights_tmp
set WEIGHTS_URL=https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip

if not exist "model_venvs" mkdir "model_venvs"
if not exist "%FORGERY_DIR%" mkdir "%FORGERY_DIR%"
if not exist "%CHECKPOINT_DIR%" mkdir "%CHECKPOINT_DIR%"

if not exist "%TRUFOR_DIR%\.git" (
    if exist "%TRUFOR_DIR%" (
        if not exist "%TRUFOR_DIR%\TruFor_train_test\test.py" (
            echo Existing backend\%TRUFOR_DIR% folder is not a complete TruFor checkout.
            echo Move or remove it, then run this script again.
            goto :error
        )
    ) else (
        echo Cloning official TruFor repository...
        git clone https://github.com/grip-unina/TruFor.git "%TRUFOR_DIR%"
        if errorlevel 1 (
            echo Could not clone TruFor automatically. Install git or manually clone:
            echo   https://github.com/grip-unina/TruFor
            echo into:
            echo   backend\%TRUFOR_DIR%
            goto :error
        )
    )
) else (
    echo TruFor repository already exists at backend\%TRUFOR_DIR%
)

if not exist "%PYTHON_EXE%" (
    echo Creating isolated forgery model venv at backend\%VENV_DIR%
    python -m venv "%VENV_DIR%"
    if errorlevel 1 goto :error
    echo Created isolated forgery model venv.
) else (
    echo Isolated forgery model venv already exists at backend\%VENV_DIR%
)

"%PYTHON_EXE%" -m pip install --upgrade pip wheel "setuptools<82"
if errorlevel 1 goto :error

echo Removing legacy OpenMMLab training packages from isolated forgery venv if present...
"%PYTHON_EXE%" -m pip uninstall -y openmim mmcls mmsegmentation mmcv-full opendatalab openxlab >nul 2>nul

echo Installing TruFor runtime dependencies into isolated forgery venv...
"%PYTHON_EXE%" -m pip install numpy opencv-python pillow scipy scikit-image matplotlib tqdm yacs timm==0.5.4 tensorboardX pyyaml requests
if errorlevel 1 goto :error

echo Installing PyTorch CPU wheels into isolated forgery venv...
"%PYTHON_EXE%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo PyTorch CPU wheel install failed. You may need to install the CUDA-specific torch build manually in %VENV_DIR%.
)

echo Skipping OpenMMLab training packages. The NOVAC runner uses TruFor inference only.
echo TruFor upstream training pins mmcv-full for Python 3.7/Torch 1.11; it is not required by test.py inference.

if exist "%TRUFOR_DIR%\TruFor_train_test\requirements.txt" (
    echo Installing TruFor repository requirements...
    "%PYTHON_EXE%" -m pip install -r "%TRUFOR_DIR%\TruFor_train_test\requirements.txt"
    if errorlevel 1 goto :error
)

if not exist "%CHECKPOINT_FILE%" (
    echo Downloading official TruFor weights...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$ErrorActionPreference='Stop';" ^
        "$zip='%WEIGHTS_ZIP%';" ^
        "$tmp='%WEIGHTS_TMP%';" ^
        "if (Test-Path $zip) { Remove-Item -LiteralPath $zip -Force };" ^
        "if (Test-Path $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force };" ^
        "Invoke-WebRequest -Uri '%WEIGHTS_URL%' -OutFile $zip -TimeoutSec 300;" ^
        "Expand-Archive -Path $zip -DestinationPath $tmp -Force;" ^
        "$found=Get-ChildItem -Path $tmp -Recurse -Filter 'trufor.pth.tar' | Select-Object -First 1;" ^
        "if ($null -eq $found) { throw 'trufor.pth.tar not found in downloaded weights zip' };" ^
        "Copy-Item -LiteralPath $found.FullName -Destination '%CHECKPOINT_FILE%' -Force;" ^
        "Remove-Item -LiteralPath $zip -Force;" ^
        "Remove-Item -LiteralPath $tmp -Recurse -Force;"

    if errorlevel 1 (
        echo Automatic checkpoint download failed.
        echo Manual checkpoint setup:
        echo   1. Download %WEIGHTS_URL%
        echo   2. Unzip it.
        echo   3. Copy trufor.pth.tar to:
        echo      backend\%CHECKPOINT_FILE%
        echo.
        echo The runner will return model_available=false until this file exists.
        goto :manual_checkpoint
    )
)

:manual_checkpoint
if not exist "%CHECKPOINT_FILE%" (
    echo.
    echo TruFor checkpoint is still missing.
    echo Expected file:
    echo   backend\%CHECKPOINT_FILE%
    echo.
    echo NOVAC can start, but TruFor inference will report model_available=false until this file exists.
) else (
    echo TruFor checkpoint ready:
    echo   backend\%CHECKPOINT_FILE%
)

echo.
echo Setup finished. Test the runner with:
echo   backend\model_venvs\forgery_venv\Scripts\python.exe backend\app\services\forgery_localization_runner.py --image backend\uploads\your_image.png
echo.
echo Main backend venv was not modified.
exit /b 0

:error
echo.
echo TruFor setup did not complete successfully.
echo Main backend venv was not modified.
exit /b 1
