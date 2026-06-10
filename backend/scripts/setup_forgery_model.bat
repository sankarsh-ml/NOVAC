@echo off
setlocal

cd /d "%~dp0\.."

if not exist "model_venvs" mkdir "model_venvs"

if not exist "model_venvs\forgery_venv\Scripts\python.exe" (
    python -m venv "model_venvs\forgery_venv"
)

"model_venvs\forgery_venv\Scripts\python.exe" -m pip install --upgrade pip

echo.
echo Forgery localization venv is ready at backend\model_venvs\forgery_venv
echo.
echo TruFor is not installed automatically because it requires model repo/checkpoint setup.
echo Place TruFor code and checkpoint under:
echo   backend\models\forgery\trufor
echo.
echo Expected checkpoint names:
echo   checkpoint.pth, ckpt.pth, or trufor.pth
echo.
echo Keep heavy TruFor dependencies inside this venv only.

endlocal
