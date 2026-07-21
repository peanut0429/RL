@echo off
chcp 65001 > nul 2>&1
setlocal enabledelayedexpansion

echo ============================================
echo   RL_SuperMario Local Setup (Windows/CPU)
echo ============================================

rem --- 1. Check Python ---
echo [1/4] Checking Python...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10 first.
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version

rem --- 2. Create venv ---
echo [2/4] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo venv created
) else (
    echo venv already exists, skip
)

call venv\Scripts\activate.bat

rem --- 3. Install dependencies ---
echo [3/4] Installing dependencies (this may take a few minutes)...

pip install "pip<24.1" -q
pip install "setuptools==65.5.0" "wheel<0.40.0" "packaging<22.0" -q
pip install gym==0.21.0 --no-build-isolation -q
pip install -r requirements.txt -q

rem --- 4. Verify ---
echo [4/4] Verifying environment...
python -c "import gym_super_mario_bros; print('  gym-super-mario-bros: OK')"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "from stable_baselines3 import PPO; print('  stable_baselines3: OK')"

echo.
echo ============================================
echo   Setup complete!
echo ============================================
echo.
echo   How to test the model:
echo     1. Download best_model*.zip from AutoDL
echo     2. Rename it to best_model.zip
echo     3. Put it in monitor_log\best_model\
echo     4. Run: venv\Scripts\activate
echo        Then: python test_model.py
echo.

pause
