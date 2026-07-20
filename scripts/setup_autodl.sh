#!/bin/bash
# ============================================================
# AutoDL 实例初始化脚本
# 用途：在 AutoDL 租用的 GPU 实例上，一键配置 RL_SuperMario 环境
# 用法：bash setup_autodl.sh
# ============================================================
set -e

echo "============================================"
echo "  RL_SuperMario AutoDL 环境配置"
echo "============================================"

# ---------- 1. 系统依赖 ----------
echo "[1/5] 安装系统依赖..."
sudo apt-get update -qq
sudo apt-get install -y -qq libsdl2-dev xvfb > /dev/null 2>&1

# ---------- 2. 创建虚拟环境 ----------
echo "[2/5] 创建 Python 虚拟环境..."
if [ ! -d "venv" ]; then
    python -m venv venv
fi
source venv/bin/activate

# ---------- 3. 升级 pip ----------
echo "[3/5] 升级 pip..."
pip install --upgrade pip setuptools wheel -q

# ---------- 4. 安装项目依赖 ----------
echo "[4/5] 安装 Python 依赖（可能需要几分钟）..."
pip install -r requirements.txt -q

# ---------- 5. 验证环境 ----------
echo "[5/5] 验证环境..."
python -c "
import gym_super_mario_bros
import torch
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

print(f'  CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU 型号: {torch.cuda.get_device_name(0)}')
    print(f'  显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')

# 验证环境能正常创建
env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env.reset()
env.close()
print('  gym-super-mario-bros: OK')
"

echo ""
echo "============================================"
echo "  环境配置完成！"
echo "  接下来执行: bash scripts/run_train.sh"
echo "============================================"
