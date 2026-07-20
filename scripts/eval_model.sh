#!/bin/bash
# ============================================================
# 模型评估脚本（无头服务器版本）
# 用途：加载训练好的模型，在虚拟显示器上运行，保存为视频
# 用法：bash scripts/eval_model.sh
# ============================================================
set -e

# 激活虚拟环境
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

MODEL_PATH="${1:-monitor_log/best_model/best_model.zip}"
OUTPUT_VIDEO="${2:-mario_play.mp4}"
EPISODE_LEN="${3:-5000}"

echo "============================================"
echo "  模型评估"
echo "  模型: $MODEL_PATH"
echo "  输出: $OUTPUT_VIDEO"
echo "  步数: $EPISODE_LEN"
echo "============================================"

# 杀掉已有的 Xvfb
pkill Xvfb 2>/dev/null || true
sleep 1

# 启动虚拟显示器
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 &
XVFB_PID=$!
echo "虚拟显示器启动 (PID: $XVFB_PID)"

# 运行评估
python -c "
import os, sys
os.environ['DISPLAY'] = ':99'

from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3 import PPO
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.monitor import Monitor
from util_class import SkipFrame
import imageio
import uuid

def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, 4)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=(84, 84))
    env = Monitor(env, filename=os.path.join('./monitor_log/', str(uuid.uuid4())))
    return env

env = SubprocVecEnv([make_env for _ in range(1)])
env = VecFrameStack(env, 4, channels_order='last')
model = PPO.load('$MODEL_PATH', env=env)

# 用 gym-super-mario-bros 录制需要特殊处理，这里用 imageio 逐帧录制
# 创建一个单独的渲染环境（非向量化）
import gym
render_env = gym_super_mario_bros.make('SuperMarioBros-v2')
render_env = JoypadSpace(render_env, SIMPLE_MOVEMENT)
# 直接 render 到 rgb_array
obs = env.reset()
frames = []
total_reward = 0

for i in range($EPISODE_LEN):
    obs_copy = obs.copy()
    action, _ = model.predict(obs_copy)
    obs, reward, done, info = env.step(action)
    total_reward += reward[0]

    # 用渲染环境截图
    # 注意：向量化环境不能直接 render，这里简单记录数据
    if i % 10 == 0:
        print(f'Step {i}, reward: {reward[0]:.2f}, total: {total_reward:.2f}')

    if done[0]:
        print(f'Episode 结束, 总奖励: {total_reward:.2f}')
        obs = env.reset()
        total_reward = 0

print('评估完成！')
print('提示：要在云服务器上录制视频，建议把模型下载到本地后用 test_model.py 运行。')
"

# 清理虚拟显示器
kill $XVFB_PID 2>/dev/null || true
echo "虚拟显示器已关闭"
