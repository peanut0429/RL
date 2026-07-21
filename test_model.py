import os
import zipfile
import time
import torch
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import PPO
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.monitor import Monitor
from util_class import SkipFrame
import uuid


def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, 4)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=(84, 84))
    monitor_dir = r'./monitor_log/'
    env = Monitor(env, filename=os.path.join(monitor_dir, str(uuid.uuid4())))
    return env


def main():
    # 模型路径：支持命令行参数或默认路径
    model_path = r'monitor_log/best_model/best_model.zip'

    # 初始化环境（本地用 DummyVecEnv，支持 render）
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, 4, channels_order='last')

    # --- 加载模型（绕过跨平台序列化问题）---
    # 用和训练时完全相同的超参数重建模型结构
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=8192,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.1,
        clip_range_vf=None,
        ent_coef=0.1,
        vf_coef=0.5,
        max_grad_norm=0.8,
        target_kl=0.03,
        device='cpu',
        verbose=0,
    )

    # 从 zip 里读 policy 权重，直接覆写到模型上
    with zipfile.ZipFile(model_path, 'r') as zf:
        with zf.open('policy.pth') as f:
            state_dict = torch.load(f, map_location='cpu')
    model.policy.load_state_dict(state_dict, strict=True)
    print(f"模型加载成功: {model_path}")

    # --- 运行游戏 ---
    obs = env.reset()
    total_reward = 0
    ep_count = 0

    for i in range(10000):
        obs_copy = obs.copy()
        action, _ = model.predict(obs_copy)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        env.render('human')
        time.sleep(0.03)  # 控制速度，约 30fps，太快看不清可加大

        if done[0]:
            ep_count += 1
            print(f'Episode {ep_count} 结束, 总奖励: {total_reward:.1f}')
            total_reward = 0
            obs = env.reset()


if __name__ == '__main__':
    main()
