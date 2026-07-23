import os
import zipfile
import time
import torch
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import PPO
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.monitor import Monitor
from util_class import SkipFrame
import uuid


def make_env_for_stage(stage):
    """为指定关卡创建环境"""
    def _init():
        env = gym_super_mario_bros.make(f'SuperMarioBros-{stage}-v2')
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        env = SkipFrame(env, 4)
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, shape=(84, 84))
        monitor_dir = r'./monitor_log/'
        env = Monitor(env, filename=os.path.join(monitor_dir, str(uuid.uuid4())))
        return env
    return _init


def evaluate_stage(stage, model_path, episodes=5, max_steps=5000):
    """在指定关卡上评估模型，返回每局奖励"""
    print(f"\n{'='*50}")
    print(f"  测试关卡: 1-{stage}")
    print(f"{'='*50}")

    env = DummyVecEnv([make_env_for_stage(stage)])
    env = VecFrameStack(env, 4, channels_order='last')

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

    with zipfile.ZipFile(model_path, 'r') as zf:
        with zf.open('policy.pth') as f:
            state_dict = torch.load(f, map_location='cpu')
    model.policy.load_state_dict(state_dict, strict=True)

    rewards = []
    clears = []
    for ep in range(episodes):
        obs = env.reset()
        total_reward = 0
        ep_clear = False

        for step in range(max_steps):
            obs_copy = obs.copy()
            action, _ = model.predict(obs_copy)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            env.render('human')
            #time.sleep(0.02)

            # info 是 [info_dict]，取第一个环境的 info
            if info[0].get('flag_get', False):
                ep_clear = True

            if done[0]:
                break

        rewards.append(total_reward)
        clears.append(ep_clear)
        status = "🏁 通关" if ep_clear else "💀 未通关"
        print(f"  Episode {ep+1}: reward={total_reward:.0f}  {status}")

    avg_reward = sum(rewards) / len(rewards)
    clear_rate = sum(clears) / len(clears) * 100
    print(f"  平均奖励: {avg_reward:.0f}  通关率: {clear_rate:.0f}%")
    return avg_reward, clear_rate


def main():
    model_path = r'monitor_log/best_model/best_model_complex.zip'

    stages = ['1-1', '1-2', '1-3', '1-4']
    results = {}
    clear_rates = {}

    for stage in stages:
        avg, cr = evaluate_stage(stage, model_path, episodes=10, max_steps=5000)
        results[stage] = avg
        clear_rates[stage] = cr

    print(f"\n{'='*50}")
    print(f"  汇总")
    print(f"{'='*50}")
    for stage, avg in results.items():
        cr = clear_rates[stage]
        bar = '█' * int(avg / 50)
        print(f"  1-{stage[-1]}: {avg:>8.0f}  {cr:>3.0f}%  {bar}")


if __name__ == '__main__':
    main()
