import os
import uuid
import torch

# GPU 加速优化
torch.backends.cudnn.benchmark = True  # 自动寻找最优卷积算法
torch.set_num_threads(1)  # 避免 PyTorch 和子进程抢 CPU 核心

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT,SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.monitor import Monitor
from util_class import SaveOnBestTrainingRewardCallback, SkipFrame, RewardWrapper



def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = RewardWrapper(env)  # 自定义奖励
    env = SkipFrame(env, 4)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=(84, 84))
    monitor_dir = r'./monitor_log/'
    env = Monitor(env, filename=os.path.join(monitor_dir, str(uuid.uuid4())))
    return env

def train_fn():
    total_timesteps = 40e6 # 总共多少步
    check_frq=100000 # 十万
    num_envs = 14  # 12 核 CPU 最大化利用
    n_steps = 1024  # 更频繁更新 GPU，减少等待
    model_params = {
        'learning_rate': 3e-4,  # 学习率
        'n_steps': n_steps,  # 每个环境每次更新的步数
        'batch_size': 8192,  # 随机抽取多少数据
        'ent_coef': 0.1,  # 熵项系数, 影响探索性

        'gamma': 0.95,  # 短视或者长远
        'clip_range': 0.1,  # 截断范围
        'gae_lambda':0.95,  # GAE参数
        "target_kl": 0.03,  # 设置KL散度早停阈值
        'n_epochs': 10,  # 更新次数
        "vf_coef": 0.5,  # 增加价值函数权重
        "max_grad_norm": 0.8,  # 梯度裁剪
        'device': 'cuda',

        # log
        'tensorboard_log':r'./tensorboard_log/',
        'verbose':1,
        'policy':"CnnPolicy"
    }

    # LOG
    monitor_dir = r'./monitor_log/'
    os.makedirs(monitor_dir, exist_ok=True)
    callback = SaveOnBestTrainingRewardCallback( check_frq,monitor_dir)

    env = SubprocVecEnv([make_env for _ in range(num_envs)])
    env = VecFrameStack(env, 4, channels_order='last')  # 帧叠加

    # 训练：优先级——微调模型 > 续训 checkpoint > 从头开始
    finetune_model = 'base_model.zip'  # 把旧模型放这，自动微调
    latest_model = os.path.join(monitor_dir, 'best_model', 'latest_model.zip')

    if os.path.exists(finetune_model):
        print(f"微调模式: 加载基础模型 {finetune_model}")
        model = PPO.load(finetune_model, env=env)
        model.learning_rate = 1e-4  # 微调用更低学习率
        print(f"学习率已降至: {model.learning_rate}")
    elif os.path.exists(latest_model):
        print(f"续训: {latest_model}")
        model = PPO.load(latest_model, env=env)
    else:
        print("从头开始训练")
        model = PPO(env=env, **model_params)

    model.learn(total_timesteps=total_timesteps, callback=callback)



if __name__ == '__main__':
    train_fn()