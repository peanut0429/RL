import os
import uuid

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT,SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.monitor import Monitor
from util_class import SaveOnBestTrainingRewardCallback, SkipFrame



def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, 4)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=(84, 84))
    monitor_dir = r'./monitor_log/'
    env = Monitor(env, filename=os.path.join(monitor_dir, str(uuid.uuid4())))
    return env

def train_fn():
    total_timesteps = 40e6 # 总共多少步
    check_frq=100000 # 十万
    num_envs = 1
    model_params = {
        'learning_rate': 3e-4,  # 学习率
        'n_steps': 2048,  # 每个环境每次更新的步数
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
    # 训练
    # model=PPO.load('monitor_log/best_model/best_model.zip', env=env, **model_params)
    model = PPO(env= env, **model_params)
    model.learn(total_timesteps=total_timesteps,callback=callback)



if __name__ == '__main__':
    train_fn()