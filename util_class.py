from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import gym
import os


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq, log_dir, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    # def _init_callback(self) -> None:
    def _init_callback(self):
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    # def _on_step(self) -> bool:
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # print('self.n_calls: ',self.n_calls)
            # self.model.save(self.save_path)
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                # if self.verbose > 0:
                #     print("Num timesteps: {}".format(self.num_timesteps))
                #     print(
                #         "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}"
                #         .format(self.best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model at {x[-1]} timesteps")
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(f'{self.save_path}/best_model{x[-1]}')
                    self.model.save(f'{self.save_path}/latest_model')  # 用于续训

        return True


class SkipFrame(gym.Wrapper):
    """SkipFrame是可以实现跳帧操作。因为连续的帧变化不大，
    我们可以跳过n个中间帧而不会丢失太多信息。第n帧聚合每个跳过帧上累积的奖励。"""

    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class RewardWrapper(gym.core.RewardWrapper):
    """
    自定义奖励包装器 —— 在基础 x 位移奖励上叠加额外引导。

    设计原则：额外奖励要和基础奖励（每步 +1~15）同量级，
    不能喧宾夺主。所有惩罚项都要有"攒够才触发"的容错窗口。
    """

    def __init__(self, env):
        super(RewardWrapper, self).__init__(env)
        self._coins = 0
        self._life = 2
        self._x_max = 0       # 本局最远 x 坐标
        self._stuck_timer = 0  # 连续无进展的步数

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        info_dict = info
        coins = info_dict['coins']
        life = info_dict['life']
        flag_get = info_dict['flag_get']
        x_pos = info_dict.get('x_pos', self._x_max)

        # --- 1. 通关奖励 ---
        if flag_get:
            reward += 500  # 等于 30-50 步移动的收益，够大但不夸张

        # --- 2. 吃金币 ---
        if coins > self._coins:
            reward += 10  # 等于 1 步移动，是正反馈但不喧宾夺主
            self._coins = coins

        # --- 3. 死亡惩罚 ---
        if life < self._life:
            reward -= 50  # 等于 5 步移动，有痛感但不会崩盘
            self._life = life

        # --- 4. 卡住惩罚（容错窗口：100 步无进展才触发） ---
        if x_pos > self._x_max + 5:  # 进步超过 5 像素才算"有进展"
            self._x_max = x_pos
            self._stuck_timer = 0
        else:
            self._stuck_timer += 1

        if self._stuck_timer > 100:  # 约 7 秒不动
            reward -= 1  # 极轻惩罚，只推动探索

        # --- 重置 ---
        if done:
            self._coins = 0
            self._life = 2
            self._x_max = 0
            self._stuck_timer = 0

        return observation, reward, done, info

# class ResizeObservation(gym.ObservationWrapper):
#     def __init__(self, env, shape):
#         super(ResizeObservation, self).__init__(env)
#         if isinstance(shape, int):
#             self.shape = (shape, shape)
#         else:
#             self.shape = tuple(shape)
#
#         obs_shape = self.shape + self.observation_space.shape[2:]
#         self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
#
#     def observation(self, observation):
#         import cv2
#         observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
#         return observation
