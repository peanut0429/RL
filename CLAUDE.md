# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在 RL_SuperMario 仓库中工作时提供指导。

## 项目

RL_SuperMario —— 一个入门级强化学习项目，使用 Stable-Baselines3 的 PPO 算法训练 AI 智能体通关超级马里奥兄弟（NES）。通过 `gym-super-mario-bros` 提供标准 Gym 环境，在 GPU（CUDA）上训练。附带 B 站视频教程。

## 常用命令

- **运行训练**：`python train.py` —— 训练 40M 步（约需数小时），模型保存到 `monitor_log/best_model/`。
- **测试已训练模型**：`python test_model.py` —— 加载 `monitor_log/best_model/best_model.zip` 并在窗口中渲染游戏画面。
- **快速验证环境**：`python test/test_mario.py` —— 随机动作运行 5000 步，验证 `gym-super-mario-bros` 环境是否正常工作。

## 依赖约束

- **Python 版本**：项目未锁定特定 Python 版本，但依赖链要求兼容 gym 0.21.0 + SB3 1.6.0（见 `requirements.txt`）。
- **关键版本锁定**：
  - `gym==0.21.0` —— 旧版 gym，为兼容 `gym-super-mario-bros==7.3.0`（依赖 gym 的 4 元组状态返回格式）。
  - `stable_baselines3==1.6.0` —— 不能更高，因为新版 SB3 需要新版 gym。
  - `protobuf==3.20.3` —— 解决 TensorBoard 兼容性问题。
  - `nes_py==8.2.1` —— NES 模拟器后端。
  - `numpy==1.24.4` —— 固定版本避免与旧版 gym 冲突。
- **GPU**：训练默认使用 CUDA（`device: 'cuda'`），需 NVIDIA GPU + CUDA 环境。
- 依赖安装难度较高（作者原话），建议参考 B 站视频教程逐步配置。

## 架构

```
train.py         → 训练入口 —— 构建环境、配置 PPO 参数、执行训练
util_class.py    → 工具类 —— 自定义回调、帧跳过包装器、奖励包装器
test_model.py    → 模型测试 —— 加载已训练模型并在渲染窗口中演示
test/
  test_mario.py  → 环境快速验证脚本（随机动作）
requirements.txt → 依赖列表（含版本锁定和中文注释）
```

### `train.py`
训练编排的核心文件：
- 使用 `gym_super_mario_bros.make('SuperMarioBros-v2')` 创建环境。
- 动作空间：`SIMPLE_MOVEMENT`（7 个离散动作，仅方向 + A/B 键组合）。
- 环境包装器链：
  1. `JoypadSpace` —— 将原始 NES 按键映射为简化动作空间
  2. `SkipFrame(skip=4)` —— 跳帧，每 4 帧执行一次动作并累积奖励
  3. `GrayScaleObservation` —— 彩色转灰度（保持 channel 维度）
  4. `ResizeObservation(shape=84×84)` —— 降采样到 84×84
  5. `Monitor` —— 记录 episode 奖励到 CSV
  6. `VecFrameStack(4)` —— 堆叠最近 4 帧（提供运动信息）
- PPO 超参数：
  - `learning_rate=3e-4`, `n_steps=2048`, `batch_size=8192`
  - `ent_coef=0.1`（较高熵系数，鼓励探索）
  - `gamma=0.95`, `GAE λ=0.95`
  - `n_epochs=10`, `target_kl=0.03`（KL 早停）
  - `clip_range=0.1`, `max_grad_norm=0.8`
- 训练总量：40M 步。
- 使用 `SubprocVecEnv` 并行环境（默认 `num_envs=1`，可调）。
- 通过 `SaveOnBestTrainingRewardCallback` 每 10 万步检查一次并在最佳模型时保存。

### `util_class.py`
三个自定义类：
- **`SaveOnBestTrainingRewardCallback`**：继承 `BaseCallback`，每 `check_freq` 步检查 Monitor 日志中最近 100 个 episode 的平均奖励，如果刷新最高记录则保存模型到 `monitor_log/best_model/`。
- **`SkipFrame`**：继承 `gym.Wrapper`，每 `skip` 帧执行同一个动作并累加奖励，减少冗余帧处理。
- **`RewardWrapper`**：继承 `gym.RewardWrapper`（当前未被 `train.py` 使用，为可选增强），提供基于游戏内信息的额外奖励：
  - 获得金币 +200
  - 触碰旗杆（通关）+200
  - 得分增加 = 奖励差值
  - 失去生命 -500

### `test_model.py`
- 从 `monitor_log/best_model/best_model.zip` 加载 PPO 模型。
- 运行 10000 步推理循环，`env.render('human')` 显示游戏画面。
- 每步打印奖励值，episode 结束自动 reset。

### `test/test_mario.py`
- 极简环境验证脚本：随机动作运行 5000 步。
- 用于确认 `gym-super-mario-bros` + `nes_py` 安装正确。

## 关键 RL 设计要点

- 环境是**标准 Gym 接口**（`gym-super-mario-bros`），无需自定义 `gym.Env` 子类 —— 与 EldenRL 截然不同。
- 观察空间是**纯视觉**：4 帧堆叠的 84×84 灰度图像（CNN 策略），没有额外的结构化状态（血量、动作历史等）。
- 动作空间是**简化离散**（SIMPLE_MOVEMENT，7 个动作），而非完整 NES 按键组合。
- 帧跳过（skip=4）以约 15fps 的有效频率运行，大幅减少计算开销。
- 训练在 **GPU** 上运行（CNN 策略需要），与 EldenRL 用 CPU 训练不同。
- 使用 `SubprocVecEnv` + `VecFrameStack` 的向量化环境管道。
- 模型选择基于训练奖励（最近 100 episode 均值），不涉及外部评估。

## 与 EldenRL 的关系

RL_SuperMario 和 EldenRL 同属 `RL2` 工作区，均为强化学习 + 游戏 AI 项目。RL_SuperMario 定位为**入门教学项目**（有完整视频教程），EldenRL 定位为**高阶研究项目**。详细对比见 `RL2/COMPARISON.md`。
