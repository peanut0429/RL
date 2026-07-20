# 云 GPU 训练工作流方法论

本文档不仅说明如何在 AutoDL 上跑通 RL_SuperMario，更重要的是讲清楚**云算力训练的通用工作流**——你可以把这套方法复用到任何需要租用 GPU 的项目上。

---

## 目录

1. [核心心智模型](#1-核心心智模型)
2. [详细工作流](#2-详细工作流)
3. [AutoDL 实战：RL_SuperMario](#3-autodl-实战rl_supermario)
4. [常见问题与排错](#4-常见问题与排错)
5. [脚本速查表](#5-脚本速查表)

---

## 1. 核心心智模型

### 云 GPU 训练的四个阶段

```
本地准备  →  环境配置  →  训练执行  →  结果回收
(一次性)    (每个实例一次)  (核心耗时)   (训练结束后)
```

**关键原则**：

| 原则 | 说明 |
|------|------|
| **算力与代码分离** | 代码永远在本地用 git 管理，云上只是执行环境。不要把唯一代码放在云服务器上 |
| **持久化存储独立** | 模型 checkpoint、日志放在 AutoDL 的 `/root/autodl-fs`（网盘）或定期下载，不要存系统盘 |
| **后台运行** | 用 `tmux`/`screen`/`nohup` 确保断开 SSH 后训练不中断 |
| **可复现** | 脚本化一切手动操作，避免每次租用实例都要"回忆上次怎么配的" |

### 哪些项目适合租 GPU

- 训练步数在 **百万级别以上** 的 RL/DL 项目
- 模型涉及 **CNN 策略**（图像输入），CPU 训练极慢
- 本地没有 NVIDIA 显卡或者显存不够
- 需要多卡并行或大 batch size

### 哪些不适合

- 训练量很小（几分钟就跑完）—— 用本地 CPU 或 Colab 免费版即可
- 需要大量交互式调试（Jupyter 可以用，但成本高）
- 数据量巨大（TB 级，网络传输是瓶颈）

---

## 2. 详细工作流

### 2.1 本地准备阶段（一次性）

在租用实例之前，先把这些事情做好：

```
1. 代码用 git 管理好（GitHub/Gitee 私有仓库）
2. requirements.txt 列清楚所有依赖（含版本号）
3. 确认代码量级（多少步、预估时间）
4. 准备好验证脚本（确认环境 OK 的最小测试）
```

**为什么用 git 而不是直接上传文件夹？**
- 版本可控：训练过程中改了参数可以追溯
- 增量上传：改一行代码不需要重新传整个项目
- 安全：代码不会因为实例销毁而丢失

### 2.2 选择实例规格

| 考量 | 建议 |
|------|------|
| **GPU 型号** | RL 项目（小网络 + 环境交互多）优先 CPU 强的 GPU。RTX 3080/3090/4090 都可以，没必要上 A100 |
| **显存** | 小规模 RL 6-10GB 足够。如果跑 batch_size 很大的监督学习，按 batch 估算显存 |
| **计费模式** | 不确定跑多久 → 按量计费；确定要跑几天 → 包天/包周更划算 |
| **镜像** | 选带 PyTorch/CUDA 的，省去自己装 CUDA 驱动和 PyTorch 的麻烦 |

### 2.3 环境配置脚本化

每次配环境的痛苦在于：缺依赖、版本冲突、忘了某一步。解决方案：

**写一个 `setup.sh`，包含：**
1. `apt-get install` 系统库（`libsdl2-dev`、`xvfb` 等编译/运行需要的）
2. `pip install -r requirements.txt` Python 依赖
3. 环境验证（跑一段最小代码确认 CUDA + 各库正常）

**写到脚本里，不要手动敲命令。** 这样：
- 换一个实例能一键配好
- 隔三个月回来还能跑
- 给别人用也方便

### 2.4 训练必须后台化

SSH 连接断开 = 前台进程被杀。**必须用 tmux**：

```bash
# 创建独立会话
tmux new -s train

# 在会话里启动训练
python train.py

# 断开 SSH（训练继续运行）
# 按 Ctrl+B，再按 D（detach）

# 重新连上 SSH 后，恢复查看
tmux attach -t train
```

**为什么 tmux 优于 nohup？**
- nohup 的输出重定向到文件，无法交互
- tmux 允许你随时 attach 进去看实时输出、Ctrl+C 中断、甚至临时改代码
- tmux 窗口可以切多 pane（一边看训练日志，一边 tensorboard）

### 2.5 监控训练进度

**TensorBoard** 是标配：

```bash
# 在 tmux 的另一个 pane 中启动
tensorboard --logdir tensorboard_log --port 6006 --bind_all
```

AutoDL 提供了**端口转发**功能：在实例详情页找到"端口转发"，把 6006 端口映射出来，本地浏览器就能直接访问 TensorBoard。

**除了 TensorBoard，还要关注：**
- `nvidia-smi`：GPU 利用率和显存是否正常
- `htop`：CPU/内存情况
- 训练日志文件有没有在正常增长

### 2.6 结果回收

训练结束后需要拿回来的东西：

| 文件 | 位置 | 用途 |
|------|------|------|
| 模型 checkpoint | `monitor_log/best_model/` | 最终产物，用于测试/部署 |
| TensorBoard 日志 | `tensorboard_log/` | 分析训练曲线 |
| 训练日志 | `training.log` | 排查训练中的异常 |

**回收方式（按推荐度排序）：**
1. **scp/rsync**：最直接，适合一次性打包下载
2. **AutoDL 网盘**：存到 `/root/autodl-fs/`，在多个实例间共享，关机不丢失
3. **git lfs**：适合模型文件版本管理，但大文件有额外费用
4. **对象存储（OSS/S3）**：适合大量数据

**重要：下载完确认文件完整后再销毁实例！**

---

## 3. AutoDL 实战：RL_SuperMario

### 3.1 租用实例

1. 打开 [AutoDL](https://www.autodl.com/)，注册/登录
2. **算力市场** → 选一个 GPU 实例
   - 推荐：RTX 3080 / 3090（性价比高）
   - 镜像：`PyTorch 2.x + Python 3.10 + CUDA 12.x`
3. 创建实例，等 1-2 分钟启动

### 3.2 连接实例

AutoDL 提供多种连接方式：

- **JupyterLab**：浏览器 IDE，适合交互式调试（点"JupyterLab"按钮）
- **SSH**：终端连接，适合脚本化操作
- **本脚本使用终端方式（SSH 或 JupyterLab 的 Terminal）**

### 3.3 一键配置

```bash
# 1. 克隆项目到实例
cd /root/autodl-tmp
git clone <你的仓库地址>
cd RL_SuperMario

# 2. 运行配置脚本
bash scripts/setup_autodl.sh
```

脚本会自动：安装系统库 → 创建虚拟环境 → 安装 Python 依赖 → 验证 CUDA + gym 环境。

### 3.4 启动训练

```bash
bash scripts/run_train.sh
```

脚本会：
- 在 tmux 会话中启动 `train.py`
- 训练日志同时输出到终端和 `training.log` 文件
- 断开 SSH 后继续运行

训练估计时间（参考）：
- RTX 3090：40M 步约 6-10 小时
- RTX 3080：40M 步约 8-14 小时

### 3.5 查看 TensorBoard

```bash
# 在 tmux 中另开 pane，或新开一个 SSH
tensorboard --logdir tensorboard_log --port 6006 --bind_all
```

然后在 AutoDL 控制台 → 实例详情 → 端口转发 → 添加 6006 端口。

### 3.6 回收结果

训练结束后，在**本地终端**执行：

```bash
# 方式一：scp 下载整个 monitor_log 和 tensorboard_log
scp -rP <SSH端口> root@<实例IP>:/root/autodl-tmp/RL_SuperMario/monitor_log ./
scp -rP <SSH端口> root@<实例IP>:/root/autodl-tmp/RL_SuperMario/tensorboard_log ./

# 方式二：先传到 AutoDL 网盘（持久化存储）
# 在实例上执行：
cp -r monitor_log /root/autodl-fs/mario_monitor_log
cp -r tensorboard_log /root/autodl-fs/mario_tensorboard_log
# 以后从网盘下载或在其他实例上继续训练
```

### 3.7 本地测试模型

把 `best_model.zip` 下载到本地的 `monitor_log/best_model/` 目录下，直接运行：

```bash
python test_model.py
```

本地有显示器，可以直接看到马里奥闯关的渲染画面。

---

## 4. 常见问题与排错

### nes_py 安装报错

```
error: Microsoft Visual C++ 14.0 is required  (Windows)
```

或

```
fatal error: SDL.h: No such file or directory  (Linux)
```

**解决**：Linux 上先装 `libsdl2-dev`（脚本已包含），Windows 上装 Visual C++ Build Tools。

### gym-super-mario-bros ROM 缺失

```
FileNotFoundError: Could not find the ROM file
```

**解决**：`gym-super-mario-bros` 自带了 ROM，如果仍然找不到：
```bash
# 手动安装 nes_py 的 ROM
pip install gym-super-mario-bros --force-reinstall
```

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**解决**：
- 减小 `batch_size`（当前 8192，可改为 4096）
- 减小 `n_steps`（当前 2048，可改为 1024）
- 换更大显存的 GPU

### 训练中途断了怎么恢复

当前代码不支持断点续训。但可以修改 `train.py` 注释的那行：

```python
# 把这行取消注释：
model = PPO.load('monitor_log/best_model/best_model.zip', env=env, **model_params)
# 并注释掉下面这行：
# model = PPO(env=env, **model_params)
```

下次训练会自动从上次最佳模型继续。

### test_model.py 在服务器上报错

`test_model.py` 第 29 行调用了 `env.render('human')`，这在无显示器服务器上会报错。

**解决**：模型评估应该在**本地**进行（下载模型到自己的电脑上跑）。如果一定要在服务器上验证，可以用 `xvfb` 虚拟显示器 + 录制视频的方式。

---

## 5. 脚本速查表

| 脚本 | 用途 | 在哪执行 |
|------|------|----------|
| `scripts/setup_autodl.sh` | 一键配置环境 | 云实例 |
| `scripts/run_train.sh` | 启动训练（tmux） | 云实例 |
| `scripts/eval_model.sh` | 模型评估/录视频 | 云实例（可选） |

### tmux 常用操作

```
tmux new -s train          # 创建名为 train 的会话
tmux attach -t train       # 连接到已有会话
Ctrl+B 然后 D               # 断开连接（训练继续跑）
Ctrl+B 然后 [               # 进入滚动模式，上下翻页
tmux kill-session -t train # 杀掉会话
tmux ls                    # 列出所有会话
```

### TensorBoard 端口映射

```
AutoDL 控制台 → 实例 → 更多 → 端口转发 → 添加 6006
本地浏览器访问：http://<转发后的地址>
```

---

## 附录：这套方法的通用模板

把这套流程抽象成模板，以后任何新项目都可以套用：

```
project/
├── scripts/
│   ├── setup_cloud.sh       # <-- 改依赖列表即可复用
│   ├── run_train.sh         # <-- 改 python train.py 即可复用
│   └── download_results.sh  # <-- 改路径即可复用
├── requirements.txt         # <-- 锁死版本号
├── CLOUD_TRAINING.md        # <-- 本文档，项目说明
└── ...
```

**泛化 checklist**：

- [ ] 项目能在一个脚本里启动训练（`python train.py` 不需要交互）
- [ ] 训练参数都是硬编码或配置文件驱动的（不需要手动输入）
- [ ] 依赖锁死了版本号
- [ ] 有一个最小验证脚本（确认环境 OK）
- [ ] 模型/日志的输出路径明确
- [ ] 知道训练大约要跑多久（方便选实例和预估费用）
