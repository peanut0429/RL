# 效果图

<img src="https://raw.githubusercontent.com/jusway/RL_SuperMario/refs/heads/main/readme_file/record.gif" alt="record" style="zoom: 200%;" />

# 项目简介

一个入门级强化学习项目，使用 Stable-Baselines3 的 PPO 算法训练 AI 智能体通关超级玛丽兄弟（NES）。AI 仅通过 84×84 的灰度游戏画面学习"什么画面下按什么键能得高分"，最终学会通关。

**项目亮点**：
- 🤖 纯视觉输入：AI 只能看到像素画面，没有任何人为特征工程
- 🧪 五轮完整实验：覆盖从单关基线到多关泛化、从基础奖励到奖励塑形的完整探索
- ☁️ 云 GPU 训练：建立了 AutoDL 远程训练全流程（环境配置 → 后台训练 → 远程监控 → 结果回收）
- 📝 详尽文档：实验报告、原理详解、云训练方法论、阶段报告、实习总结

**训练成果**：

| 关卡 | 通关率 | 说明 |
|------|--------|------|
| 1-1 | 95% | ✅ 几乎完美 |
| 1-4 | 75% | ✅ 城堡关较稳定 |
| 1-2 | 30% | ⚠️ 进行中，通过奖励函数优化提升中 |
| 1-3 | 0% | ❌ x 位移奖励的系统性盲区 |

# 快速开始

```bash
# 训练
python train.py

# 测试已训练模型
python test_model.py

# 手动游玩
python play_mario.py
```

# B 站教程（已完结）

【强化学习训练超级马里奥（stablebaseline3框架）】 https://www.bilibili.com/video/BV1CERYY3EjA/?p=2&share_source=copy_web&vd_source=dbb60edcfbcee053b3e3e7aa16ec24be

# 文档索引

| 文档 | 内容 |
|------|------|
| `PRINCIPLES.md` | RL 入门原理与参数详解 |
| `EXPERIMENT_REPORT.md` | 五轮实验完整报告 |
| `PROGRESS_REPORT_16M.md` | 16M 步训练阶段分析 |
| `INTERNSHIP_SUMMARY.md` | 实习工作总结 |
| `CLOUD_TRAINING.md` | 云 GPU 训练方法论 |

# 前置知识截图

![](https://github.com/jusway/RL_SuperMario/blob/main/readme_file/%E8%AF%BE%E7%A8%8B%E5%89%8D%E7%BD%AE%E7%9F%A5%E8%AF%86.png?raw=true)

# 视频列表

![image-20250326184251931](https://github.com/jusway/RL_SuperMario/blob/main/readme_file/%E8%A7%86%E9%A2%91%E9%80%89%E9%9B%86.png?raw=true)
