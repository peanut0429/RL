#!/bin/bash
# ============================================================
# 本地监控脚本 —— 通过 SSH 查看 AutoDL 训练状态
# 用法：bash scripts/check_remote.sh <SSH端口> <实例IP>
# 例：bash scripts/check_remote.sh 12345 192.168.1.100
# ============================================================

PORT=${1:?"请提供 SSH 端口"}
HOST=${2:?"请提供实例 IP"}

echo "=== Mario 训练状态 ==="

# 1. GPU 状态
echo ""
echo "[GPU]"
ssh -p $PORT root@$HOST "nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader" 2>/dev/null

# 2. 最近训练日志
echo ""
echo "[最近日志]"
ssh -p $PORT root@$HOST "tail -5 ~/autodl-tmp/RL/training.log 2>/dev/null || tail -5 ~/autodl-tmp/RL_SuperMario/training.log 2>/dev/null" 2>/dev/null

# 3. 当前步数
echo ""
echo "[训练进度]"
ssh -p $PORT root@$HOST "python3 -c \"
import pandas as pd, glob, os, numpy as np
files = glob.glob('autodl-tmp/RL/monitor_log/*.csv') or glob.glob('autodl-tmp/RL_SuperMario/monitor_log/*.csv')
if files:
    latest = max(files, key=os.path.getmtime)
    df = pd.read_csv(latest, comment='#')
    rewards = df['r'].dropna()
    if len(rewards) > 0:
        print(f'Episode 总数: {len(rewards)}')
        print(f'最近 10 局平均: {rewards.tail(10).mean():.0f}')
        print(f'最近 100 局平均: {rewards.tail(100).mean():.0f}')
        print(f'最高: {rewards.max():.0f}')
\"" 2>/dev/null

echo ""
echo "=== 连接训练终端 ==="
echo "ssh -p $PORT root@$HOST -t 'tmux attach -t mario_train'"
