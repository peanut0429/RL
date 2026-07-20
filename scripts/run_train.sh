#!/bin/bash
# ============================================================
# 训练启动脚本
# 用途：在 tmux 会话中启动训练，断开 SSH 后继续运行
# 用法：bash scripts/run_train.sh
# ============================================================
set -e

# 激活虚拟环境（如果存在）
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# 确保日志目录存在
mkdir -p monitor_log tensorboard_log

# ---------- 配置 ----------
SESSION_NAME="mario_train"

# 检查是否已在 tmux 中运行
if [ -n "$TMUX" ]; then
    echo "已在 tmux 会话中，直接启动训练..."
    python train.py
    exit 0
fi

# 如果 tmux 会话已存在，先提示
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  tmux 会话 '$SESSION_NAME' 已存在"
    echo ""
    echo "选择操作:"
    echo "  1) 附加到已有会话（查看实时输出）"
    echo "  2) 杀掉旧会话，重新开始训练"
    echo "  3) 退出"
    read -p "请输入 [1/2/3]: " choice
    case $choice in
        1)
            echo "附加到已有会话... (按 Ctrl+B 再按 D 可安全断开)"
            sleep 1
            tmux attach -t "$SESSION_NAME"
            exit 0
            ;;
        2)
            tmux kill-session -t "$SESSION_NAME"
            echo "已杀掉旧会话"
            ;;
        3)
            exit 0
            ;;
    esac
fi

# 创建新的 tmux 会话并启动训练
echo "============================================"
echo "  启动训练 (tmux 会话: $SESSION_NAME)"
echo "============================================"
echo ""
echo "  ⚡ 训练启动后会持续运行，即使断开 SSH 也不会中断"
echo "  📊 训练日志保存在: tensorboard_log/"
echo "  💾 最佳模型保存在: monitor_log/best_model/"
echo ""
echo "  常用操作:"
echo "    查看实时日志:  tmux attach -t $SESSION_NAME"
echo "    安全断开连接:  按 Ctrl+B 再按 D"
echo "    查看 TensorBoard: tensorboard --logdir tensorboard_log --port 6006 --bind_all"
echo ""

tmux new-session -d -s "$SESSION_NAME" "python train.py 2>&1 | tee training.log"

echo "训练已在后台启动！"
echo ""
echo "查看实时输出: tmux attach -t $SESSION_NAME"
