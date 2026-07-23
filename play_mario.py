"""
手动玩马里奥 —— 你自己来操作，看看你能得几分
按键映射（SIMPLE_MOVEMENT）：
    方向键 →：向右走
    方向键 ←：向左走
    空格键  ：跳跃
    数字 0  ：什么也不按（NOOP）

按 ESC 退出。
"""
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import pygame
from pygame.locals import KEYDOWN, KEYUP, K_RIGHT, K_LEFT, K_SPACE, K_0, K_ESCAPE, K_LSHIFT, K_UP, K_DOWN

# ===== 设置 =====
ACTION_MODE = SIMPLE_MOVEMENT   # SIMPLE_MOVEMENT(7键) 或 COMPLEX_MOVEMENT(12键)
WORLD = 1
STAGE = 1
# ================

stage_name = f'SuperMarioBros-{WORLD}-{STAGE}-v2'

# 创建环境
env = gym_super_mario_bros.make(stage_name)
env = JoypadSpace(env, ACTION_MODE)
env.reset()

# 初始化 pygame
pygame.init()
screen = pygame.display.set_mode((512, 480))
pygame.display.set_caption(f"手动马里奥 —— {WORLD}-{STAGE} (ESC退出)")

# SIMPLE_MOVEMENT 映射：0-NOOP, 1-right, 2-right+A, 3-right+B, 4-right+A+B, 5-A, 6-left
if ACTION_MODE is SIMPLE_MOVEMENT:
    KEY_MAP = {
        K_RIGHT: 1,           # 右
        K_LEFT: 6,            # 左
        K_SPACE: 5,           # 原地跳
        K_0: 0,               # NOOP
    }
    # 组合键：方向+跳
    print("\n操作说明:")
    print("  →  向右走")
    print("  ←  向左走")
    print("  空格 跳跃")
    print("  → + 空格  向右跳（按住两个键）")
    print("  0  NOOP（什么都不做）")
    print("  ESC 退出\n")
else:
    # COMPLEX_MOVEMENT: 0-NOOP, 1-up, 2-down, 3-left, 4-left+A, 5-left+B, 6-left+A+B,
    #                   7-right, 8-right+A, 9-right+B, 10-right+A+B, 11-A, 12-B
    KEY_MAP = {
        K_RIGHT: 7,           # 右
        K_LEFT: 3,            # 左
        K_SPACE: 11,          # 跳
        K_LSHIFT: 12,         # 跑
        K_UP: 1,              # 上（爬藤蔓）
        K_DOWN: 2,            # 下（蹲/管道）
        K_0: 0,               # NOOP
    }
    print("\n操作说明:")
    print("  →  向右走")
    print("  ←  向左走")
    print("  空格 跳跃")
    print("  Shift 加速跑")
    print("  → + 空格 + Shift  跑跳")
    print("  ↑ 爬藤蔓  ↓ 蹲/下管道")
    print("  0  NOOP")
    print("  ESC 退出\n")

# 获取游戏画面
obs = env.render(mode='rgb_array')
screen.blit(pygame.surfarray.make_surface(obs.swapaxes(0, 1)), (0, 0))
pygame.display.flip()

clock = pygame.time.Clock()
done = True
action = 0
total_reward = 0
episode_count = 0
step_count = 0

print(f"第 {episode_count+1} 局开始！")

running = True
while running:
    clock.tick(60)  # 60fps

    # --- 事件处理：只处理退出 ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False

    # --- 每帧读取按键（不管有没有事件）---
    keys = pygame.key.get_pressed()
    action = 0  # 默认 NOOP

    if ACTION_MODE is SIMPLE_MOVEMENT:
        if keys[K_RIGHT] and keys[K_SPACE]:
            action = 2   # 右 + 跳
        elif keys[K_RIGHT]:
            action = 1   # 右
        elif keys[K_LEFT]:
            action = 6   # 左
        elif keys[K_SPACE]:
            action = 5   # 跳

    elif ACTION_MODE is COMPLEX_MOVEMENT:
        if keys[K_RIGHT] and keys[K_SPACE] and keys[K_LSHIFT]:
            action = 10  # 右 + 跳 + 跑
        elif keys[K_RIGHT] and keys[K_SPACE]:
            action = 8   # 右 + 跳
        elif keys[K_RIGHT] and keys[K_LSHIFT]:
            action = 9   # 右 + 跑
        elif keys[K_LEFT] and keys[K_SPACE] and keys[K_LSHIFT]:
            action = 6   # 左 + 跳 + 跑
        elif keys[K_LEFT] and keys[K_SPACE]:
            action = 4   # 左 + 跳
        elif keys[K_LEFT] and keys[K_LSHIFT]:
            action = 5   # 左 + 跑
        elif keys[K_RIGHT]:
            action = 7   # 右
        elif keys[K_LEFT]:
            action = 3   # 左
        elif keys[K_SPACE]:
            action = 11  # 跳
        elif keys[K_LSHIFT]:
            action = 12  # 跑
        elif keys[K_UP]:
            action = 1   # 上
        elif keys[K_DOWN]:
            action = 2   # 下

    # 环境步进
    if done:
        env.reset()
        step_count = 0
    else:
        step_count += 1

    obs, reward, done, info = env.step(action)
    total_reward += reward

    # 渲染
    obs_rendered = env.render(mode='rgb_array')
    screen.blit(pygame.surfarray.make_surface(obs_rendered.swapaxes(0, 1)), (0, 0))

    # 显示信息
    font = pygame.font.SysFont('Arial', 18)
    text_surface = font.render(
        f"Reward: {total_reward:.0f} | Episode: {episode_count+1} | " +
        f"Coins: {info.get('coins',0)} | Life: {info.get('life',2)}",
        True, (255, 255, 255), (0, 0, 0)
    )
    screen.blit(text_surface, (10, 10))

    pygame.display.flip()

    if done:
        episode_count += 1
        flag_text = "🏁 通关！" if info.get('flag_get', False) else "💀 "
        print(f"{flag_text} 第 {episode_count} 局结束 | 奖励: {total_reward:.0f} | 步数: {step_count}")
        total_reward = 0

pygame.quit()
env.close()
print(f"\n共玩了 {episode_count} 局，做得好！")
