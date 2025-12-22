import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from dm_rl_env import load_env
from train_agent import DMControlWrapper

# 尝试导入 OpenCV 用于显示画面
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("警告: 未安装 opencv-python，无法显示画面，只能打印 log。")
    print("建议安装: pip install opencv-python")


def main():
    # 1. 路径设置
    model_path = "aubo_scan_final_policy.zip"

    print(f"正在加载训练好的模型: {model_path}...")
    try:
        # device='cpu' 消除 GPU 警告
        model = PPO.load(model_path, device='cpu')
    except FileNotFoundError:
        print("错误: 找不到模型文件。请先运行 train_agent.py 进行训练！")
        return

    # 2. 加载仿真环境
    print("加载仿真环境...")
    dm_env = load_env()

    # 【关键】这里传入的是实例，Wrapper 必须直接接收实例
    env = DMControlWrapper(dm_env)

    # 3. 开始可视化循环
    obs, _ = env.reset()

    print("开始演示... (按 Ctrl+C 停止, 或在图像窗口按 'q' 退出)")

    step_count = 0
    total_reward = 0

    try:
        while True:
            # A. 模型预测
            action, _ = model.predict(obs, deterministic=True)

            # B. 环境执行
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # C. 渲染与显示
            if HAS_CV2:
                # 获取 RGB 图像
                rgb_array = env.render()
                # OpenCV 默认使用 BGR，需要转换颜色空间
                bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

                cv2.imshow("Aubo RL Simulation", bgr_array)

                # 按 'q' 退出
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
            else:
                # 如果没有 OpenCV，稍微睡一下模拟实时
                time.sleep(0.02)

            # D. 回合结束处理
            if terminated or truncated:
                print(f"回合结束! 总步数: {step_count}, 总奖励: {total_reward:.2f}")
                obs, _ = env.reset()
                step_count = 0
                total_reward = 0
                time.sleep(0.5)  # 休息一下

    except KeyboardInterrupt:
        print("\n演示结束")
    finally:
        env.close()
        if HAS_CV2:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()