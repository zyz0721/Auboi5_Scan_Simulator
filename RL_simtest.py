import time
import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from dm_rl_env import load_env

# 尝试导入 OpenCV 用于显示画面
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("警告: 未安装 opencv-python，无法显示画面，只能打印 log。")


class DMControlWrapper(gym.Env):
    def __init__(self, dm_env_instance=None):
        if dm_env_instance is None:
            self.env = load_env()
        else:
            self.env = dm_env_instance

        self.metadata = {'render.modes': ['rgb_array']}

        # 动作/观测空间适配
        action_spec = self.env.action_spec()
        self.action_space = spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            dtype=np.float32
        )
        obs_spec = self.env.observation_spec()
        dim = sum(np.prod(v.shape) for v in obs_spec.values())
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32
        )

    def _flatten_obs(self, obs_dict):
        return np.concatenate([v.ravel() for v in obs_dict.values()]).astype(np.float32)

    def reset(self, seed=None, options=None):
        time_step = self.env.reset()
        return self._flatten_obs(time_step.observation), {}

    def step(self, action):
        time_step = self.env.step(action)
        obs = self._flatten_obs(time_step.observation)
        reward = time_step.reward or 0.0
        terminated = time_step.last()
        truncated = False

        info = {}
        if reward < -5.0:
            info['collision'] = True

        # --- 修复后的误差计算逻辑 ---
        try:
            physics = self.env.physics

            # 【修复点】优先尝试获取 wrist3_Link 的位置
            if 'wrist3_Link' in physics.named.data.xpos.axes.row.names:
                ee_pos = physics.named.data.xpos['wrist3_Link']
            elif 'ee_site' in physics.named.data.site_xpos.axes.row.names:
                ee_pos = physics.named.data.site_xpos['ee_site']
            else:
                # 最后的保底：取最后一个 body 的位置
                ee_pos = physics.data.xpos[-1]

            # 获取目标点
            if self.env.task._current_base_target is not None:
                target = self.env.task._current_base_target[:3]
                dist = np.linalg.norm(target - ee_pos)
                info['dist_error'] = dist
            else:
                info['dist_error'] = 9.99  # 还没开始动

        except Exception as e:
            # 如果还报错，打印出来看看到底是啥问题
            print(f"[Debug Error] 计算误差失败: {e}")
            info['dist_error'] = -1.0  # 用 -1 表示计算出错

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.physics.render(camera_id=-1, height=480, width=640)


def main():
    # 1. 路径设置
    model_paths = ["aubo_scan_safe_policy_final.zip", "aubo_scan_safe_policy.zip", "aubo_scan_final_policy.zip"]
    model_path = None

    for p in model_paths:
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        print("❌ 错误: 找不到模型文件。请先运行 train_agent.py 进行训练！")
        return

    print(f"正在加载模型: {model_path}...")

    # 2. 加载环境
    print("⏳ 正在初始化仿真环境...")
    dm_env = load_env()
    env = DMControlWrapper(dm_env)

    # 3. 加载模型
    try:
        model = PPO.load(model_path, device='cpu')
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 4. 开始循环
    obs, _ = env.reset()
    print("\n" + "=" * 50)
    print("🎮 演示开始！")
    print("   按 'q' 键退出")
    print("=" * 50 + "\n")

    step_count = 0
    total_reward = 0

    while True:
        # A. 预测动作
        action, _ = model.predict(obs, deterministic=True)

        # B. 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        # C. 打印实时数据
        if step_count % 10 == 0:
            dist_err = info.get('dist_error', 0.0)

            status = "🟢 正常"
            if info.get('collision'):
                status = "🔴 碰撞/违规!"

            # 格式化输出
            if dist_err == -1.0:
                err_str = "NaN(计算错误)"
            else:
                err_str = f"{dist_err * 100:.1f}cm"

            print(f"Step: {step_count:04d} | 奖励: {reward:.2f} | 误差: {err_str} | 状态: {status}")

        # D. 渲染画面
        if HAS_CV2:
            rgb_array = env.render()
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

            dist_err = info.get('dist_error', 0.0)
            err_disp = f"{dist_err * 100:.1f}cm" if dist_err != -1.0 else "Error"

            cv2.putText(bgr_array, f"Error: {err_disp}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if info.get('collision'):
                cv2.putText(bgr_array, "COLLISION!", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Aubo RL Test", bgr_array)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        # E. 回合结束重置
        if terminated or truncated:
            print(f"\n 回合结束! 总奖励: {total_reward:.2f}")
            obs, _ = env.reset()
            step_count = 0
            total_reward = 0
            time.sleep(1.0)

    if HAS_CV2:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()