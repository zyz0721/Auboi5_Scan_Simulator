import time
import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from dm_rl_env import load_env
from scipy.spatial.transform import Rotation as R

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

        # --- 计算详细误差 (位置 + 角度) ---
        try:
            physics = self.env.physics

            # 1. 获取 wrist3_Link 的位置(pos)和旋转矩阵(mat)
            if 'wrist3_Link' in physics.named.data.xpos.axes.row.names:
                ee_pos = physics.named.data.xpos['wrist3_Link']
                ee_mat = physics.named.data.xmat['wrist3_Link'].reshape(3, 3)
            else:
                ee_pos = physics.data.xpos[-1]
                ee_mat = physics.data.xmat[-1].reshape(3, 3)

            # 2. 获取当前目标
            task = self.env.task
            if task._current_base_target is not None:
                target_pos = task._current_base_target[:3]
                target_euler = task._current_base_target[3:]

                # 计算位置误差 (Euclidean Distance)
                dist = np.linalg.norm(target_pos - ee_pos)
                info['dist_error'] = dist

                # 计算角度误差 (Geodesic Distance on SO3)
                # 目标旋转矩阵
                target_rot = R.from_euler('xyz', target_euler, degrees=False)
                target_mat = target_rot.as_matrix()

                # 计算旋转差: R_diff = R_curr * R_target^T
                # Trace(R_diff) = 1 + 2cos(theta)
                r_diff = np.dot(ee_mat, target_mat.T)
                trace = np.trace(r_diff)
                trace = np.clip(trace, -1.0, 3.0)  # 防止数值误差越界
                angle_rad = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
                angle_deg = np.degrees(angle_rad)

                info['ang_error'] = angle_deg
            else:
                info['dist_error'] = 0.0
                info['ang_error'] = 0.0

        except Exception as e:
            # print(f"[Debug] Calc Error: {e}")
            info['dist_error'] = -1.0
            info['ang_error'] = -1.0

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

    print(f"✅ 正在加载模型: {model_path}...")

    print("⏳ 正在初始化仿真环境...")
    dm_env = load_env()
    env = DMControlWrapper(dm_env)

    try:
        model = PPO.load(model_path, device='cpu')
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    obs, _ = env.reset()
    print("\n" + "=" * 60)
    print("🎮 演示开始！(位置 + 角度误差实时监控)")
    print("   按 'q' 键退出")
    print("=" * 60 + "\n")

    step_count = 0
    total_reward = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        # --- 控制台打印 ---
        if step_count % 10 == 0:
            dist_err = info.get('dist_error', 0.0)
            ang_err = info.get('ang_error', 0.0)

            status = "🟢"
            if info.get('collision'): status = "🔴 碰撞!"

            if dist_err == -1.0:
                d_str, a_str = "Error", "Error"
            else:
                d_str = f"{dist_err * 1000:5.2f}mm"
                a_str = f"{ang_err:5.2f}°"

            print(f"Step: {step_count:04d} | R: {reward:5.2f} | 距离误差: {d_str} | 角度误差: {a_str} | {status}")

        # --- 画面显示 ---
        if HAS_CV2:
            rgb_array = env.render()
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

            dist_err = info.get('dist_error', 0.0)
            ang_err = info.get('ang_error', 0.0)

            if dist_err != -1.0:
                # 位置误差 (绿色)
                cv2.putText(bgr_array, f"Pos Err: {dist_err * 1000:.2f} mm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # 角度误差 (黄色)
                cv2.putText(bgr_array, f"Ang Err: {ang_err:.1f} deg", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if info.get('collision'):
                cv2.putText(bgr_array, "COLLISION!", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Aubo RL Test", bgr_array)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        if terminated or truncated:
            print(f"\n🔔 回合结束! 总奖励: {total_reward:.2f}")
            obs, _ = env.reset()
            step_count = 0
            total_reward = 0
            time.sleep(1.0)

    if HAS_CV2:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()