import numpy as np
import os
import torch
import gymnasium as gym
from gymnasium import spaces

# SB3 核心组件
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# 引入环境加载函数
from dm_rl_env import load_env


class DMControlWrapper(gym.Env):
    """
    将 dm_control 环境包装为 gymnasium 环境。
    """

    def __init__(self, dm_env_instance=None):
        if dm_env_instance is None:
            self.env = load_env()
        else:
            self.env = dm_env_instance

        self.metadata = {'render.modes': ['rgb_array']}

        # 1. 动作空间转换 (float32)
        action_spec = self.env.action_spec()
        self.action_space = spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            dtype=np.float32
        )

        # 2. 观测空间转换
        obs_spec = self.env.observation_spec()
        dim = sum(np.prod(v.shape) for v in obs_spec.values())

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(dim,),
            dtype=np.float32
        )

    def _flatten_obs(self, obs_dict):
        """展平观测"""
        return np.concatenate([v.ravel() for v in obs_dict.values()]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        time_step = self.env.reset()
        return self._flatten_obs(time_step.observation), {}

    def step(self, action):
        time_step = self.env.step(action)
        obs = self._flatten_obs(time_step.observation)
        reward = time_step.reward or 0.0
        terminated = time_step.last()
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.physics.render(camera_id=-1, height=480, width=640)


def make_env_fn(rank, seed=0):
    """工厂函数"""

    def _init():
        env = DMControlWrapper()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


if __name__ == "__main__":

    # === 1. 并行核心数优化 ===
    # 不占满所有核心，留4个给系统和梯度计算，否则会卡顿
    max_cores = os.cpu_count() or 4
    num_cpu = max(1, max_cores - 4)

    print(f"物理核心: {max_cores}, 实际使用并行环境数: {num_cpu}")

    # === 2. 创建并行环境 ===
    env_fns = [make_env_fn(i, seed=42) for i in range(num_cpu)]

    # start_method='fork' 在 Linux 下更快，但 'spawn' 更稳定
    # 如果遇到奇怪的死锁，可以去掉 start_method 参数让其使用默认
    env = SubprocVecEnv(env_fns)

    print("并行环境初始化完成。")

    # === 3. PPO 参数深度优化 (针对并行提速) ===
    # 逻辑:
    # n_steps * num_cpu = 每次更新前的总采样数 (Buffer Size)
    # 之前是 2048 * 16 = 32768，太大了，导致更新间隔极长
    # 现在改为 256 * 14 ≈ 3584，更新频率会快很多

    n_steps = 256
    batch_size = 512  # 增大 batch size 利用矩阵计算加速

    # 如果有 GPU，务必使用 GPU 进行梯度更新 (Learner)，CPU 只负责采样 (Worker)
    # 如果没有 GPU，这行会自动回退到 CPU
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"训练设备: {device} (Learner)")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,

        # --- 速度优化参数 ---
        learning_rate=3e-4,
        n_steps=n_steps,  # 减小这个值，让 log 刷得快一点
        batch_size=batch_size,  # 增大这个值，加速网络更新
        n_epochs=10,  # 每次更新复用数据的次数

        # --- 稳定性参数 ---
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # 稍微增加探索熵，防止过早收敛

        device=device  # 尽量用 GPU 训练网络
    )

    # === 4. 检查点 ===
    checkpoint_callback = CheckpointCallback(
        save_freq=max(20000 // num_cpu, 1),
        save_path='./logs/',
        name_prefix='aubo_scan_rl'
    )

    print(f"开始训练... 每轮收集数据量: {n_steps * num_cpu}")
    try:
        model.learn(total_timesteps=3000000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("训练被用户中断。正在保存当前模型...")
    finally:
        env.close()

    model.save("aubo_scan_final_policy")
    print("模型已保存: aubo_scan_final_policy.zip")
