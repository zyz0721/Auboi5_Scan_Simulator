import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces

# SB3 核心组件
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# 引入环境加载函数
from dm_rl_env import load_env


class DMControlWrapper(gym.Env):
    """
    将 dm_control 环境包装为 gymnasium 环境。
    支持传入实例，也支持内部自动加载。
    """

    def __init__(self, dm_env_instance=None):
        # 如果没有传入实例，则现场创建一个 (用于多进程生成)
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
        # 计算展平后的维度
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
        # dm_control 内部管理随机性，这里重置环境
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
        return self.env.physics.render(camera_id=0, height=480, width=640)


def make_env_fn(rank, seed=0):
    """
    工厂函数：返回一个'创建环境的函数' (_init)，而不是直接返回环境。
    这是 SubprocVecEnv 要求的格式。
    """

    def _init():
        # 1. 在子进程中创建全新的环境实例 (避免 Pickle 问题)
        env = DMControlWrapper()

        # 2. 添加 Monitor 包装器 (关键！记录 Reward)
        env = Monitor(env)

        # 3. 设置随机种子 (确保每个进程的随机性不同)
        env.reset(seed=seed + rank)
        return env

    return _init


# ----------------------------------------------------------------
# 主训练流程
# ----------------------------------------------------------------
if __name__ == "__main__":

    # === 并行配置 ===
    # 自动获取 CPU 核心数，或者手动指定 (推荐 4-8 核，太多可能会爆内存)
    num_cpu = os.cpu_count() - 8
    print(f"检测到 {num_cpu} 个 CPU 核心，准备启动并行训练...")

    # === 创建并行环境 (Vectorized Environment) ===
    # 【修复点】不再使用 make_vec_env，而是直接构造函数列表传给 SubprocVecEnv
    # 这样我们可以显式地将 rank (i) 传进去
    env_fns = [make_env_fn(i, seed=42) for i in range(num_cpu)]

    # SubprocVecEnv 会为列表中的每个函数启动一个进程
    env = SubprocVecEnv(env_fns)

    print("并行环境初始化完成。")

    # === 配置 PPO 算法 ===
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,  # 每个环境采样多少步才更新
        batch_size=256,  # 增大 batch_size
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device="cpu"  # 强制 CPU，避免 GPU 显存溢出和 CUDA 初始化冲突
    )

    # === 设置检查点保存 ===
    checkpoint_callback = CheckpointCallback(
        save_freq=max(20000 // num_cpu, 1),  # 根据并行数调整保存频率
        save_path='./logs/',
        name_prefix='aubo_scan_rl'
    )

    print(f"开始在 {num_cpu} 个环境中并行训练...")
    try:
        # total_timesteps 是所有环境步数的总和
        model.learn(total_timesteps=2000000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("训练被用户中断。正在保存当前模型...")
    finally:
        # 关闭所有子进程，释放资源
        env.close()

    model.save("aubo_scan_final_policy")
    print("模型已保存: aubo_scan_final_policy.zip")