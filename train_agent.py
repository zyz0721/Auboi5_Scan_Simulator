import numpy as np
import os
import multiprocessing
from dm_rl_env import load_env

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed


class DMControlWrapper(gym.Env):
    """
    将 dm_control 环境包装为 gymnasium 环境
    """

    def __init__(self, env_loader_func):
        # 注意：这里传入加载函数，确保每个进程独立初始化
        self.env = env_loader_func()
        self.metadata = {'render.modes': ['rgb_array']}

        # 获取动作空间范围
        action_spec = self.env.action_spec()
        self.action_space = spaces.Box(
            low=action_spec.minimum,
            high=action_spec.maximum,
            dtype=np.float32
        )

        # 获取观测空间
        time_step = self.env.reset()
        obs = self._flatten_obs(time_step.observation)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32
        )

    def _flatten_obs(self, obs_dict):
        """将有序字典观测展平为向量"""
        return np.concatenate([v.ravel() for v in obs_dict.values()])

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
        return self.env.physics.render()


def make_env(rank, seed=0):
    """
    环境工厂函数：返回一个用于创建环境的函数 (Thunk)
    """

    def _init():
        # 在子进程中创建环境
        env = DMControlWrapper(load_env)
        # 设置不同的随机种子
        env.reset(seed=seed + rank)
        return env

    return _init


# ----------------------------------------------------------------
# 主训练流程
# ----------------------------------------------------------------
if __name__ == "__main__":
    # 【关键修复】强制使用 'spawn' 启动方式
    # MuJoCo/CasADi 在 fork 模式下容易死锁或崩溃
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # 1. 设置并行数量
    # 按照你的要求，限制为 16 个核心
    max_cores = 16
    physical_cores = multiprocessing.cpu_count()
    num_cpu = min(max_cores, physical_cores)

    print(f"物理核心数: {physical_cores}, 限制使用: {num_cpu}")
    print(f"正在启动 {num_cpu} 个并行环境进程 (Spawn模式)...")

    # 2. 创建并行环境
    # 【关键修复】不使用 make_vec_env，而是直接构造函数列表
    # 这样可以明确地传递 rank，避免 TypeError
    env_fns = [make_env(rank=i) for i in range(num_cpu)]

    # start_method='spawn' 确保子进程环境干净
    env = SubprocVecEnv(env_fns, start_method='spawn')

    # 启用 Monitor 记录日志
    env = VecMonitor(env)

    print("并行环境初始化完成。")

    # 3. 配置 PPO 算法
    device = "cpu"
    print(f"使用计算设备: {device}")

    # 计算 n_steps，确保 batch_size 合理
    # 每个进程采集 128 步，总共 num_cpu * 128 个样本用于一次更新
    steps_per_env = 128

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=steps_per_env,
        batch_size=256,
        gamma=0.99,
        device=device,
        ent_coef=0.01
    )

    # 4. 设置检查点保存
    # 注意：因为是并行采样，save_freq 是指“主进程循环次数”或者“总步数”取决于实现
    # 在 SB3 CheckpointCallback 中，频率是按总步数 (num_envs * steps) 计算的吗？
    # 实际上 SB3 的 callback 也是按 update 调用的。
    # 这里设置为每 10000 步保存一次
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // num_cpu, 1),
        save_path='./logs/',
        name_prefix='aubo_scan_rl'
    )

    # 5. 开始训练
    print("开始训练...")
    try:
        model.learn(total_timesteps=2000000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("训练被用户中断。正在保存当前模型...")

    model.save("aubo_scan_final_policy")
    print("模型已保存: aubo_scan_final_policy.zip")

    # 关闭环境进程
    env.close()

    # 6. 验证 (单核运行)
    print("开始演示验证 (使用单独的单核环境)...")
    eval_env = DMControlWrapper(load_env)
    obs, _ = eval_env.reset()
    for i in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = eval_env.step(action)
        # eval_env.render()
        if done:
            obs, _ = eval_env.reset()