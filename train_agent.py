import numpy as np
import os
from dm_rl_env import load_env

# 我们使用 shimmy 或手动封装将 dm_control 转换为 gymnasium 兼容环境
# 为了不引入额外依赖，这里写一个简单的 Wrapper
import gymnasium as gym
from gymnasium import spaces


class DMControlWrapper(gym.Env):
    """
    将 dm_control 环境包装为 gymnasium 环境，
    以便使用 Stable Baselines3 等库进行训练。
    """

    def __init__(self, dm_env):
        self.env = dm_env
        self.metadata = {'render.modes': ['rgb_array']}

        # 获取动作空间范围
        action_spec = self.env.action_spec()
        self.action_space = spaces.Box(
            low=action_spec.minimum,
            high=action_spec.maximum,
            dtype=np.float32
        )

        # 获取观测空间 (简化处理：将 dict obs 展平)
        # 第一次 reset 获取 obs 形状
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
        # dm_control 的 reset 不需要 seed，随机性由 task 内部管理
        time_step = self.env.reset()
        return self._flatten_obs(time_step.observation), {}

    def step(self, action):
        time_step = self.env.step(action)
        obs = self._flatten_obs(time_step.observation)
        reward = time_step.reward or 0.0
        terminated = time_step.last()
        truncated = False  # dm_control 通常只处理 terminated
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        # 简单的渲染接口
        return self.env.physics.render()


# ----------------------------------------------------------------
# 主训练流程
# ----------------------------------------------------------------
if __name__ == "__main__":
    # 1. 检查是否安装了必要的库
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
    except ImportError:
        print("请安装 stable-baselines3: pip install stable-baselines3[extra]")
        exit()

    print("正在初始化 DeepMind MuJoCo 环境...")
    # 加载原生 dm_control 环境
    dm_env = load_env()

    # 包装为 Gym 环境
    env = DMControlWrapper(dm_env)

    print("环境初始化完成。")
    print(f"动作空间: {env.action_space}")
    print(f"观测空间: {env.observation_space}")

    # 2. 配置 PPO 算法
    # 对于 Sim-to-Real，较小的学习率和较大的 batch size 通常更稳定
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        device="cuda" if np.os.system("nvidia-smi") == 0 else "cpu"
    )

    # 3. 设置检查点保存
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./logs/',
        name_prefix='aubo_scan_rl'
    )

    # 4. 开始训练
    print("开始训练...")
    try:
        model.learn(total_timesteps=1000000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("训练被用户中断。正在保存当前模型...")

    model.save("aubo_scan_final_policy")
    print("模型已保存: aubo_scan_final_policy.zip")

    # 5. 验证 (可选)
    print("开始演示验证...")
    obs, _ = env.reset()
    for i in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        # 注意：在服务器或无头模式下可能无法 render
        # env.render()
        if done:
            obs, _ = env.reset()