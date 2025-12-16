import numpy as np
import collections
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

# 引入你的自定义工具
try:
    from curve_utils import CurveManager
    from casadi_ik import CasadiIK
except ImportError:
    raise ImportError("请确保 curve_utils.py 和 casadi_ik.py 在当前目录下")

# 常量定义
CONTROL_TIMESTEP = 0.02  # 50Hz，与实机控制频率保持一致
PHYSICS_TIMESTEP = 0.002  # 500Hz 物理仿真频率


class AuboScanTask(base.Task):
    """
    针对 Aubo i5 随形扫描任务的自定义 RL 任务。
    实现了 '残差控制' (Residual Control) 逻辑。
    """

    def __init__(self, xml_path, random_state=None):
        self._xml_path = xml_path
        self._random_state = random_state or np.random.RandomState()

        # 初始化基准控制器组件
        self.curve_manager = CurveManager()
        self.ik_solver = CasadiIK()

        # 动作空间缩放因子 (Residual Scale)
        # RL 输出 [-1, 1]，映射为 [-5mm, +5mm] 的修正量
        self.action_scale = 0.005

        # 延迟模拟队列 (Sim-to-Real Gap)
        # 模拟 Windows 通信延迟，队列长度随机化
        self._latency_buffer = collections.deque(maxlen=5)

        # 缓存当前步的基准目标，用于计算 Observation
        self._current_base_target = None

    def initialize_episode(self, physics):
        """每个 Episode 开始时的初始化"""
        # 1. 域随机化 (Domain Randomization)
        # 随机修改摩擦力、阻尼、甚至连杆质量，增强鲁棒性
        self._randomize_physics(physics)

        # 2. 生成新的扫描路径
        # 这里可以加入随机性：不同的扫描形状、不同的起始点
        # 假设 generate_path 返回路径点列表
        # 简单起见，这里重新初始化 curve_manager 并生成一段直线或正弦路径
        # 实际使用中，应调用 self.curve_manager.generate_random_curve()
        self.path_points = self._generate_mock_path()
        self.path_index = 0

        # 3. 机器人复位到路径起点
        start_pose = self.path_points[0]
        q_start = self.ik_solver.calculate(start_pose[:3], start_pose[3:])

        if q_start is None:
            # 如果起点无解，强制复位到一个安全姿态
            q_start = [0, -0.2, -1.5, 0.3, 1.57, 0]

        physics.data.qpos[:] = q_start
        physics.data.qvel[:] = 0
        physics.forward()

        # 4. 重置延迟队列
        self._latency_buffer.clear()
        # 填充初始动作
        for _ in range(self._latency_buffer.maxlen):
            self._latency_buffer.append(q_start)

    def before_step(self, action, physics):
        """
        核心残差逻辑：
        Final_Action = IK(Base_Path) + RL_Action
        """
        # 1. 获取当前的基准路径点 (Base Target)
        if self.path_index < len(self.path_points):
            base_pose_target = self.path_points[self.path_index]
            self.path_index += 1
        else:
            base_pose_target = self.path_points[-1]  # 保持在终点

        self._current_base_target = base_pose_target

        # 2. 计算基准关节角 (Base Action)
        # 使用上一帧的关节角作为 IK 的 guess，提高速度
        q_guess = physics.data.qpos[:]
        q_base = self.ik_solver.calculate(
            base_pose_target[:3],
            base_pose_target[3:],
            q_guess=q_guess
        )

        if q_base is None:
            q_base = q_guess  # IK 失败则保持不动

        # 3. 解析 RL 动作 (Residual Action)
        # 假设 RL 输出的是笛卡尔空间的微调 (dx, dy, dz)
        # 这样比直接输出关节角残差更容易学习接触任务
        # 注意：这里简化处理，假设 action 是关节角残差，如果是笛卡尔残差需先叠加 pose 再解 IK

        # 方案 A: RL 输出关节角残差 (简单直接)
        # residual = action * 0.05 # 缩放幅度
        # q_target = q_base + residual

        # 方案 B: RL 输出末端位置残差 (更适合扫描任务)
        # action 维度应为 3 (x, y, z)
        residual_pos = action[:3] * self.action_scale
        target_pos_with_residual = base_pose_target[:3] + residual_pos

        q_target = self.ik_solver.calculate(
            target_pos_with_residual,
            base_pose_target[3:],  # 保持姿态不变
            q_guess=q_base
        )

        if q_target is None:
            q_target = q_base

        # 4. 延迟模拟 (Latency Simulation)
        # 将计算出的目标放入队列，取出旧的目标执行
        self._latency_buffer.append(q_target)

        # 随机选择延迟帧数 (模拟 Windows 网络抖动)
        # 比如有时延迟 1 帧，有时延迟 3 帧
        delay_steps = self._random_state.randint(1, self._latency_buffer.maxlen)
        # 确保索引有效
        idx = max(0, len(self._latency_buffer) - 1 - delay_steps)
        delayed_q_target = self._latency_buffer[idx]

        # 5. 下发控制指令
        # 假设 xml 中定义的是位置伺服 (position actuator)
        physics.set_control(delayed_q_target)

    def get_observation(self, physics):
        """构造观测空间"""
        obs = collections.OrderedDict()

        # 1. 关节状态
        obs['qpos'] = physics.data.qpos[:].copy()
        obs['qvel'] = physics.data.qvel[:].copy()

        # 2. 末端执行器状态 (通过 site 获取)
        # 假设 xml 中末端有一个名为 'ee_site' 的 site
        ee_pos = physics.named.data.site_xpos['detector_site']  # 需确保 XML 里有这个 site
        ee_mat = physics.named.data.site_xmat['detector_site']
        obs['ee_pos'] = ee_pos.astype(np.float32)

        # 3. 追踪误差 (Base Target - Current EE)
        if self._current_base_target is not None:
            target_pos = self._current_base_target[:3]
            obs['tracking_error'] = (target_pos - ee_pos).astype(np.float32)
        else:
            obs['tracking_error'] = np.zeros(3, dtype=np.float32)

        # 4. 接触力/传感器读数 (如果有力传感器)
        # obs['sensors'] = ...

        return obs

    def get_reward(self, physics):
        """
        奖励函数设计：R = R_track + R_stable + R_contact
        """
        # 1. 追踪奖励：距离路径越近越好
        ee_pos = physics.named.data.site_xpos['detector_site']
        target_pos = self._current_base_target[:3]
        dist = np.linalg.norm(target_pos - ee_pos)
        r_track = np.exp(-100 * dist * dist)  # 高斯核奖励

        # 2. 稳定性奖励：惩罚过大的关节速度
        qvel = physics.data.qvel[:]
        r_stable = -0.01 * np.linalg.norm(qvel)

        # 3. 扫描距离保持奖励 (假设理想扫描距离是 0，即贴合)
        # 这里用 z 轴高度举例，如果是曲面需计算法向距离
        # r_contact = np.exp(-50 * abs(dist_to_surface - ideal_dist))

        return r_track + r_stable

    def get_termination(self, physics):
        """终止条件"""
        # 如果关节角发散或发生严重碰撞，提前结束
        if self.path_index >= len(self.path_points) + 10:
            return 0.0  # 任务结束
        return None

    def _randomize_physics(self, physics):
        """Sim-to-Real 关键：域随机化"""
        # 随机化关节阻尼
        dof = physics.model.nv
        physics.model.dof_damping[:] *= self._random_state.uniform(0.8, 1.2, dof)

        # 随机化连杆质量 (模拟线缆重量等)
        num_bodies = physics.model.nbody
        physics.model.body_mass[:] *= self._random_state.uniform(0.9, 1.1, num_bodies)

        # 随机化摩擦 (如果定义了 pair 或 geom friction)
        # physics.model.geom_friction[:] *= ...

    def _generate_mock_path(self):
        """生成测试用的扫描路径数据"""
        # 实际应调用 CurveManager
        points = []
        center = np.array([0.5, 0.0, 0.3])
        for t in np.linspace(0, 2, 200):  # 2秒，200个点
            # 简单的直线运动
            pos = center + np.array([0, t * 0.1, 0])
            # 姿态：垂直向下
            rot = np.array([3.14, 0, 0])
            points.append(np.concatenate([pos, rot]))
        return points


def load_env():
    """环境加载入口函数"""
    # 加载你的 XML 模型
    # 注意：确保 xml 文件路径正确，且 xml 中包含了 detector_site
    xml_path = "mjcf/aubo_i5_withdetector.xml"

    # 实例化 Task
    task = AuboScanTask(xml_path)

    # 使用 dm_control.rl.control 封装为标准 Environment
    env = control.Environment(
        physics=mujoco.Physics.from_xml_path(xml_path),
        task=task,
        time_limit=10.0,  # 每回合 10 秒
        control_timestep=CONTROL_TIMESTEP
    )
    return env