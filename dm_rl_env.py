import numpy as np
import collections
import os
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from scipy.spatial.transform import Rotation as R

# 引入你的自定义工具
try:
    from curve_utils import CurvePathPlanner
    # 使用你原本的 Kinematics 类
    from casadi_ik import Kinematics
except ImportError:
    raise ImportError("请确保 curve_utils.py 和 casadi_ik.py 在当前目录下")

# 常量定义
CONTROL_TIMESTEP = 0.02  # 50Hz
PHYSICS_TIMESTEP = 0.002  # 500Hz


class AuboScanTask(base.Task):
    """
    针对 Aubo i5 随形扫描任务的自定义 RL 任务。
    适配用户原有的 casadi_ik.py 接口。
    """

    def __init__(self, xml_path, random_state=None):
        # -------------------------------------------------------------
        # 【修正3】必须调用父类初始化
        # base.Task 会初始化 _visualize_reward, _random 等关键属性
        # -------------------------------------------------------------
        super().__init__(random=random_state)

        # 保持对 self._random_state 的兼容 (base.Task 使用 self._random)
        self._random_state = self._random

        self.curve_manager = CurvePathPlanner()

        # -------------------------------------------------------------
        # 【修正1】初始化适配
        # -------------------------------------------------------------
        # 假设 Kinematics(model_path, ee_frame)
        self.ik_solver = Kinematics("wrist3_Link")

        self.ik_solver.buildFromMJCF(xml_path)

        self.action_scale = 0.005
        self._latency_buffer = collections.deque(maxlen=5)
        self._current_base_target = None

    def _pose_to_matrix(self, pose_6d):
        """
        辅助函数：将 [x, y, z, rx, ry, rz] 转换为 4x4 齐次变换矩阵
        """
        t = pose_6d[:3]
        r_euler = pose_6d[3:]

        # 创建 4x4 矩阵
        mat = np.eye(4)
        mat[:3, 3] = t

        # 欧拉角转旋转矩阵 (xyz 顺序)
        r = R.from_euler('xyz', r_euler, degrees=False)
        mat[:3, :3] = r.as_matrix()

        return mat

    def initialize_episode(self, physics):
        """每个 Episode 开始时的初始化"""
        self._randomize_physics(physics)

        self.path_points = self._generate_mock_path()
        self.path_index = 0

        # start_pose 是 [x, y, z, rx, ry, rz] (6维)
        start_pose_6d = self.path_points[0]

        # -------------------------------------------------------------
        # 【修正2】IK 调用适配 (6D -> 4x4 Matrix)
        # -------------------------------------------------------------
        start_matrix = self._pose_to_matrix(start_pose_6d)

        # 初始猜测
        q_guess = [0, -0.2, -1.5, 0.3, 1.57, 0]

        # 调用 ik, 解包返回值 (dof, info)
        q_start, _ = self.ik_solver.ik(start_matrix, q_guess)

        if q_start is None:
            q_start = [0, -0.2, -1.5, 0.3, 1.57, 0]

        physics.data.qpos[:] = q_start
        physics.data.qvel[:] = 0
        physics.forward()

        self._latency_buffer.clear()
        for _ in range(self._latency_buffer.maxlen):
            self._latency_buffer.append(q_start)

    def before_step(self, action, physics):
        """核心控制循环"""
        # 1. 获取当前目标点 [x, y, z, rx, ry, rz]
        if self.path_index < len(self.path_points):
            base_pose_target_6d = self.path_points[self.path_index]
            self.path_index += 1
        else:
            base_pose_target_6d = self.path_points[-1]

        self._current_base_target = base_pose_target_6d

        # 2. 获取当前关节角作为 IK 猜测
        q_guess = physics.data.qpos[:].copy()

        # 3. 计算名义 IK (Base Action)
        base_matrix = self._pose_to_matrix(base_pose_target_6d)
        q_base, _ = self.ik_solver.ik(base_matrix, q_guess)

        if q_base is None:
            q_base = q_guess

        # 4. 叠加 RL 动作 (残差)
        # 假设 RL 输出的是位置微调 [dx, dy, dz]
        residual_pos = action[:3] * self.action_scale

        # 构造新的带残差的目标 [x+dx, y+dy, z+dz, rx, ry, rz]
        target_pose_with_residual = base_pose_target_6d.copy()
        target_pose_with_residual[:3] += residual_pos

        # 5. 再次 IK 计算最终关节角
        target_matrix = self._pose_to_matrix(target_pose_with_residual)
        q_target, _ = self.ik_solver.ik(target_matrix, q_base)

        if q_target is None:
            q_target = q_base

        # 6. 延迟模拟
        self._latency_buffer.append(q_target)
        delay_steps = self._random_state.randint(1, self._latency_buffer.maxlen)
        idx = max(0, len(self._latency_buffer) - 1 - delay_steps)
        delayed_q_target = self._latency_buffer[idx]

        # 7. 下发控制
        physics.set_control(delayed_q_target)

    def get_observation(self, physics):
        """构造观测空间，强制转为 float32 消除 Gym Warning"""
        obs = collections.OrderedDict()

        obs['qpos'] = physics.data.qpos[:].astype(np.float32)
        obs['qvel'] = physics.data.qvel[:].astype(np.float32)

        # 尝试获取末端位置
        try:
            # 优先尝试 site
            ee_pos = physics.named.data.site_xpos['detector_site']
        except Exception:
            try:
                # 其次尝试 link
                ee_pos = physics.named.data.xpos['wrist3_Link']
            except:
                # 最后尝试最后一个 body
                ee_pos = physics.data.xpos[-1]

        obs['ee_pos'] = ee_pos.astype(np.float32)

        # 追踪误差
        if self._current_base_target is not None:
            target_pos = self._current_base_target[:3]
            obs['tracking_error'] = (target_pos - ee_pos).astype(np.float32)
        else:
            obs['tracking_error'] = np.zeros(3, dtype=np.float32)

        return obs

    def get_reward(self, physics):
        # 获取末端位置用于计算奖励
        try:
            ee_pos = physics.named.data.site_xpos['detector_site']
        except:
            ee_pos = physics.data.xpos[-1]

        target_pos = self._current_base_target[:3]
        dist = np.linalg.norm(target_pos - ee_pos)

        # 简单的距离奖励：越近分越高
        return np.exp(-50 * dist ** 2)

    def get_termination(self, physics):
        if self.path_index >= len(self.path_points) + 10:
            return 0.0
        return None

    def _randomize_physics(self, physics):
        # 简单的域随机化
        if hasattr(physics.model, 'dof_damping'):
            dof = physics.model.nv
            physics.model.dof_damping[:] *= self._random_state.uniform(0.8, 1.2, dof)

    def _generate_mock_path(self):
        """
        生成测试路径 [x, y, z, rx, ry, rz]
        实际应该调用 self.curve_manager.generate_path()
        """
        points = []
        start = np.array([0.4, -0.2, 0.4])
        # 假设末端垂直向下 (Rx=pi, Ry=0, Rz=0)
        # 注意：这取决于你的 IK 解算器定义的末端坐标系
        rot = np.array([3.14, 0, 0])

        for t in np.linspace(0, 1, 100):
            pos = start + np.array([0, t * 0.4, 0])
            # 将 pos 和 rot 拼接成 6维向量
            points.append(np.concatenate([pos, rot]))
        return points


def load_env():
    # 确保路径正确
    xml_path = "mjcf/aubo_i5_withdetector.xml"

    # 简单的路径容错处理
    if not os.path.exists(xml_path):
        xml_path = os.path.join(os.getcwd(), xml_path)

    task = AuboScanTask(xml_path)

    env = control.Environment(
        physics=mujoco.Physics.from_xml_path(xml_path),
        task=task,
        time_limit=10.0,
        control_timestep=CONTROL_TIMESTEP
    )
    return env