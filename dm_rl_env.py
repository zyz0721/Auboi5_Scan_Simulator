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
    from casadi_ik import Kinematics
except ImportError:
    print("警告: 未找到 curve_utils 或 casadi_ik，请确保文件在目录下。")
    pass

# 常量定义
CONTROL_TIMESTEP = 0.02  # 50Hz
PHYSICS_TIMESTEP = 0.002  # 500Hz

# --- 配置参数 ---
MIN_HEIGHT_LIMIT = 0.05  # 最小高度限制 (米)
COLLISION_PENALTY = -10.0  # 发生碰撞时的惩罚分数
NAN_PENALTY = -50.0  # 发生数值爆炸(NaN)时的严厉惩罚
COLLISION_TERMINATE = True  # 发生碰撞是否立即结束回合


class AuboScanTask(base.Task):
    """
    智能避障版 Aubo 扫描任务环境 (Safe IK 增强版)
    """

    def __init__(self, xml_path, random_state=None):
        super().__init__(random=random_state)
        self._random_state = self._random

        self.curve_manager = CurvePathPlanner()

        # IK 求解器初始化
        self.ik_solver = Kinematics("wrist3_Link")
        self.ik_solver.buildFromMJCF(xml_path)

        self.action_scale = 0.005
        self._latency_buffer = collections.deque(maxlen=5)
        self._current_base_target = None

        # 统计变量
        self.collision_count = 0
        self.ik_fail_count = 0

    def _pose_to_matrix(self, pose_6d):
        """[x,y,z,rx,ry,rz] -> 4x4 Matrix"""
        t = pose_6d[:3]
        r_euler = pose_6d[3:]
        mat = np.eye(4)
        mat[:3, 3] = t
        r = R.from_euler('xyz', r_euler, degrees=False)
        mat[:3, :3] = r.as_matrix()
        return mat

    def _safe_ik(self, target_matrix, q_guess):
        """
        【核心修复】带异常捕获的 IK 调用。
        无论 CasADi 报什么错 (NaN, Singularity)，这里都会接住。
        """
        # 1. 输入检查: 如果输入本身就是 NaN，直接拒绝
        if np.isnan(target_matrix).any() or np.isnan(q_guess).any():
            return None, "Input contains NaN"

        try:
            # 2. 尝试求解
            # 这里的 try...except 是防止进程退出的最后一道防线
            result = self.ik_solver.ik(target_matrix, q_guess)

            # 适配不同的返回格式
            if isinstance(result, tuple):
                q_sol = result[0]
                info = result[1] if len(result) > 1 else "success"
            else:
                q_sol = result
                info = "success"

            return q_sol, info

        except Exception as e:
            # 3. 捕获任何错误，绝对不让程序崩溃
            self.ik_fail_count += 1
            # 只有在非常频繁报错时才打印，避免刷屏
            if self.ik_fail_count % 100 == 0:
                print(f"[Warning] IK Failed {self.ik_fail_count} times. Ignored.")
            return None, str(e)

    def _check_collision_and_height(self, physics):
        """
        检测当前物理状态是否合法。
        """
        # 0. NaN 检查 (物理引擎是否炸了)
        if np.isnan(physics.data.qpos).any() or np.isnan(physics.data.qvel).any():
            return False, "physics_nan"

        # 1. 高度检测
        try:
            if 'wrist3_Link' in physics.named.data.site_xpos:
                z_val = physics.named.data.site_xpos['wrist3_Link'][2]
            else:
                z_val = physics.data.xpos[-1][2]

            if z_val < MIN_HEIGHT_LIMIT:
                return False, "height_violation"
        except:
            pass

            # 2. 物理碰撞检测
        try:
            for i in range(physics.data.ncon):
                contact = physics.data.contact[i]
                g1, g2 = contact.geom1, contact.geom2
                b1 = physics.model.geom_bodyid[g1]
                b2 = physics.model.geom_bodyid[g2]

                if b1 == 0 or b2 == 0: continue
                return False, "physical_collision"
        except Exception:
            return False, "collision_check_error"

        return True, "valid"

    def initialize_episode(self, physics):
        self._randomize_physics(physics)
        self.collision_count = 0
        self.ik_fail_count = 0

        # 生成无碰撞路径
        valid_path = []
        for _ in range(5):
            raw_path = self._generate_mock_path_pattern()
            valid_path = self._filter_valid_path(physics, raw_path)
            if len(valid_path) > 10:
                break

        if len(valid_path) == 0:
            # 兜底：如果生成失败，使用一个极其安全的姿态
            default_pose = np.concatenate([[0.4, -0.2, 0.5], [3.14, 0, 0]])
            valid_path = [default_pose] * 50

        self.path_points = valid_path
        self.path_index = 0

        start_pose_6d = self.path_points[0]
        start_matrix = self._pose_to_matrix(start_pose_6d)

        # 初始 IK (使用 Safe IK)
        q_guess = [0, -0.2, -1.5, 0.3, 1.57, 0]
        q_start, _ = self._safe_ik(start_matrix, q_guess)

        if q_start is None:
            q_start = q_guess

        physics.data.qpos[:] = q_start
        physics.data.qvel[:] = 0
        physics.forward()

        self._latency_buffer.clear()
        for _ in range(self._latency_buffer.maxlen):
            self._latency_buffer.append(q_start)

    def _filter_valid_path(self, physics, raw_path):
        valid_points = []
        qpos_backup = physics.data.qpos.copy()
        last_q = [0, -0.2, -1.5, 0.3, 1.57, 0]

        for pose_6d in raw_path:
            mat = self._pose_to_matrix(pose_6d)
            # 使用 Safe IK
            q_sol, _ = self._safe_ik(mat, last_q)

            if q_sol is not None:
                physics.data.qpos[:] = q_sol
                physics.forward()
                is_valid, reason = self._check_collision_and_height(physics)
                if is_valid:
                    valid_points.append(pose_6d)
                    last_q = q_sol
            else:
                pass  # IK 失败直接跳过

        physics.data.qpos[:] = qpos_backup
        physics.forward()
        return valid_points

    def before_step(self, action, physics):
        # 检查是否因为 NaN 导致不需要计算了
        if np.isnan(physics.data.qpos).any():
            return

        if self.path_index < len(self.path_points):
            base_pose_target_6d = self.path_points[self.path_index]
            self.path_index += 1
        else:
            base_pose_target_6d = self.path_points[-1]

        self._current_base_target = base_pose_target_6d

        q_guess = physics.data.qpos[:].copy()

        # --- IK 阶段 1: 基准姿态 ---
        base_matrix = self._pose_to_matrix(base_pose_target_6d)

        # 【关键】替换原来的 ik() 调用为 _safe_ik()
        q_base, _ = self._safe_ik(base_matrix, q_guess)

        if q_base is None:
            q_base = q_guess
            # 仅在需要时增加计数，不要让训练中断
            self.ik_fail_count += 1

        # --- IK 阶段 2: 叠加 RL 动作 ---
        residual_pos = action[:3] * self.action_scale
        target_pose_with_residual = base_pose_target_6d.copy()
        target_pose_with_residual[:3] += residual_pos

        target_matrix = self._pose_to_matrix(target_pose_with_residual)

        # 【关键】替换原来的 ik() 调用为 _safe_ik()
        q_target, _ = self._safe_ik(target_matrix, q_base)

        if q_target is None:
            q_target = q_base

        self._latency_buffer.append(q_target)
        delay_steps = self._random_state.randint(1, self._latency_buffer.maxlen)
        idx = max(0, len(self._latency_buffer) - 1 - delay_steps)
        delayed_q_target = self._latency_buffer[idx]

        physics.set_control(delayed_q_target)

    def get_reward(self, physics):
        is_safe, reason = self._check_collision_and_height(physics)

        if reason == "physics_nan":
            return NAN_PENALTY

        if not is_safe:
            self.collision_count += 1
            return COLLISION_PENALTY

        try:
            if 'wrist3_Link' in physics.named.data.site_xpos:
                ee_pos = physics.named.data.site_xpos['wrist3_Link']
            else:
                ee_pos = physics.data.xpos[-1]
        except:
            ee_pos = physics.data.xpos[-1]

        target_pos = self._current_base_target[:3]
        dist = np.linalg.norm(target_pos - ee_pos)

        reward_dist = np.exp(-10 * dist)
        reward_alive = 0.1

        return reward_dist + reward_alive

    def get_termination(self, physics):
        is_safe, reason = self._check_collision_and_height(physics)

        if reason == "physics_nan":
            return 1.0  # Failure

        if not is_safe and COLLISION_TERMINATE:
            return 1.0  # Failure

        if self.path_index >= len(self.path_points) + 5:
            return 0.0  # Success

        return None

    def get_observation(self, physics):
        obs = collections.OrderedDict()

        qpos = physics.data.qpos[:].astype(np.float32)
        qvel = physics.data.qvel[:].astype(np.float32)

        # 【防御】防止 NaN 进入神经网络
        if np.isnan(qpos).any(): qpos[:] = 0.0
        if np.isnan(qvel).any(): qvel[:] = 0.0

        obs['qpos'] = qpos
        obs['qvel'] = qvel

        try:
            if 'wrist3_Link' in physics.named.data.site_xpos:
                ee_pos = physics.named.data.site_xpos['wrist3_Link']
            else:
                ee_pos = physics.data.xpos[-1]
        except:
            ee_pos = np.zeros(3)

        if np.isnan(ee_pos).any(): ee_pos[:] = 0.0
        obs['ee_pos'] = ee_pos.astype(np.float32)

        if self._current_base_target is not None:
            target_pos = self._current_base_target[:3]
            obs['tracking_error'] = (target_pos - ee_pos).astype(np.float32)
        else:
            obs['tracking_error'] = np.zeros(3, dtype=np.float32)

        return obs

    def _randomize_physics(self, physics):
        if hasattr(physics.model, 'dof_damping'):
            dof = physics.model.nv
            physics.model.dof_damping[:] *= self._random_state.uniform(0.9, 1.1, dof)

    def _generate_mock_path_pattern(self):
        points = []
        start_z = self._random_state.uniform(0.0, 0.6)
        center_x = self._random_state.uniform(0.3, 0.6)
        center_y = self._random_state.uniform(-0.3, 0.3)
        radius = 0.15
        for theta in np.linspace(0, 2 * np.pi, 60):
            x = center_x + radius * np.cos(theta)
            y = center_y + radius * np.sin(theta)
            z = start_z + 0.05 * np.sin(theta * 2)
            rot = [3.14, 0, 0]
            points.append(np.concatenate([[x, y, z], rot]))
        return points


def load_env():
    xml_path = "mjcf/aubo_i5_withdetector.xml"
    if not os.path.exists(xml_path):
        xml_path = os.path.join("mjcf", "aubo_i5_withdetector.xml")
    if not os.path.exists(xml_path):
        xml_path = "mjcf/aubo_i5_withdetector.xml"

    task = AuboScanTask(xml_path)

    env = control.Environment(
        physics=mujoco.Physics.from_xml_path(xml_path),
        task=task,
        time_limit=15.0,
        control_timestep=CONTROL_TIMESTEP
    )
    return env