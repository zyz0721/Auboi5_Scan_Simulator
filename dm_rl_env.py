import numpy as np
import collections
import os
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from scipy.spatial.transform import Rotation as R

print(">>> Safe-IK Environment (Comfort Zone Optimized) Loaded <<<")

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
MIN_HEIGHT_LIMIT = 0.2
COLLISION_PENALTY = -500.0
NAN_PENALTY = -1000.0
COLLISION_TERMINATE = True


class AuboScanTask(base.Task):
    def __init__(self, xml_path, random_state=None):
        super().__init__(random=random_state)
        self._random_state = self._random

        self.curve_manager = CurvePathPlanner()
        # 锁定 wrist3_Link
        self.ik_solver = Kinematics("wrist3_Link")
        self.ik_solver.buildFromMJCF(xml_path)

        self.action_scale = 0.005
        self._latency_buffer = collections.deque(maxlen=5)
        self._current_base_target = None

        self.collision_count = 0
        self.ik_fail_count = 0

    def _pose_to_matrix(self, pose_6d):
        t = pose_6d[:3]
        r_euler = pose_6d[3:]
        mat = np.eye(4)
        mat[:3, 3] = t
        r = R.from_euler('xyz', r_euler, degrees=False)
        mat[:3, :3] = r.as_matrix()
        return mat

    def _safe_ik(self, target_matrix, q_guess):
        if np.isnan(target_matrix).any() or np.isnan(q_guess).any():
            return None, "Input contains NaN"

        try:
            result = self.ik_solver.ik(target_matrix, q_guess)
            if isinstance(result, tuple):
                q_sol = result[0]
                info = result[1] if len(result) > 1 else "success"
            else:
                q_sol = result
                info = "success"
            return q_sol, info
        except Exception as e:
            self.ik_fail_count += 1
            return None, str(e)

    def _check_collision_and_height(self, physics):
        if np.isnan(physics.data.qpos).any() or np.isnan(physics.data.qvel).any():
            return False, "physics_nan"

        try:
            if 'wrist3_Link' in physics.named.data.xpos.axes.row.names:
                z_val = physics.named.data.xpos['wrist3_Link'][2]
            else:
                z_val = physics.data.xpos[-1][2]

            if z_val < MIN_HEIGHT_LIMIT:
                return False, "height_violation"
        except:
            pass

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

    def _is_joint_in_comfort_zone(self, qpos):
        """
        【新增】检查关节角是否在舒适区内。
        Aubo i5 的极限通常是 +/- 3.04 (约174度)。
        如果某个关节到了 2.9，说明已经很极限了，容易出问题。
        """
        # 设置一个稍微保守的阈值 (比如极限的 90%)
        limit_threshold = 2.8  # 约 160度

        if np.any(np.abs(qpos) > limit_threshold):
            return False  # 关节太极限了，不要这个点
        return True

    def initialize_episode(self, physics):
        self._randomize_physics(physics)
        self.collision_count = 0

        valid_path = []
        for _ in range(10):  # 多尝试几次，确保找到好路径
            raw_path = self._generate_mock_path_pattern()
            valid_path = self._filter_valid_path(physics, raw_path)
            if len(valid_path) > 20:
                break

        if len(valid_path) == 0:
            default_pose = np.concatenate([[0.4, -0.2, 0.5], [3.14, 0, 0]])
            valid_path = [default_pose] * 50

        self.path_points = valid_path
        self.path_index = 0

        start_pose_6d = self.path_points[0]
        start_matrix = self._pose_to_matrix(start_pose_6d)

        q_guess = [0, 0, 0, 0, 0, 0]
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
        last_q = [0, 0, 0, 0, 0, 0]

        for pose_6d in raw_path:
            mat = self._pose_to_matrix(pose_6d)
            q_sol, _ = self._safe_ik(mat, last_q)

            if q_sol is not None:
                # 【新增】除了检查碰撞，还要检查关节是否舒适
                if not self._is_joint_in_comfort_zone(q_sol):
                    # 如果关节太极限，视为无效点，直接跳过
                    # (这会截断路径，保留之前有效的部分)
                    break

                physics.data.qpos[:] = q_sol
                physics.forward()
                is_valid, reason = self._check_collision_and_height(physics)

                if is_valid:
                    valid_points.append(pose_6d)
                    last_q = q_sol
                else:
                    break  # 遇到障碍也截断
            else:
                break  # IK无解也截断

        physics.data.qpos[:] = qpos_backup
        physics.forward()
        return valid_points

    def before_step(self, action, physics):
        if np.isnan(physics.data.qpos).any():
            return

        if self.path_index < len(self.path_points):
            base_pose_target_6d = self.path_points[self.path_index]
            self.path_index += 1
        else:
            base_pose_target_6d = self.path_points[-1]

        self._current_base_target = base_pose_target_6d

        q_guess = physics.data.qpos[:].copy()

        base_matrix = self._pose_to_matrix(base_pose_target_6d)
        q_base, _ = self._safe_ik(base_matrix, q_guess)

        if q_base is None: q_base = q_guess

        residual_pos = action[:3] * self.action_scale
        target_pose_with_residual = base_pose_target_6d.copy()
        target_pose_with_residual[:3] += residual_pos

        target_matrix = self._pose_to_matrix(target_pose_with_residual)
        q_target, _ = self._safe_ik(target_matrix, q_base)

        if q_target is None: q_target = q_base

        self._latency_buffer.append(q_target)
        delay_steps = self._random_state.randint(1, self._latency_buffer.maxlen)
        idx = max(0, len(self._latency_buffer) - 1 - delay_steps)
        delayed_q_target = self._latency_buffer[idx]

        physics.set_control(delayed_q_target)

    def get_reward(self, physics):
        is_safe, reason = self._check_collision_and_height(physics)

        if reason == "physics_nan": return NAN_PENALTY
        if not is_safe: return COLLISION_PENALTY

        try:
            if 'wrist3_Link' in physics.named.data.xpos.axes.row.names:
                ee_pos = physics.named.data.xpos['wrist3_Link']
                ee_mat = physics.named.data.xmat['wrist3_Link'].reshape(3, 3)
            else:
                ee_pos = physics.data.xpos[-1]
                ee_mat = physics.data.xmat[-1].reshape(3, 3)
        except:
            ee_pos = physics.data.xpos[-1]
            ee_mat = physics.data.xmat[-1].reshape(3, 3)

        target_pos = self._current_base_target[:3]
        target_euler = self._current_base_target[3:]
        target_rot = R.from_euler('xyz', target_euler, degrees=False)
        target_mat = target_rot.as_matrix()

        dist = np.linalg.norm(target_pos - ee_pos)

        r_diff = np.dot(ee_mat, target_mat.T)
        trace = np.trace(r_diff)
        trace = np.clip(trace, -1.0, 3.0)
        angle_error = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))

        reward_pos = 2.0 * np.exp(-2000 * dist)
        reward_rot = 2.0 * np.exp(-100 * angle_error)

        total_reward = reward_pos * reward_rot
        return total_reward

    def get_termination(self, physics):
        is_safe, reason = self._check_collision_and_height(physics)

        if reason == "physics_nan" or (not is_safe and COLLISION_TERMINATE):
            return 1.0

        if self.path_index >= len(self.path_points) + 5:
            return 0.0

        return None

    def get_observation(self, physics):
        obs = collections.OrderedDict()

        qpos = physics.data.qpos[:].astype(np.float32)
        qvel = physics.data.qvel[:].astype(np.float32)

        if np.isnan(qpos).any(): qpos[:] = 0.0
        if np.isnan(qvel).any(): qvel[:] = 0.0

        obs['qpos'] = qpos
        obs['qvel'] = qvel

        try:
            if 'wrist3_Link' in physics.named.data.xpos.axes.row.names:
                ee_pos = physics.named.data.xpos['wrist3_Link']
                ee_mat = physics.named.data.xmat['wrist3_Link'].reshape(3, 3)
            else:
                ee_pos = physics.data.xpos[-1]
                ee_mat = np.eye(3)
        except:
            ee_pos = np.zeros(3)
            ee_mat = np.eye(3)

        if np.isnan(ee_pos).any(): ee_pos[:] = 0.0
        obs['ee_pos'] = ee_pos.astype(np.float32)

        if self._current_base_target is not None:
            target_pos = self._current_base_target[:3]
            pos_error = (target_pos - ee_pos).astype(np.float32)

            target_euler = self._current_base_target[3:]
            current_rot = R.from_matrix(ee_mat)
            current_euler = current_rot.as_euler('xyz', degrees=False)

            rot_error = target_euler - current_euler
            rot_error = (rot_error + np.pi) % (2 * np.pi) - np.pi

            obs['tracking_error'] = np.concatenate([pos_error, rot_error]).astype(np.float32)
        else:
            obs['tracking_error'] = np.zeros(6, dtype=np.float32)

        return obs

    def _randomize_physics(self, physics):
        if hasattr(physics.model, 'dof_damping'):
            dof = physics.model.nv
            physics.model.dof_damping[:] *= self._random_state.uniform(0.9, 1.1, dof)

    def _generate_mock_path_pattern(self):
        points = []

        # 【优化】缩小随机生成范围，避免生成到工作空间边缘
        # 之前的 radius 最大 0.2，center_x 最大 0.6 => 最远 0.8 (接近极限)
        # 现在稍微收缩一点
        start_z = self._random_state.uniform(0.2, 0.5)  # 提高最低高度
        center_x = self._random_state.uniform(0.3, 0.55)  # 稍微减小最大 X
        center_y = self._random_state.uniform(-0.25, 0.25)
        radius = self._random_state.uniform(0.1, 0.15)  # 稍微减小半径

        num_points = 200

        for theta in np.linspace(0, 2 * np.pi, num_points):
            x = center_x + radius * np.cos(theta)
            y = center_y + radius * np.sin(theta)
            # 让Z轴波动也平缓一点
            z = start_z + 0.03 * np.sin(theta * 3)
            rot = [3.14, 0, 0]
            points.append(np.concatenate([[x, y, z], rot]))
        return points


def load_env():
    xml_path = "aubo_i5_withdetector.xml"
    if not os.path.exists(xml_path):
        xml_path = os.path.join("mjcf", "aubo_i5_withdetector.xml")
    if not os.path.exists(xml_path):
        xml_path = "aubo_i5_withdetector.xml"

    task = AuboScanTask(xml_path)

    env = control.Environment(
        physics=mujoco.Physics.from_xml_path(xml_path),
        task=task,
        time_limit=15.0,
        control_timestep=CONTROL_TIMESTEP
    )
    return env