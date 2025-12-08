import mujoco
import mujoco.viewer
import numpy as np
import glfw
import time
from casadi_ik import Kinematics
import trimesh.transformations as tf  # 依然需要 trimesh 库来辅助做矩阵计算


class MuJoCoSimulator:
    """
    MuJoCo 仿真控制器 - 增加数据输出接口
    """

    def __init__(self, xml_path, end_joint, robot_model_path, cam_res=(640, 640)):
        # 降低分辨率以提高曲线图刷新性能
        self.xml_path = xml_path
        self.cam_res = cam_res
        self.end_joint = end_joint
        self.robot_model_path = robot_model_path

        self.paused = True
        self.running = True
        self.target_qpos = None

        self.physics_dt = 0.002
        self.control_dt = 0.02
        self.n_substeps = int(self.control_dt / self.physics_dt)
        self.scan_interval = 0.1  # 扫描点切换速度

        self.current_idx = 0
        self.last_scan_time = 0
        self.path_points = None
        self.path_normals = None
        self.scan_height = 0.1

        # 缓存位姿矩阵 (4x4)
        self.T_target_cache = np.eye(4)
        self.actual_pose_cache = np.eye(4)

        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.physics_dt
        self.target_qpos = self.data.qpos[:6].copy()

        # IK 求解器
        self.ik_solver = Kinematics(self.end_joint)
        self.ik_solver.buildFromMJCF(self.robot_model_path)

        # 【关键】：获取 XML 中定义的末端 Site ID
        # 这样我们就能直接读取末端的实际位置，而不需要额外加 sensor
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

        self.init_offscreen()
        # self.viewer = mujoco.viewer.launch_passive(self.model, self.data) # 可选：关闭独立窗口

    def init_offscreen(self):
        if not glfw.init(): return
        glfw.window_hint(glfw.VISIBLE, 0)
        self.offscreen_window = glfw.create_window(self.cam_res[0], self.cam_res[1], "Offscreen", None, None)
        glfw.make_context_current(self.offscreen_window)
        self.gl_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.viewport = mujoco.MjrRect(0, 0, self.cam_res[0], self.cam_res[1])
        self.rgb_buffer = np.zeros((self.cam_res[1], self.cam_res[0], 3), dtype=np.uint8)

        self.cam_global = mujoco.MjvCamera()
        self.cam_global.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam_global.lookat = [0.0, -0.6, 0.2]
        self.cam_global.distance = 1.5
        self.cam_global.azimuth = 180
        self.cam_global.elevation = -45

    def set_path(self, points, normals, height=0.1):
        self.path_points = points
        self.path_normals = normals
        self.scan_height = height
        self.current_idx = 0
        self.perform_ik_step()

    def perform_ik_step(self):
        if self.path_points is None: return
        target_pt = self.path_points[self.current_idx]
        target_nm = self.path_normals[self.current_idx]

        # 1. 计算目标位姿矩阵 T_target
        self.T_target_cache = self.compute_target_matrix(target_pt, target_nm, self.scan_height)

        init_q = self.data.qpos[:6].copy()
        try:
            # IK 计算
            q_sol, info = self.ik_solver.ik(self.T_target_cache, current_arm_motor_q=init_q)
            if info["success"]:
                self.target_qpos = np.clip(q_sol, self.model.jnt_range[:6, 0], self.model.jnt_range[:6, 1])
        except:
            pass

    def step(self):
        # 路径自动递增
        if not self.paused and self.path_points is not None:
            now = time.time()
            if now - self.last_scan_time > self.scan_interval:
                self.current_idx = (self.current_idx + 1) % len(self.path_points)
                self.last_scan_time = now

        self.perform_ik_step()

        # 物理步进
        for _ in range(self.n_substeps):
            if self.target_qpos is not None:
                self.data.ctrl[:6] = self.target_qpos
            mujoco.mj_step(self.model, self.data)

        # 【关键】：从 Data 中读取 ee_site 的实时位姿
        if self.ee_site_id != -1:
            pos = self.data.site_xpos[self.ee_site_id].copy()  # 3维位置
            mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()  # 3x3 旋转矩阵

            # 组装成 4x4 矩阵
            T = np.eye(4)
            T[:3, :3] = mat
            T[:3, 3] = pos
            self.actual_pose_cache = T

    def get_realtime_data(self):
        """
        GUI 获取数据的接口
        返回:
        - target: [x,y,z, r,p,y] (6维数组)
        - actual: [x,y,z, r,p,y] (6维数组)
        - global_img: 图像数据
        """

        # 辅助函数：矩阵转 6D 向量
        def mat2pose6d(T):
            if T is None: return np.zeros(6)
            xyz = T[:3, 3]
            # trimesh.transformations.euler_from_matrix 默认是 'sxyz' (静态坐标系)
            rpy = tf.euler_from_matrix(T[:3, :3], 'sxyz')
            return np.concatenate([xyz, rpy])

        # 转换数据
        target_6d = mat2pose6d(self.T_target_cache)
        actual_6d = mat2pose6d(self.actual_pose_cache)

        # 渲染图像 (仅渲染 Global 视角，节省性能)
        img_global = None
        if self.offscreen_window:
            glfw.make_context_current(self.offscreen_window)
            mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(), None, self.cam_global,
                                   mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            self._add_markers_to_scene(self.scene)
            mujoco.mjr_render(self.viewport, self.scene, self.gl_context)
            mujoco.mjr_readPixels(self.rgb_buffer, None, self.viewport, self.gl_context)
            img_global = np.flipud(self.rgb_buffer).copy()

        return {
            'target': target_6d,
            'actual': actual_6d,
            'global_img': img_global
        }

    def _add_markers_to_scene(self, scene):
        # 调试用：在场景中画个小红球表示当前目标点
        if self.path_points is not None and self.current_idx < len(self.path_points):
            pt = self.path_points[self.current_idx]
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                size=[0.005, 0, 0], pos=pt, mat=np.eye(3).flatten(), rgba=[1, 0, 0, 1])
            scene.ngeom += 1

    def compute_target_matrix(self, position, normal, standoff=0.1):
        # 简单的姿态计算：Z轴反向对准法线
        target_pos = position + normal * standoff
        z_axis = -normal / np.linalg.norm(normal)
        ref_axis = np.array([1, 0, 0])
        # 防止万向节死锁
        if np.abs(np.dot(z_axis, ref_axis)) > 0.9: ref_axis = np.array([0, 1, 0])
        y_axis = np.cross(z_axis, ref_axis);
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis);
        x_axis /= np.linalg.norm(x_axis)
        T = np.eye(4)
        T[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
        T[:3, 3] = target_pos
        return T

    def close(self):
        self.running = False
        if self.offscreen_window: glfw.destroy_window(self.offscreen_window)
        glfw.terminate()