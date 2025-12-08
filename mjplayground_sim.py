import mujoco
import mujoco.viewer
import numpy as np
import glfw
import time
from casadi_ik import Kinematics


class MuJoCoSimulator:
    """
    MuJoCo 仿真控制器 - playground性能优化版
    常用需修改参数：注释中标注[参数可调]，搜索可查询，为 mujoco 仿真环境中一些基础设置，可后续单独封装
    优化策略：
    1. Control Decimation: 分离物理频率(500Hz)与控制频率(50Hz)，大幅减少 CasADi 计算次数
    2. Memory Pre-allocation: 预分配渲染内存，减少 GC 压力
    """

    def __init__(self, xml_path, end_joint, robot_model_path, cam_res=(1280, 1280)):

        self.xml_path = xml_path
        self.cam_res = cam_res
        self.end_joint = end_joint
        self.robot_model_path = robot_model_path

        # 仿真状态标志位
        self.paused = True
        self.running = True
        self.target_qpos = None  # 缓存当前目标的关节角度

        # 时间步长管理
        self.physics_dt = 0.002  # [参数可调]物理仿真步长 (500Hz)
        self.control_dt = 0.02  # [参数可调]控制/IK 步长 (50Hz)
        # 计算每次控制指令后，物理引擎需要空跑多少步 (Decimation)
        self.n_substeps = int(self.control_dt / self.physics_dt)

        # 扫描控制参数
        self.current_idx = 0
        self.last_scan_time = 0
        self.scan_interval = 0.5  # [参数可调]扫描点切换间隔

        self.path_points = None
        self.path_normals = None
        self.scan_height = 0.1  # [参数可调] 扫描时末端离表面的高度 (米)
        self.T_target_cache = None

        # 加载模型
        try:
            self.model = mujoco.MjModel.from_xml_path(self.xml_path)
            self.data = mujoco.MjData(self.model)

            # 强制设置物理步长，确保与逻辑一致
            self.model.opt.timestep = self.physics_dt

            # 初始化位置
            self.target_qpos = self.data.qpos[:6].copy()
        except Exception as e:
            print(f"MuJoCo 模型加载失败: {e}")
            raise e

        # 初始化 IK 求解器 (CasADi)
        try:
            self.ik_solver = Kinematics(self.end_joint)
            self.ik_solver.buildFromMJCF(self.robot_model_path)
        except Exception as e:
            print(f"IK 求解器初始化失败: {e}")

        # 初始化渲染
        self.init_offscreen()

        # 启动被动查看器
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def init_offscreen(self):
        # 初始化离屏渲染
        if not glfw.init():
            raise Exception("GLFW 初始化失败")

        glfw.window_hint(glfw.VISIBLE, 0)
        self.offscreen_window = glfw.create_window(self.cam_res[0], self.cam_res[1], "Offscreen", None, None)
        glfw.make_context_current(self.offscreen_window)

        self.gl_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)

        # 预分配渲染内存，避免循环中重复 malloc
        self.viewport = mujoco.MjrRect(0, 0, self.cam_res[0], self.cam_res[1])
        # 预分配 Buffer (注意 OpenGL 读取通常需要对齐，这里简化处理)
        self.rgb_buffer = np.zeros((self.cam_res[1], self.cam_res[0], 3), dtype=np.uint8)

        # 配置摄像头
        self.cam_robot = mujoco.MjvCamera()
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "ee_camera")
        if cam_id != -1:
            self.cam_robot.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam_robot.fixedcamid = cam_id

        self.cam_global = mujoco.MjvCamera()
        self.cam_global.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam_global.lookat = [0.1, -0.1, 0.4]
        self.cam_global.distance = 1.8
        self.cam_global.elevation = -30
        self.cam_global.azimuth = 135

    def set_path(self, points, normals, height=0.1):
        if points is None or len(points) == 0: return

        self.path_points = points
        self.path_normals = normals
        self.scan_height = height
        self.current_idx = 0

        # 立即计算第一个点
        self.perform_ik_step()

        # 重置物理状态
        if self.target_qpos is not None:
            self.data.qpos[:6] = self.target_qpos
            self.data.qvel[:6] = 0.0
            mujoco.mj_forward(self.model, self.data)

    def perform_ik_step(self):
        # 执行一次 IK 计算
        if self.path_points is None: return

        target_pt = self.path_points[self.current_idx]
        target_nm = self.path_normals[self.current_idx]

        T_target = self.compute_target_matrix(target_pt, target_nm, self.scan_height)
        self.T_target_cache = T_target

        # Warm Start: 使用当前角度加速收敛
        init_q = self.data.qpos[:6].copy()

        try:
            # CasADi 计算瓶颈所在
            q_sol, info = self.ik_solver.ik(T_target, current_arm_motor_q=init_q)
            if info["success"]:
                # 限制关节极限
                sol = np.clip(q_sol, self.model.jnt_range[:6, 0], self.model.jnt_range[:6, 1])
                self.target_qpos = sol
        except Exception:
            pass

    def step(self):

        # A. 控制层 (50Hz)
        # 负责路径规划和 IK 解算，频率较低
        if not self.paused and self.path_points is not None:
            now = time.time()
            if now - self.last_scan_time > self.scan_interval:
                self.current_idx = (self.current_idx + 1) % len(self.path_points)
                self.last_scan_time = now

        # 每一帧 Control Step 计算一次 IK 目标(CasADi 在这里被调用，频率降低)
        self.perform_ik_step()

        # B. 物理层 (500Hz)
        # 物理引擎为了数值稳定性需要高频运行,让物理引擎连续跑 n_substeps 步来追赶一个 control_step
        for _ in range(self.n_substeps):

            # 应用控制目标
            if self.target_qpos is not None:
                # 如果模型有位置执行器(position servo)，推荐使用 data.ctrl
                if self.model.nu >= 6:
                    self.data.ctrl[:6] = self.target_qpos
                else:
                    # 否则直接设置 qpos (Kinematic mode)
                    self.data.qpos[:6] = self.target_qpos
                    self.data.qvel[:6] = 0.0

            # 物理步进
            mujoco.mj_step(self.model, self.data)

        # C. 查看器同步
        if self.viewer.is_running():
            self.viewer.user_scn.ngeom = 0
            self._add_markers_to_scene(self.viewer.user_scn)
            self.viewer.sync()

    def render_offscreen(self):

        # 优化的渲染函数：复用内存，减少拷贝
        glfw.make_context_current(self.offscreen_window)

        # 1. 渲染机器人视角
        mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(), None, self.cam_robot,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        self._add_markers_to_scene(self.scene)
        mujoco.mjr_render(self.viewport, self.scene, self.gl_context)

        # 读取像素到预分配的 Buffer
        mujoco.mjr_readPixels(self.rgb_buffer, None, self.viewport, self.gl_context)
        # 必须 flip，因为 OpenGL 原点在左下角。使用 .copy() 确保返回的数据是独立的
        rgb_robot = np.flipud(self.rgb_buffer).copy()

        # 2. 渲染全局视角
        mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(), None, self.cam_global,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        self._add_markers_to_scene(self.scene)
        mujoco.mjr_render(self.viewport, self.scene, self.gl_context)

        # 复用 Buffer
        mujoco.mjr_readPixels(self.rgb_buffer, None, self.viewport, self.gl_context)
        rgb_global = np.flipud(self.rgb_buffer).copy()

        return rgb_robot, rgb_global

    def _add_markers_to_scene(self, scene):
        # 辅助函数：绘制调试用的红绿小球
        if self.path_points is not None and scene.ngeom + 2 < scene.maxgeom:

            # 红色：表面点
            pt_surface = self.path_points[self.current_idx]
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.003, 0, 0],
                pos=pt_surface,
                mat=np.eye(3).flatten(),
                rgba=[1, 0, 0, 1]
            )
            scene.ngeom += 1

            # 绿色：末端目标点
            if self.T_target_cache is not None:
                pt_tcp = self.T_target_cache[:3, 3]
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.003, 0, 0],
                    pos=pt_tcp,
                    mat=np.eye(3).flatten(),
                    rgba=[0, 1, 0, 0.6]
                )
                scene.ngeom += 1

    def compute_target_matrix(self, position, normal, standoff=0.1):
        # 计算对齐法向量的目标位姿
        target_pos = position + normal * standoff
        z_axis = -normal / np.linalg.norm(normal)
        ref_axis = np.array([1, 0, 0])
        if np.abs(np.dot(z_axis, ref_axis)) > 0.9: ref_axis = np.array([0, 1, 0])
        y_axis = np.cross(z_axis, ref_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        T = np.eye(4)
        T[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
        T[:3, 3] = target_pos
        return T

    def manual_adjust(self, direction):
        if self.path_points is None: return
        self.current_idx = (self.current_idx + direction) % len(self.path_points)
        self.perform_ik_step()

    def close(self):
        self.running = False
        if self.viewer:
            self.viewer.close()
        if self.offscreen_window:
            glfw.destroy_window(self.offscreen_window)
        glfw.terminate()