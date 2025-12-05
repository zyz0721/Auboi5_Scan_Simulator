import mujoco
import mujoco.viewer
import numpy as np
import glfw
import time
from casadi_ik import Kinematics


class MuJoCoSimulator:
    """
    MuJoCo 仿真控制器类库
    负责管理物理仿真环境、IK解算、离屏渲染以及场景几何体的绘制
    常用需修改参数：注释中标注[参数可调]，搜索可查询，为 mujoco 仿真环境中一些基础设置，可后续单独封装
    注意：cam_res离屏渲染摄像头的分辨率 (宽, 高)，分辨率越高，mjr_readPixels 越耗时，FPS越低。
    推荐：(200, 200) ~ (400, 400)
    其他基本可在GUI界面中调整
    """

    def __init__(self, xml_path, end_joint, robot_model_path, cam_res=(1280, 1280)):

        self.xml_path = xml_path
        self.cam_res = cam_res
        self.end_joint = end_joint
        self.robot_model_path = robot_model_path

        # 仿真状态标志位
        self.paused = True  # True: 暂停自动扫描，False: 运行中
        self.running = True  # True: 程序运行中，False: 准备退出
        self.target_qpos = None  # 缓存当前目标的关节角度，用于位置锁定

        # 扫描控制参数
        self.current_idx = 0  # 当前路径点的索引
        self.last_scan_time = 0  # 上一次切换点的时间戳
        self.scan_interval = 0.5  # [参数可调] 扫描间隔 (秒)

        # 路径数据
        self.path_points = None  # 路径点列表 (N, 3)
        self.path_normals = None  # 法向量列表 (N, 3)
        self.scan_height = 0.1  # [参数可调] 扫描时末端离表面的高度 (米)
        self.T_target_cache = None  # 缓存当前计算出的末端目标矩阵 (用于绘制指示球)

        # 加载模型与数据
        try:
            self.model = mujoco.MjModel.from_xml_path(self.xml_path)
            self.data = mujoco.MjData(self.model)
            # 初始化时记录当前姿态作为目标，防止瞬间弹跳
            self.target_qpos = self.data.qpos[:6].copy()
        except Exception as e:
            print(f"MuJoCo 模型加载失败: {e}")
            raise e

        # 初始化逆运动学求解器
        try:
            # 目标 Link 名称需与 XML 中一致 分别为机械臂末端节点名称和机械臂xml或urdf
            self.ik_solver = Kinematics(self.end_joint)
            self.ik_solver.buildFromMJCF(self.robot_model_path)
        except Exception as e:
            print(f"IK 求解器初始化失败: {e}")

        # 初始化渲染环境
        self.init_offscreen()

        # 启动被动查看器 (主仿真窗口)
        # launch_passive 允许在主线程控制 step，同时拥有一个交互式窗口
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def init_offscreen(self):

        # 初始化离屏渲染摄像头画面
        if not glfw.init():
            raise Exception("GLFW 初始化失败")

        # 创建一个不可见的窗口用于 OpenGL 上下文
        glfw.window_hint(glfw.VISIBLE, 0)
        self.offscreen_window = glfw.create_window(self.cam_res[0], self.cam_res[1], "Offscreen", None, None)
        glfw.make_context_current(self.offscreen_window)

        # 创建 MuJoCo 渲染上下文
        self.gl_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)

        # 配置摄像头
        # 1: 机器人手眼相机
        self.cam_robot = mujoco.MjvCamera()
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "ee_camera")
        if cam_id != -1:
            self.cam_robot.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam_robot.fixedcamid = cam_id

        # 2: 全局上帝视角
        self.cam_global = mujoco.MjvCamera()
        self.cam_global.type = mujoco.mjtCamera.mjCAMERA_FREE

        # [参数可调] 全局相机视角设置
        self.cam_global.lookat = [0.1, -0.1, 0.4]  # 观察中心点 (x, y, z)
        self.cam_global.distance = 1.8  # 距离 (越小越近)
        self.cam_global.elevation = -30  # 俯仰角 (负值代表俯视)
        self.cam_global.azimuth = 135  # 方位角 (水平旋转)

    def set_path(self, points, normals, height=0.1):

        # 设置新的扫描路径，并重置机器人状态到起点
        if points is None or len(points) == 0: return

        self.path_points = points
        self.path_normals = normals
        self.scan_height = height
        self.current_idx = 0

        # 立即计算第一个点的 IK
        self.perform_ik_step()

        # 将机器人移到起点，避免物理不稳定
        if self.target_qpos is not None:
            self.data.qpos[:6] = self.target_qpos
            self.data.qvel[:6] = 0.0
            mujoco.mj_forward(self.model, self.data)

            # 双重确认：在当前状态下再次求解 IK 以提高精度
            self.perform_ik_step()
            self.data.qpos[:6] = self.target_qpos
            mujoco.mj_forward(self.model, self.data)

    def manual_adjust(self, direction):

        # 键盘控制手动切换路径点
        if self.path_points is None: return
        self.current_idx = (self.current_idx + direction) % len(self.path_points)
        self.perform_ik_step()

    def perform_ik_step(self):

        # 执行单次逆运动学计算
        if self.path_points is None: return

        target_pt = self.path_points[self.current_idx]
        target_nm = self.path_normals[self.current_idx]

        # 计算末端执行器的目标 4x4 矩阵
        T_target = self.compute_target_matrix(target_pt, target_nm, self.scan_height)
        self.T_target_cache = T_target  # 缓存用于绘制绿球

        # 使用当前关节角度作为 IK 求解的初始猜测 (Warm Start)
        init_q = self.data.qpos[:6].copy()
        try:
            q_sol, info = self.ik_solver.ik(T_target, current_arm_motor_q=init_q)
            if info["success"]:
                # 限制关节角度在物理极限内
                sol = np.clip(q_sol, self.model.jnt_range[:6, 0], self.model.jnt_range[:6, 1])
                self.target_qpos = sol
        except:
            pass

    def _add_markers_to_scene(self, scene):

        # MuJoCo 场景中添加指示小球
        if self.path_points is not None:
            # 确保场景几何体缓冲区未满
            if scene.ngeom + 2 < scene.maxgeom:

                # [参数可调] A. 红色球：表示物体表面的原始采样点
                pt_surface = self.path_points[self.current_idx]
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.0025, 0, 0],  # 半径设置
                    pos=pt_surface,
                    mat=np.eye(3).flatten(),
                    rgba=[1, 0, 0, 1]  # 红色，不透明
                )
                scene.ngeom += 1

                # [参数可调] B. 绿色球：表示机械臂末端的目标位置 (TCP)
                if self.T_target_cache is not None:
                    pt_tcp = self.T_target_cache[:3, 3]
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.0025, 0, 0],  # 半径设置
                        pos=pt_tcp,
                        mat=np.eye(3).flatten(),
                        rgba=[0, 1, 0, 0.6]  # 绿色，半透明
                    )
                    scene.ngeom += 1

    def step(self):

        # 物理仿真主循环步进
        # [参数可调] 子步进次数：增加此值可提高物理稳定性，减少此值可提高性能
        substeps = 10

        for _ in range(substeps):
            # 自动扫描逻辑
            if not self.paused and self.path_points is not None:
                now = time.time()
                if now - self.last_scan_time > self.scan_interval:
                    self.current_idx = (self.current_idx + 1) % len(self.path_points)
                    self.perform_ik_step()
                    self.last_scan_time = now

            # 位置控制
            if self.target_qpos is not None:
                self.data.qpos[:6] = self.target_qpos
                self.data.qvel[:6] = 0.0

            # 执行一步物理计算
            mujoco.mj_step(self.model, self.data)

        # 同步交互式查看器
        if self.viewer.is_running():
            # 必须在 viewer 的场景中也添加 marker，否则主窗口看不到球
            self.viewer.user_scn.ngeom = 0
            self._add_markers_to_scene(self.viewer.user_scn)
            self.viewer.sync()

    def render_offscreen(self):

        # 渲染 GUI 所需图像
        glfw.make_context_current(self.offscreen_window)
        viewport = mujoco.MjrRect(0, 0, self.cam_res[0], self.cam_res[1])

        # 1. 渲染机器人视角：更新场景 -> 添加Marker -> 渲染
        mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(), None, self.cam_robot,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        self._add_markers_to_scene(self.scene)
        mujoco.mjr_render(viewport, self.scene, self.gl_context)

        # 摄像头读取像素 (耗时操作影响帧数)
        rgb_robot = np.zeros((self.cam_res[1], self.cam_res[0], 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb_robot, None, viewport, self.gl_context)
        rgb_robot = np.flipud(rgb_robot)  # OpenGL 坐标系翻转

        # 2. 渲染全局视角
        mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(), None, self.cam_global,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        self._add_markers_to_scene(self.scene)
        mujoco.mjr_render(viewport, self.scene, self.gl_context)

        rgb_global = np.zeros((self.cam_res[1], self.cam_res[0], 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb_global, None, viewport, self.gl_context)
        rgb_global = np.flipud(rgb_global)

        return rgb_robot, rgb_global

    def compute_target_matrix(self, position, normal, standoff=0.1):

        # 根据点位置和法向量，计算末端 6D 姿态矩阵，确保 Z 轴对齐法向量，且 X/Y 轴方向合理
        target_pos = position + normal * standoff
        z_axis = -normal / np.linalg.norm(normal)  # Z 轴指向物体表面

        # 构建坐标系
        ref_axis = np.array([1, 0, 0])
        # 防止死锁
        if np.abs(np.dot(z_axis, ref_axis)) > 0.9: ref_axis = np.array([0, 1, 0])

        y_axis = np.cross(z_axis, ref_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        T = np.eye(4)
        T[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
        T[:3, 3] = target_pos
        return T

    def close(self):
        # 清理仿真占用资源
        self.running = False
        if self.viewer:
            self.viewer.close()
        if self.offscreen_window:
            glfw.destroy_window(self.offscreen_window)
        glfw.terminate()