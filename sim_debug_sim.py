import os
import sys
import time
import numpy as np
import trimesh
import pyvista as pv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QGroupBox, QRadioButton, QCheckBox, QFormLayout,
                             QFrame, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from pyvistaqt import QtInteractor
import pyqtgraph as pg  # 需安装: pip install pyqtgraph

import curve_utils
from mujoco_debug_sim import MuJoCoSimulator
from aubo_interface import AuboRealRobot  # 导入刚才生成的新接口文件

# 常量定义
SCENE_XML_PATH = "mjcf/scene_with_curvemodel.xml"
SAMPLE_PATH = "models/curve_model.stl"
END_JOINT = "wrist3_Link"
ROBOTICARM_MODEL_PATH = "mjcf/aubo_i5_withcam.xml"
CAM_RES = (1280, 1280)
SAMPLE_OFFSET = [-0.25, -0.6, 0.2]

# 设置 pyqtgraph 的背景色为白色，前景色为黑色
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class PoseMonitorWidget(pg.GraphicsLayoutWidget):
    """
    自定义控件：6D 位姿实时监控
    包含 6 个子图：X, Y, Z, Roll, Pitch, Yaw
    """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ci.layout.setContentsMargins(5, 5, 5, 5)
        self.ci.layout.setSpacing(5)

        # 6个维度的标签
        self.labels = ['X (m)', 'Y (m)', 'Z (m)', 'Roll (rad)', 'Pitch (rad)', 'Yaw (rad)']

        self.plots = []
        self.curves_target = []
        self.curves_actual = []

        # 创建 2行3列 的图表布局
        for i in range(6):
            row = i // 3
            col = i % 3
            # 添加子图
            p = self.addPlot(row=row, col=col, title=self.labels[i])
            p.showGrid(x=True, y=True, alpha=0.3)

            # 创建曲线对象：蓝色=目标，红色=实际(加粗)
            c_target = p.plot(pen=pg.mkPen(color='#3498db', width=2), name="Target")
            c_actual = p.plot(pen=pg.mkPen(color='#e74c3c', width=2), name="Actual")

            self.plots.append(p)
            self.curves_target.append(c_target)
            self.curves_actual.append(c_actual)

        # 数据缓冲区 (保留最近 300 帧数据)
        self.buffer_size = 300
        self.data_target = np.zeros((6, self.buffer_size))
        self.data_actual = np.zeros((6, self.buffer_size))
        # 用 NaN 初始化，避免一开始显示一条横线
        self.data_target[:] = np.nan
        self.data_actual[:] = np.nan
        self.ptr = 0

    def update_data(self, target_6d, actual_6d):
        """
        接收新数据并刷新图表
        :param target_6d: [x, y, z, r, p, y]
        :param actual_6d: [x, y, z, r, p, y]
        """
        if target_6d is None or actual_6d is None:
            return

        # 数据左移一位
        self.data_target[:, :-1] = self.data_target[:, 1:]
        self.data_actual[:, :-1] = self.data_actual[:, 1:]

        # 填入最新数据到最后一位
        self.data_target[:, -1] = target_6d
        self.data_actual[:, -1] = actual_6d

        self.ptr += 1

        # 刷新所有曲线
        for i in range(6):
            self.curves_target[i].setData(self.data_target[i])
            self.curves_actual[i].setData(self.data_actual[i])


class SquareLabel(QLabel):
    # 保持原有的图像显示控件，用于显示全局视图
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.setLineWidth(1)
        self.setStyleSheet("background-color: #000; border: 1px solid #555; color: #aaa;")
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(100, 100)

    def heightForWidth(self, width):
        return width

    def sizeHint(self):
        return self.size()

    def resizeEvent(self, event):
        # 强制保持正方形比例 (可选)
        if self.width() != self.height():
            self.setFixedHeight(self.width())
        super().resizeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aubo Robot Simulation & Control System")
        self.resize(1600, 1000)

        # 简单的暗色主题样式
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QWidget { color: #e0e0e0; font-family: "Segoe UI"; font-size: 10pt; }
            QLineEdit { background-color: #404040; border: 1px solid #555; padding: 4px; color: #fff; }
            QPushButton { background-color: #0078d4; border: none; border-radius: 4px; padding: 6px; color: white; font-weight: bold;}
            QPushButton:hover { background-color: #1084e0; }
            QGroupBox { border: 1px solid #555; margin-top: 10px; padding-top: 10px; font-weight: bold;}
            QGroupBox::title { color: #aaa; }
            QLabel { color: #ddd; }
        """)

        # 核心对象
        self.stl_path = SAMPLE_PATH
        self.trimesh_obj = None
        self.current_points = []

        self.sim = None  # 仿真后端
        self.real_robot = None  # 真实机器人接口

        # 初始化仿真
        try:
            self.sim = MuJoCoSimulator(SCENE_XML_PATH, END_JOINT, ROBOTICARM_MODEL_PATH, CAM_RES)
        except Exception as e:
            print(f"仿真环境初始化失败: {e}")

        # 设置定时器循环 (约30FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.game_loop)
        self.timer.start(33)

        self.init_ui()
        self.load_model()

        # 这里演示如何初始化真实机器人 (但不自动连接，防止误操作)
        # 真实连接请在界面上点击按钮或在代码中显式调用
        self.real_robot = AuboRealRobot("192.168.1.10")

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # === 1. 左侧控制面板 ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 参数设置区
        left_layout.addWidget(self.create_settings_panel())

        # 真实机器人控制区
        grp_real = QGroupBox("真实机械臂控制")
        l_real = QVBoxLayout()

        self.btn_connect = QPushButton("连接机械臂 (Connect)")
        self.btn_connect.clicked.connect(self.toggle_connection)
        self.btn_sync = QPushButton("同步当前轨迹 (Sync)")
        self.btn_sync.clicked.connect(self.sync_motion)
        self.btn_sync.setEnabled(False)  # 连接后才可用

        l_real.addWidget(self.btn_connect)
        l_real.addWidget(self.btn_sync)
        grp_real.setLayout(l_real)
        left_layout.addWidget(grp_real)

        left_layout.addStretch()
        left_panel.setFixedWidth(320)
        main_layout.addWidget(left_panel)

        # === 2. 右侧显示区域 ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # --- 上半部分：曲线图 + 全局监控 ---
        top_container = QWidget()
        top_layout = QHBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # [新增] 6D 位姿误差曲线图
        self.pose_plotter = PoseMonitorWidget()

        # [保留] 全局仿真视图
        self.lbl_cam_global = SquareLabel("Simulation View")
        self.lbl_cam_global.setFixedWidth(400)  # 固定宽度

        top_layout.addWidget(self.pose_plotter, stretch=1)  # 曲线图自适应拉伸
        top_layout.addWidget(self.lbl_cam_global, stretch=0)

        right_layout.addWidget(top_container, stretch=4)  # 上半部分占 40% 高度

        # --- 下半部分：3D 路径规划预览 ---
        self.plotter = QtInteractor(right_panel)
        # 添加一些基础视觉元素
        self.plotter.set_background("#e1e1e1")
        right_layout.addWidget(self.plotter, stretch=6)  # 下半部分占 60% 高度

        main_layout.addWidget(right_panel)

    def create_settings_panel(self):
        grp = QGroupBox("扫描参数设置")
        layout = QFormLayout()

        self.inp_step = QLineEdit("10.0")
        self.inp_z_thresh = QLineEdit("0.2")
        self.inp_height = QLineEdit("0.1")
        self.btn_gen = QPushButton("生成路径并仿真")
        self.btn_gen.clicked.connect(self.generate_path)
        self.btn_gen.setStyleSheet("background-color: #27ae60; color: white;")

        layout.addRow("步长 (mm):", self.inp_step)
        layout.addRow("Z轴法向阈值:", self.inp_z_thresh)
        layout.addRow("扫描高度 (m):", self.inp_height)
        layout.addRow(self.btn_gen)

        grp.setLayout(layout)
        return grp

    def load_model(self):
        if not os.path.exists(self.stl_path): return
        # 加载用于计算的 trimesh 对象
        self.trimesh_obj = curve_utils.CurvePathPlanner.ensure_single_mesh(trimesh.load(self.stl_path))
        self.trimesh_obj.apply_scale(1000.0)  # 转mm

        # 加载用于显示的 pyvista 对象
        mesh = pv.read(self.stl_path)
        mesh.points *= 1000.0
        self.plotter.add_mesh(mesh, color="white", opacity=0.5, show_edges=False)
        self.plotter.add_axes()
        self.plotter.view_isometric()

    def generate_path(self):
        if not self.trimesh_obj: return
        step = float(self.inp_step.text())
        z_thresh = float(self.inp_z_thresh.text())

        # 简单使用 Zigzag 策略生成
        pts, norms = curve_utils.CurvePathPlanner.generate_zigzag_path(self.trimesh_obj, step, z_thresh)

        if len(pts) > 0:
            self.current_points = pts
            # 更新3D预览
            self.plotter.clear()
            self.load_model()
            self.plotter.add_points(pts, color='red', point_size=6, render_points_as_spheres=True)

            # 绘制路径线
            line = pv.lines_from_points(pts)
            self.plotter.add_mesh(line, color='blue', line_width=2)

            # 发送给仿真后端
            offset = np.array(SAMPLE_OFFSET)
            scan_h = float(self.inp_height.text())
            pts_m = pts / 1000.0 + offset  # mm -> m 并加上场景偏移

            self.sim.set_path(pts_m, norms, height=scan_h)
            self.sim.paused = False  # 开始仿真运行

    def toggle_connection(self):
        if not self.real_robot.connected:
            if self.real_robot.connect():
                self.btn_connect.setText("断开连接 (Disconnect)")
                self.btn_connect.setStyleSheet("background-color: #c0392b;")
                self.btn_sync.setEnabled(True)
            else:
                self.btn_connect.setText("连接失败 (重试)")
        else:
            self.real_robot.disconnect()
            self.btn_connect.setText("连接机械臂 (Connect)")
            self.btn_connect.setStyleSheet("background-color: #0078d4;")
            self.btn_sync.setEnabled(False)

    def sync_motion(self):
        # 简单示例：将仿真当前的这一帧目标角度发送给真机
        # 实际应用中建议建立一个队列发送整条轨迹
        if self.sim and self.sim.target_qpos is not None:
            q = self.sim.target_qpos
            self.real_robot.move_j(q, v=0.2, a=0.2)
            print(f"已发送姿态: {q}")

    def game_loop(self):
        if not self.sim: return
        self.sim.step()

        # 1. 获取仿真数据 (包含 6D 位姿和全局图像)
        # get_realtime_data() 是我们在 Backend 中新加的函数
        sim_data = self.sim.get_realtime_data()

        # 2. 更新左上角的曲线图
        if sim_data['target'] is not None and sim_data['actual'] is not None:
            self.pose_plotter.update_data(sim_data['target'], sim_data['actual'])

        # 3. 更新右上角的全局视图
        if sim_data['global_img'] is not None:
            h, w, c = sim_data['global_img'].shape
            # 注意 OpenGL 图片可能需要 flip，视具体情况而定
            # 这里已经在 backend 做了 flipud，所以直接用
            qimg = QImage(sim_data['global_img'].data, w, h, 3 * w, QImage.Format_RGB888)
            self.lbl_cam_global.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        if self.sim: self.sim.close()
        if self.real_robot: self.real_robot.disconnect()
        super().closeEvent(event)


if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())