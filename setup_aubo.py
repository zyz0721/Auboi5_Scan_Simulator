import sys
import os
import numpy as np
import trimesh
import pyvista as pv
import vtk
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QGroupBox, QRadioButton, QCheckBox, QFormLayout,
                             QFrame, QMessageBox, QFileDialog, QScrollArea)
from PyQt5.QtCore import Qt
from pyvistaqt import QtInteractor
import curve_utils

INPUT_FILE = 'models/organic_sample.STL'


class RobotPathInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """
    曲面路径规划自定义窗口执行
    功能：选取spiral/zigzag扫描模式；
         设置扫描步长和等高筛选阈值；
         设置扫描参数及框选区域
    """

    def __init__(self, parent=None):
        self.AddObserver("RightButtonPressEvent", self.right_button_press)
        self.AddObserver("RightButtonReleaseEvent", self.right_button_release)

    def right_button_press(self, obj, event):
        self.StartPan()

    def right_button_release(self, obj, event):
        self.EndPan()


class DarkStyle:
    SHEET = """
    QMainWindow { background-color: #2b2b2b; }
    QWidget { color: #e0e0e0; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; font-size: 10pt; }
    QGroupBox { 
        border: 1px solid #555; border-radius: 5px; margin-top: 15px; font-weight: bold;
    }
    QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
    QLineEdit, QPushButton, QComboBox {
        background-color: #3c3f41; border: 1px solid #555; padding: 5px; border-radius: 3px;
    }
    QPushButton { background-color: #2d5a88; color: white; font-weight: bold; }
    QPushButton:hover { background-color: #356a9e; }
    """


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("离线轨迹生成")
        self.resize(1500, 900)
        self.setStyleSheet(DarkStyle.SHEET)

        self.current_points = np.array([])
        self.current_normals = np.array([])
        self.mesh = None

        self.init_ui()
        self.load_mesh()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ====== 左侧控制面板 (带滚动条适配多参数) ======
        scroll_area = QScrollArea()
        scroll_area.setFixedWidth(800)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")

        left_widget = QWidget()
        self.control_layout = QVBoxLayout(left_widget)
        scroll_area.setWidget(left_widget)

        # 1. 扫描路径基础设置
        group_path = QGroupBox("1. 基础扫描参数设置")
        form_path = QFormLayout()

        mode_layout = QHBoxLayout()
        self.rb_spiral = QRadioButton("Spiral (螺旋)")
        self.rb_zigzag = QRadioButton("Zigzag (弓字)")
        self.rb_spiral.setChecked(True)
        mode_layout.addWidget(self.rb_spiral)
        mode_layout.addWidget(self.rb_zigzag)
        form_path.addRow("扫描模式:", mode_layout)

        self.le_step = QLineEdit("5.0")
        self.le_radius = QLineEdit("100.0")
        self.le_cx = QLineEdit("0.0")
        self.le_cy = QLineEdit("0.0")
        form_path.addRow("扫描步长 (mm):", self.le_step)
        form_path.addRow("最大半径 (mm):", self.le_radius)
        form_path.addRow("中心 X (mm):", self.le_cx)
        form_path.addRow("中心 Y (mm):", self.le_cy)
        group_path.setLayout(form_path)
        self.control_layout.addWidget(group_path)

        # 2. 导出 Aubo 机械臂专属设置面板 (新增功能)
        self.init_aubo_ui(self.control_layout)

        # 3. 操作按钮区
        group_actions = QGroupBox("3. 执行操作")
        action_layout = QVBoxLayout()

        self.btn_generate = QPushButton("1. 生成并在右侧预览路径")
        self.btn_generate.setMinimumHeight(35)
        self.btn_generate.clicked.connect(self.generate_path)

        self.btn_save_npz = QPushButton("2. 保存为 Numpy (.npz)")
        self.btn_save_npz.clicked.connect(self.save_path)

        self.btn_export_aubo = QPushButton("3. 导出 Aubo 离线执行文件 (.aubo)")
        self.btn_export_aubo.setMinimumHeight(40)
        self.btn_export_aubo.setStyleSheet("background-color: #e67e22; color: white; font-size: 11pt;")
        self.btn_export_aubo.clicked.connect(self.export_aubo)

        action_layout.addWidget(self.btn_generate)
        action_layout.addWidget(self.btn_save_npz)
        action_layout.addWidget(self.btn_export_aubo)
        group_actions.setLayout(action_layout)
        self.control_layout.addWidget(group_actions)

        self.control_layout.addStretch()
        main_layout.addWidget(scroll_area)

        # ====== 右侧 3D 视图区 ======
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#1e1e1e")
        style = RobotPathInteractorStyle()
        self.plotter.interactor.SetInteractorStyle(style)
        main_layout.addWidget(self.plotter.interactor)

    def init_aubo_ui(self, layout):
        """初始化 Aubo 相关参数的 UI 面板"""
        group = QGroupBox("2. Aubo 机械臂参数及零点映射设置")
        form = QFormLayout()

        # 运行环境参数
        self.le_tool = QLineEdit("tool2")
        self.le_user = QLineEdit("user2")
        self.le_blend = QLineEdit("0.01")
        self.le_vel = QLineEdit("0.05")
        self.le_acc = QLineEdit("0.5")

        form.addRow("工具坐标系 (Tool):", self.le_tool)
        form.addRow("工件坐标系 (User):", self.le_user)
        form.addRow("交融半径 (Blend, m):", self.le_blend)
        form.addRow("运行线速度 (m/s):", self.le_vel)
        form.addRow("运行加速度 (m/s²):", self.le_acc)

        # 零点标定参数 (首个点的映射基准)
        line = QFrame();
        line.setFrameShape(QFrame.HLine);
        line.setStyleSheet("color: #555;")
        form.addRow(line)
        form.addRow(QLabel("<font color='#a0c4ff'>起始点 (首个点) 在真实机械臂下的位姿：</font>"))

        # 默认使用你提供的初始坐标
        self.le_z_x = QLineEdit("0.134523")
        self.le_z_y = QLineEdit("0.276265")
        self.le_z_z = QLineEdit("0.023243")
        self.le_z_rx = QLineEdit("-2.669935")
        self.le_z_ry = QLineEdit("0.003403")
        self.le_z_rz = QLineEdit("1.570796")

        h_pos1 = QHBoxLayout()
        h_pos1.addWidget(QLabel("X:"))
        h_pos1.addWidget(self.le_z_x)
        h_pos1.addWidget(QLabel("Y:"))
        h_pos1.addWidget(self.le_z_y)
        form.addRow("零点位置 (m):", h_pos1)

        h_pos2 = QHBoxLayout()
        h_pos2.addWidget(QLabel("Z:"))
        h_pos2.addWidget(self.le_z_z)
        h_pos2.addWidget(QLabel(" "))
        h_pos2.addWidget(QLabel(" "))
        form.addRow("", h_pos2)

        h_rot1 = QHBoxLayout()
        h_rot1.addWidget(QLabel("RX:"))
        h_rot1.addWidget(self.le_z_rx)
        h_rot1.addWidget(QLabel("RY:"))
        h_rot1.addWidget(self.le_z_ry)
        form.addRow("零点姿态 (rad):", h_rot1)

        h_rot2 = QHBoxLayout()
        h_rot2.addWidget(QLabel("RZ:"))
        h_rot2.addWidget(self.le_z_rz)
        h_rot2.addWidget(QLabel(" "))
        h_rot2.addWidget(QLabel(" "))
        form.addRow("", h_rot2)

        self.cb_unit_convert = QCheckBox("自动将规划的坐标(mm)转为机械臂尺度(m)")
        self.cb_unit_convert.setChecked(True)
        form.addRow(self.cb_unit_convert)

        group.setLayout(form)
        layout.addWidget(group)

    def load_mesh(self):
        try:
            if os.path.exists(INPUT_FILE):
                self.mesh = trimesh.load(INPUT_FILE)
                pv_mesh = pv.wrap(self.mesh)
                pv_mesh.points *= 1000.0
                self.plotter.add_mesh(pv_mesh, color="#8a8a8a", opacity=0.8)
            else:
                print(f"Warning: Model file {INPUT_FILE} not found.")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载模型失败: {str(e)}")

    def generate_path(self):
        if self.mesh is None:
            QMessageBox.warning(self, "警告", "未加载有效模型")
            return

        try:
            step = float(self.le_step.text())
            radius = float(self.le_radius.text())
            cx = float(self.le_cx.text())
            cy = float(self.le_cy.text())
        except ValueError:
            QMessageBox.warning(self, "错误", "参数输入必须为数字")
            return

        # 调用 curve_utils 进行路径生成
        # 这里默认调用螺旋，您可以根据需要完善 zigzag 的逻辑对接
        if self.rb_spiral.isChecked():
            pts, norms = curve_utils.CurvePathPlanner.compute_spiral_3d(
                self.mesh, cx, cy, radius, step, z_thresh=-0.5
            )
        else:
            # Placeholder for zigzag
            pts, norms = curve_utils.CurvePathPlanner.compute_spiral_3d(
                self.mesh, cx, cy, radius, step, z_thresh=-0.5
            )

        self.current_points = pts
        self.current_normals = norms
        self.update_plotter(pts, norms)

    def update_plotter(self, points, normals):
        self.plotter.clear()
        if self.mesh is not None:
            self.plotter.add_mesh(pv.wrap(self.mesh), color="#8a8a8a", opacity=0.8)

        cam_pos = self.plotter.camera.position
        cam_focal = self.plotter.camera.focal_point
        cam_up = self.plotter.camera.up

        if len(points) > 0:
            try:
                step = float(self.le_step.text())
            except:
                step = 5.0

            # 1：连线
            lines = []
            for i in range(len(points) - 1):
                lines.extend([2, i, i + 1])
            if lines:
                line_mesh = pv.PolyData(points)
                line_mesh.lines = lines
                self.plotter.add_mesh(line_mesh, color="red", line_width=2)

            # 2：点云
            self.plotter.add_mesh(points, scalars=points[:, 2], cmap="viridis",
                                  point_size=6, render_points_as_spheres=True, show_scalar_bar=False)

            # 3：法向量样式
            cone_source = pv.Cone(radius=0.04, height=0.15, direction=(1, 0, 0), resolution=12)
            pdata = pv.PolyData(points)
            pdata["normals"] = normals
            glyphs = pdata.glyph(scale=False, orient="normals", geom=cone_source, factor=step * 0.6)
            self.plotter.add_mesh(glyphs, color="#dddddd", opacity=0.6)

        else:
            QMessageBox.information(self, "提示", "未生成有效路径")

        self.plotter.camera.position = cam_pos
        self.plotter.camera.focal_point = cam_focal
        self.plotter.camera.up = cam_up

    def save_path(self):
        if len(self.current_points) == 0:
            QMessageBox.warning(self, "警告", "没有路径可保存")
            return
        mode_str = "spiral" if self.rb_spiral.isChecked() else "zigzag"
        filename = f"scan_path_{mode_str}.npz"
        np.savez(filename, points=self.current_points, normals=self.current_normals, mode=mode_str)
        QMessageBox.information(self, "保存成功", f"文件已保存为 {filename}")

    # ================= 新增：生成 Aubo 执行文件核心逻辑 =================
    def export_aubo(self):
        if not hasattr(self, 'current_points') or len(self.current_points) == 0:
            QMessageBox.warning(self, "警告", "请先点击生成路径，确认有轨迹点后再导出！")
            return

        # 1. 解析面板设置的基准参数
        try:
            blend = float(self.le_blend.text())
            vel = float(self.le_vel.text())
            acc = float(self.le_acc.text())
            z_x = float(self.le_z_x.text())
            z_y = float(self.le_z_y.text())
            z_z = float(self.le_z_z.text())
            z_rx = float(self.le_z_rx.text())
            z_ry = float(self.le_z_ry.text())
            z_rz = float(self.le_z_rz.text())
        except ValueError:
            QMessageBox.warning(self, "错误", "Aubo参数输入框中含有非法字符，请确保全部为数字！")
            return

        tool_name = self.le_tool.text().strip()
        user_name = self.le_user.text().strip()

        # 2. 选择文件保存路径
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存 Aubo 执行文件", "scan_path.aubo", "Aubo Script (*.aubo);;All Files (*)", options=options)
        if not file_path:
            return

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # ====== 写入环境和安全初始化参数 ======
                f.write("init_global_move_profile()\n")
                f.write(f"toolpos,toolori=get_tool_kinematics_param(\"{tool_name}\")\n")
                f.write(f"set_arrival_ahead_blend_mode({blend})\n")
                f.write("set_joint_maxvelc({0.259,0.259,0.259,0.31,0.31,0.31})\n")
                f.write("set_joint_maxacc({8.65,8.65,8.65,10.35,10.35,10.35})\n")
                f.write(f"set_end_maxvelc({vel})\n")
                f.write(f"set_end_maxacc({acc})\n\n")

                # ====== 计算位姿映射基准 ======
                p0 = self.current_points[0]
                n0 = self.current_normals[0]

                # 获取首个设定姿态的旋转矩阵 (Aubo使用固定轴 RPY，对应 trimesh 的 'sxyz')
                R0_mat = trimesh.transformations.euler_matrix(z_rx, z_ry, z_rz, axes='sxyz')

                # 比例因子：如果模型尺寸是mm，则需要换算成m给机械臂
                scale = 0.001 if self.cb_unit_convert.isChecked() else 1.0

                # ====== 遍历规划点位生成指令 ======
                for i in range(len(self.current_points)):
                    pi = self.current_points[i]
                    ni = self.current_normals[i]

                    # 1. 计算位置相对偏差，叠加到你设置的零点坐标上
                    dx = (pi[0] - p0[0]) * scale
                    dy = (pi[1] - p0[1]) * scale
                    dz = (pi[2] - p0[2]) * scale

                    target_x = z_x + dx
                    target_y = z_y + dy
                    target_z = z_z + dz

                    # 2. 计算法向量相对姿态偏置，叠加到你设置的零点姿态上
                    if np.allclose(n0, ni):
                        target_rx, target_ry, target_rz = z_rx, z_ry, z_rz
                    else:
                        # 计算从 n0 旋转到当前点 ni 的空间变换矩阵
                        rot_mat = trimesh.geometry.align_vectors(n0, ni)
                        # 将这个旋转变化叠加到首个姿态矩阵 R0_mat 上
                        target_R_mat = np.dot(rot_mat, R0_mat)
                        # 从最终的变换矩阵中提取出 RPY 欧拉角
                        target_rx, target_ry, target_rz = trimesh.transformations.euler_from_matrix(target_R_mat,
                                                                                                    axes='sxyz')

                    # 3. 写入单行 move_line 运动指令
                    cmd = (f"move_line(get_target_pose("
                           f"{{{target_x:.6f},{target_y:.6f},{target_z:.6f}}},"
                           f"rpy2quaternion({{{target_rx:.6f},{target_ry:.6f},{target_rz:.6f}}}),"
                           f"false,toolpos,toolori,get_user_coord_param(\"{user_name}\")),true)\n")
                    f.write(cmd)

            QMessageBox.information(self, "成功", f"Aubo 离线文件生成完毕！\n请导入示教器运行验证。\n路径：{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "导出错误", f"发生异常：\n{str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())