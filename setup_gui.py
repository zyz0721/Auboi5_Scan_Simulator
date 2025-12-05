import sys
import os
import numpy as np
import trimesh
import pyvista as pv
import vtk
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QGroupBox, QRadioButton, QCheckBox, QFormLayout,
                             QFrame, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from pyvistaqt import QtInteractor
import curve_utils

INPUT_FILE = 'models/sample.STL'


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
        border: 1px solid #555; border-radius: 5px; margin-top: 10px; font-weight: bold; 
        background-color: #333;
    }
    QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; left: 10px; color: #4facfe; }
    QLineEdit { 
        background-color: #404040; border: 1px solid #555; border-radius: 3px; padding: 4px; color: #fff; 
    }
    QLineEdit:focus { border: 1px solid #4facfe; }
    QPushButton { 
        background-color: #0078d4; border: none; border-radius: 4px; padding: 8px; color: white; font-weight: bold; 
    }
    QPushButton:hover { background-color: #1084e0; }
    QPushButton:pressed { background-color: #006cc1; }
    QLabel { color: #ccc; }
    """


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Scan Path Planner")
        self.resize(1600, 1000)
        self.setStyleSheet(DarkStyle.SHEET)

        # 数据
        self.stl_path = INPUT_FILE
        self.trimesh_obj = None
        self.pv_mesh = None
        self.current_points = []
        self.current_normals = []

        # 初始化 UI
        self.init_ui()

        # 加载模型
        self.load_model()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 左侧控制面板
        control_panel = QFrame()
        control_panel.setFixedWidth(350)
        panel_layout = QVBoxLayout(control_panel)

        # 1. 模式
        grp_mode = QGroupBox("扫描模式")
        v_mode = QVBoxLayout()
        self.rb_spiral = QRadioButton("Spiral (螺旋)")
        self.rb_spiral.setChecked(True)
        self.rb_spiral.toggled.connect(self.update_ui_state)
        self.rb_zigzag = QRadioButton("Zigzag (弓字)")
        v_mode.addWidget(self.rb_spiral)
        v_mode.addWidget(self.rb_zigzag)
        grp_mode.setLayout(v_mode)
        panel_layout.addWidget(grp_mode)

        # 2. 通用
        grp_common = QGroupBox("通用参数")
        f_common = QFormLayout()
        self.inp_step = QLineEdit("10.0")
        self.inp_z_thresh = QLineEdit("0.2")
        f_common.addRow("采样步长 (mm):", self.inp_step)
        f_common.addRow("法向Z阈值 (0-1):", self.inp_z_thresh)
        grp_common.setLayout(f_common)
        panel_layout.addWidget(grp_common)

        # 3. 螺旋
        self.grp_spiral = QGroupBox("螺旋参数")
        f_spiral = QFormLayout()
        self.inp_radius = QLineEdit("150.0")
        self.chk_auto_center = QCheckBox("自动计算中心")
        self.chk_auto_center.setChecked(True)
        self.chk_auto_center.toggled.connect(self.update_ui_state)
        self.inp_cx = QLineEdit("0.0")
        self.inp_cy = QLineEdit("0.0")
        f_spiral.addRow("最大半径 (mm):", self.inp_radius)
        f_spiral.addRow(self.chk_auto_center)
        f_spiral.addRow("中心 X:", self.inp_cx)
        f_spiral.addRow("中心 Y:", self.inp_cy)
        self.grp_spiral.setLayout(f_spiral)
        panel_layout.addWidget(self.grp_spiral)

        # 4. ROI
        grp_roi = QGroupBox("ROI 区域限制")
        g_roi = QFormLayout()
        self.inp_xmin = QLineEdit()
        self.inp_xmax = QLineEdit()
        self.inp_ymin = QLineEdit()
        self.inp_ymax = QLineEdit()
        self.inp_zmin = QLineEdit()
        self.inp_zmax = QLineEdit()

        for w in [self.inp_xmin, self.inp_xmax, self.inp_ymin, self.inp_ymax, self.inp_zmin, self.inp_zmax]:
            w.setPlaceholderText("不限")

        h_x = QHBoxLayout()
        h_x.addWidget(QLabel("X:"))
        h_x.addWidget(self.inp_xmin)
        h_x.addWidget(QLabel("~"))
        h_x.addWidget(self.inp_xmax)
        h_y = QHBoxLayout()
        h_y.addWidget(QLabel("Y:"))
        h_y.addWidget(self.inp_ymin)
        h_y.addWidget(QLabel("~"))
        h_y.addWidget(self.inp_ymax)
        h_z = QHBoxLayout()
        h_z.addWidget(QLabel("Z:"))
        h_z.addWidget(self.inp_zmin)
        h_z.addWidget(QLabel("~"))
        h_z.addWidget(self.inp_zmax)

        g_roi.addRow(h_x)
        g_roi.addRow(h_y)
        g_roi.addRow(h_z)
        grp_roi.setLayout(g_roi)
        panel_layout.addWidget(grp_roi)

        # 5. 按钮
        btn_update = QPushButton("生成路径 / 更新预览")
        btn_update.clicked.connect(self.generate_path)
        btn_update.setFixedHeight(40)

        btn_save = QPushButton("保存路径 (.npz)")
        btn_save.clicked.connect(self.save_path)
        btn_save.setStyleSheet("background-color: #2ea043;")

        panel_layout.addStretch()
        panel_layout.addWidget(btn_update)
        panel_layout.addWidget(btn_save)

        lbl_hint = QLabel("交互说明:\n[鼠标左键] 旋转\n[鼠标右键] 平移\n[滚轮] 缩放")
        lbl_hint.setStyleSheet("color: #4facfe; font-weight: bold; padding: 5px;")
        panel_layout.addWidget(lbl_hint)

        main_layout.addWidget(control_panel)

        # 右侧 3D 视图
        self.plotter = QtInteractor(central_widget)
        self.plotter.set_background("#e1e1e1")
        main_layout.addWidget(self.plotter)

        self.update_ui_state()

    def update_ui_state(self):
        is_spiral = self.rb_spiral.isChecked()
        self.grp_spiral.setVisible(is_spiral)
        is_auto = self.chk_auto_center.isChecked()
        self.inp_cx.setEnabled(not is_auto)
        self.inp_cy.setEnabled(not is_auto)

    def load_model(self):
        if not os.path.exists(self.stl_path):
            QMessageBox.critical(self, "错误", f"找不到文件: {self.stl_path}")
            return

        try:
            self.trimesh_obj = curve_utils.CurvePathPlanner.ensure_single_mesh(trimesh.load(self.stl_path))
            self.trimesh_obj.apply_scale(1000.0)  # 模型单位转换至mm
            self.pv_mesh = pv.read(self.stl_path)
            self.pv_mesh.points *= 1000.0    # mesh单位转换至mm


            c = self.trimesh_obj.bounding_box.centroid
            self.inp_cx.setText(f"{c[0]:.2f}")
            self.inp_cy.setText(f"{c[1]:.2f}")

            self.plotter.clear()
            self.plotter.add_mesh(self.pv_mesh, color="white", opacity=0.6, show_edges=False, smooth_shading=True)

            try:
                self.plotter.add_camera_orientation_widget()
            except AttributeError:
                self.plotter.add_axes()

            self.plotter.add_mesh(pv.Box(self.pv_mesh.bounds), color='grey', style='wireframe', opacity=0.3)
            self.plotter.view_isometric()

            style = RobotPathInteractorStyle()
            self.plotter.iren.interactor.SetInteractorStyle(style)

        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))

    def get_float(self, line_edit, default=None):
        try:
            txt = line_edit.text()
            if not txt and default is not None: return default
            return float(txt)
        except:
            return default

    def get_roi_dict(self):
        return {
            'x': {'min': self.get_float(self.inp_xmin), 'max': self.get_float(self.inp_xmax)},
            'y': {'min': self.get_float(self.inp_ymin), 'max': self.get_float(self.inp_ymax)},
            'z': {'min': self.get_float(self.inp_zmin), 'max': self.get_float(self.inp_zmax)}
        }

    # 辅助函数：计算表面点
    def get_surface_point(self, x, y):
        # 获取 (x,y) 对应的 mesh 表面 Z 高度
        ray_origin = np.array([[x, y, self.trimesh_obj.bounds[1, 2] + 50]])
        ray_dir = np.array([[0, 0, -1]])
        locs, _, _ = self.trimesh_obj.ray.intersects_location(ray_origin, ray_dir, multiple_hits=False)
        if len(locs) > 0:
            return locs[0]
        return np.array([x, y, self.trimesh_obj.bounds[1, 2]])

    def generate_path(self):
        if self.trimesh_obj is None: return

        cam_pos = self.plotter.camera.position
        cam_focal = self.plotter.camera.focal_point
        cam_up = self.plotter.camera.up

        self.plotter.clear()
        self.plotter.add_mesh(self.pv_mesh, color="white", opacity=0.4, show_edges=False, smooth_shading=True)
        try:
            self.plotter.add_camera_orientation_widget()
        except:
            self.plotter.add_axes()

        step = self.get_float(self.inp_step, 10.0)
        z_thresh = self.get_float(self.inp_z_thresh, 0.2)

        points, normals = np.array([]), np.array([])

        if self.rb_spiral.isChecked():
            radius = self.get_float(self.inp_radius, 150.0)
            if self.chk_auto_center.isChecked():
                c = self.trimesh_obj.bounding_box.centroid
                cx, cy = c[0], c[1]
                self.inp_cx.setText(f"{cx:.2f}")
                self.inp_cy.setText(f"{cy:.2f}")
            else:
                cx = self.get_float(self.inp_cx, 0.0)
                cy = self.get_float(self.inp_cy, 0.0)

            # 1：红点贴合表面
            center_pt = self.get_surface_point(cx, cy)
            # 稍微抬高防止Z-fighting
            display_pt = center_pt + np.array([0, 0, 0.5])
            self.plotter.add_points(display_pt, color="red", point_size=15, render_points_as_spheres=True,
                                    label="Center")

            points, normals = curve_utils.CurvePathPlanner.compute_spiral_3d(
                self.trimesh_obj, cx, cy, radius, step, z_thresh
            )
        else:
            points, normals = curve_utils.CurvePathPlanner.generate_zigzag_path(
                self.trimesh_obj, step, z_thresh
            )

        if len(points) > 0:
            roi_dict = self.get_roi_dict()
            points, normals = curve_utils.CurvePathPlanner.filter_by_roi(points, normals, roi_dict)

            b = self.trimesh_obj.bounds
            bounds = [
                roi_dict['x']['min'] if roi_dict['x']['min'] is not None else b[0, 0],
                roi_dict['x']['max'] if roi_dict['x']['max'] is not None else b[1, 0],
                roi_dict['y']['min'] if roi_dict['y']['min'] is not None else b[0, 1],
                roi_dict['y']['max'] if roi_dict['y']['max'] is not None else b[1, 1],
                roi_dict['z']['min'] if roi_dict['z']['min'] is not None else b[0, 2],
                roi_dict['z']['max'] if roi_dict['z']['max'] is not None else b[1, 2]
            ]
            self.plotter.add_mesh(pv.Box(bounds), color="green", style='wireframe', opacity=0.3, line_width=2)

        self.current_points = points
        self.current_normals = normals

        if len(points) > 0:
            # 1. 路径线
            line_poly = pv.lines_from_points(points)
            line_poly["scalars"] = np.arange(line_poly.n_points)
            self.plotter.add_mesh(line_poly, cmap="turbo", line_width=3, show_scalar_bar=False)

            # 2：采样点
            # 使用 points 的 Z 轴高度作为颜色映射依据，类似 V4 的 scatter(c=Z, cmap='viridis')
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
        QMessageBox.information(self, "保存成功", f"文件已保存至:\n{os.path.abspath(filename)}")


if __name__ == '__main__':
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())