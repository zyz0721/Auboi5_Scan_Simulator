import trimesh
import numpy as np
import os
import math

INPUT_FILE = 'models/sample.STL'
OUTPUT_FILE = 'models/sample_fixed.stl'

# 1. 缩放因子
# 0.001 表示将毫米转换为米 (1/1000)，1.0 表示不缩放
SCALE_FACTOR = 1.0

# 2. 旋转角度 (单位：度)
# 正数表示逆时针旋转，负数表示顺时针旋转
ROTATE_YAW_DEG = 90.0


def process_stl():
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到文件 {INPUT_FILE}")
        return

    print(f"正在读取 {INPUT_FILE} ...")
    mesh = trimesh.load(INPUT_FILE)

    # 1: 打印原始尺寸
    extents = mesh.extents
    print(f"\n[原始] 模型尺寸 (X, Y, Z):")
    print(f"  {extents[0]:.4f} x {extents[1]:.4f} x {extents[2]:.4f}")

    # 安全检查：防止重复缩小
    if SCALE_FACTOR < 0.1 and np.max(extents) < 5.0:
        print("\n警告: 模型看起来已经是米单位了 (尺寸很小)。")
        print(f"你确定还要执行 {SCALE_FACTOR} 倍的缩放吗？")
        confirm = input("输入 'y' 继续，其他键跳过缩放仅旋转: ")
        if confirm.lower() != 'y':
            print("已跳过缩放操作。")
        else:
            print(f"执行缩放: {SCALE_FACTOR}")
            mesh.apply_scale(SCALE_FACTOR)
    else:
        print(f"执行缩放: {SCALE_FACTOR}")
        mesh.apply_scale(SCALE_FACTOR)

    # 2: 执行旋转
    if abs(ROTATE_YAW_DEG) > 0.001:
        print(f"\n执行旋转: 绕 Z 轴 {ROTATE_YAW_DEG} 度")

        # 将角度转换为弧度
        theta = np.radians(ROTATE_YAW_DEG)

        # 构建绕 Z 轴的 4x4 旋转矩阵
        # [ cos -sin   0   0 ]
        # [ sin  cos   0   0 ]
        # [   0    0   1   0 ]
        # [   0    0   0   1 ]
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.eye(4)
        rotation_matrix[0, 0] = c
        rotation_matrix[0, 1] = -s
        rotation_matrix[1, 0] = s
        rotation_matrix[1, 1] = c

        # 应用变换
        mesh.apply_transform(rotation_matrix)
    else:
        print("\n旋转角度为 0，跳过旋转。")

    # 3: 打印最终结果并保存
    new_extents = mesh.extents
    print(f"\n[最终] 模型尺寸 (X, Y, Z):")
    print(f"  {new_extents[0]:.4f} x {new_extents[1]:.4f} x {new_extents[2]:.4f}")

    # 可选：将模型底部移动到 Z=0 (如果旋转后位置跑偏了)
    # mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

    mesh.export(OUTPUT_FILE)
    print(f"\n成功保存至: {OUTPUT_FILE}")
    print(f"包含操作: 缩放 x{SCALE_FACTOR}, 旋转 {ROTATE_YAW_DEG}°")


if __name__ == '__main__':
    process_stl()