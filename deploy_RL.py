import time
import numpy as np
import math
from stable_baselines3 import PPO

# 引入之前的工具文件
from aubo_interface import AuboRealRobot
from casadi_ik import Kinematics
from curve_utils import CurvePathPlanner


def get_real_observation(robot, target_pos, last_q, dt):
    """
    构造与训练环境(dm_rl_env)完全一致的观测向量
    Obs 顺序: [qpos(6), qvel(6), ee_pos(3), tracking_error(3)]
    """
    # 1. 获取关节角 qpos
    q_now = robot.get_current_joints()
    if q_now is None:
        q_now = last_q  # 读不到就用上一次的

    # 2. 计算关节速度 qvel (简单的差分)
    # 训练中 MuJoCo 直接给速度，实机通常噪声大，差分法足够用
    q_vel = [(q - lq) / dt for q, lq in zip(q_now, last_q)]

    # 3. 获取末端位置 ee_pos (FK)
    # 这里我们利用 casadi_ik 的内部正运动学，或者使用 robot SDK 的正解
    # 为了简单，这里假设 robot.robot_interface 提供了正解
    # 如果没有，可以使用 ik_solver.fk(q_now) (需要你在 CasadiIK 里加个 forward 方法)
    # 暂时使用 SDK 方法 (注意单位转换):
    waypoint = robot.robot_interface.getRobotState().getWayPoint()
    ee_pos = [waypoint['x'], waypoint['y'], waypoint['z']]

    # 4. 计算追踪误差
    tracking_error = np.array(target_pos) - np.array(ee_pos)

    # 拼装 (必须转为 float32)
    obs = np.concatenate([
        q_now,
        q_vel,
        ee_pos,
        tracking_error
    ]).astype(np.float32)

    return obs, q_now


def main():
    # ================= 配置 =================
    model_path = "aubo_scan_final_policy.zip"  # 训练好的模型
    robot_ip = "192.168.1.10"
    control_freq = 50.0  # Hz
    dt = 1.0 / control_freq
    action_scale = 0.005  # 必须与 dm_rl_env.py 中保持一致！
    # =======================================

    # 1. 加载模型
    print(f"正在加载模型: {model_path} ...")
    model = PPO.load(model_path)

    # 2. 初始化硬件和算法
    robot = AuboRealRobot(robot_ip)
    ik_solver = Kinematics(ee_frame="wrist3_Link")  # 确保和训练用的 IK 一致
    curve_manager = CurvePathPlanner()  # 你的路径生成器

    if not robot.connect_and_startup():
        print("机械臂连接失败")
        return

    try:
        # 3. 生成扫描路径 (Base Path)
        # 假设 generate_path 返回 [[x,y,z,rx,ry,rz], ...]
        # 这里用模拟数据代替，实际请调用 curve_manager
        path_points = []
        start_pos = np.array([0.4, -0.2, 0.4])
        for t in np.linspace(0, 1, 200):  # 4秒轨迹
            pos = start_pos + np.array([0, t * 0.2, 0])
            rot = np.array([3.14, 0, 0])
            path_points.append(np.concatenate([pos, rot]))

        # 4. 移动到起始点 (MoveJ)
        print("移动到起始点...")
        start_pose = path_points[0]
        # 注意：这里需要先把 Euler 转为 IK 需要的格式，参考 dm_rl_env.py
        # 简单起见假设 ik_solver.ik 支持直接传 6维
        q_start = ik_solver.ik(start_pose, robot.get_current_joints())
        if q_start:
            robot.move_j(q_start)
        else:
            print("起始点 IK 无解")
            return

        # 5. 进入伺服模式
        if not robot.enter_servo_mode():
            return

        print("开始 RL 辅助扫描...")
        last_q = robot.get_current_joints()

        # --- 主控制循环 ---
        for target_pose_6d in path_points:
            loop_start = time.time()

            target_pos = target_pose_6d[:3]

            # A. 构造 Observation
            # RL 需要根据当前状态决定怎么微调
            obs, current_q = get_real_observation(robot, target_pos, last_q, dt)
            last_q = current_q

            # B. 模型推理 (Inference)
            # deterministic=True 表示使用训练好的最优策略，不要随机探索
            action, _ = model.predict(obs, deterministic=True)

            # C. 计算最终指令
            # 1. 计算基准 IK (Base)
            q_base = ik_solver.ik(target_pose_6d, current_q)
            if q_base is None: q_base = current_q

            # 2. 应用 RL 残差
            # action 是 [-1, 1]，乘以 scale 变成实际位移 (米)
            residual = action[:3] * action_scale

            # 3. 修正目标位置
            final_target_pos = target_pose_6d[:3] + residual
            final_target_pose_6d = np.concatenate([final_target_pos, target_pose_6d[3:]])

            # 4. 最终 IK
            q_final = ik_solver.ik(final_target_pose_6d, q_base)
            if q_final is None: q_final = q_base

            # D. 发送给机械臂
            # 处理缓冲区满的情况
            while True:
                ret = robot.send_servo_point(q_final, dt)
                if ret == 0: break  # 成功
                if ret == 2:
                    time.sleep(0.002)  # 满，等一会
                else:
                    break  # 错误

            # E. 频率控制
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("用户停止")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        robot.stop()
        robot.disconnect()


if __name__ == "__main__":
    main()
