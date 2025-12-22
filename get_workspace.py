import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import random


def load_model(xml_name="aubo_i5_withdetector.xml"):
    # 兼容两种路径查找
    paths = [xml_name, os.path.join("mjcf", xml_name)]
    for path in paths:
        if os.path.exists(path):
            try:
                print(f"正在加载模型: {path}")
                return mujoco.MjModel.from_xml_path(path)
            except Exception as e:
                print(f"加载失败 ({e})")
    print(f"未找到 {xml_name}")
    return None


def is_in_feasible_workspace(model, data, min_z=0.05):
    """
    判断当前姿态是否在“可行工作空间”内。
    这里的定义是：末端探测器的高度必须大于 min_z。
    如果不满足，说明这个姿态本身就不在我们的考虑范围内（比如太低了）。
    """
    # 必须先更新运动学以获取xpos
    mujoco.mj_kinematics(model, data)

    target_body = "detector_body"
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body)

    if body_id != -1:
        z_height = data.xpos[body_id][2]
        # 如果高度低于限制，视为“不可行空间”，直接返回False
        if z_height < min_z:
            return False, z_height
        return True, z_height

    return True, 0.0  # 如果找不到body，默认可行


def check_physical_collision(model, data):
    """
    纯粹的物理碰撞检测。
    只在姿态已经确定“可行”之后调用。
    """
    mujoco.mj_collision(model, data)

    collisions = []
    for i in range(data.ncon):
        contact = data.contact[i]

        geom1, geom2 = contact.geom1, contact.geom2
        body1, body2 = model.geom_bodyid[geom1], model.geom_bodyid[geom2]

        if body1 == body2: continue

        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1)
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2)

        # 排除与世界/地面的碰撞（因为我们已经用高度限制过滤了地面问题）
        # 如果你希望检测“撞到地面”也算碰撞，可以去掉 'floor'
        if name1 in ['world', 'floor'] or name2 in ['world', 'floor']: continue

        # 排除相邻父子物体的碰撞
        p1, p2 = model.body_parentid[body1], model.body_parentid[body2]
        if p1 == body2 or p2 == body1: continue

        pair_str = f"{name1} <-> {name2}"
        if pair_str not in collisions:
            collisions.append(pair_str)

    if len(collisions) > 0:
        return True, ", ".join(collisions)
    return False, None


def search_valid_collisions(n_attempts=5000, min_height=0.05):
    model = load_model()
    if model is None: return None, None, []
    data = mujoco.MjData(model)

    collision_cases = []

    valid_workspace_count = 0

    print(f"开始搜索...")
    print(f"过滤条件: 探测器高度必须 > {min_height}m")
    print(f"目标: 在满足上述高度的姿态中，寻找发生物理碰撞的案例。")

    for _ in range(n_attempts):
        # 1. 随机采样
        qpos = []
        for i in range(model.nq):
            limit_min, limit_max = model.jnt_range[i]
            # 0.98 稍微留一点余量，避免由于数值误差在极限位置误报
            qpos.append(np.random.uniform(limit_min, limit_max) * 0.95)

        data.qpos[:] = np.array(qpos)

        # 2. 【筛选】检查是否在可行工作空间 (高度限制)
        in_workspace, z_val = is_in_feasible_workspace(model, data, min_z=min_height)

        if not in_workspace:
            # 如果不在工作空间（比如太低了），直接跳过！
            # 我们不关心它是否碰撞，因为它根本不是我们想要的作业姿态
            continue

        valid_workspace_count += 1

        # 3. 【检测】在可行空间内，检查是否有物理碰撞
        has_collision, info = check_physical_collision(model, data)

        if has_collision:
            collision_cases.append({
                'qpos': data.qpos.copy(),
                'info': info,
                'z_height': z_val
            })

    print("-" * 30)
    print(f"搜索统计:")
    print(f"  总尝试采样数: {n_attempts}")
    print(f"  落在可行高度内的姿态数: {valid_workspace_count}")
    print(f"  在可行高度内发生的碰撞数: {len(collision_cases)}")
    if valid_workspace_count > 0:
        rate = len(collision_cases) / valid_workspace_count * 100
        print(f"  可行空间内的碰撞率: {rate:.2f}%")
    print("-" * 30)

    return model, data, collision_cases


def replay_collisions(model, data, cases):
    if not cases:
        print("没有找到符合条件的碰撞案例。")
        return

    print(f"\n即将回放 {len(cases)} 个案例...")
    print("这些案例的特点：高度是合法的(>Limit)，但发生了物理碰撞。")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 随机展示最多 15 个
        display_cases = cases if len(cases) < 15 else random.sample(cases, 15)

        for i, case in enumerate(display_cases):
            if not viewer.is_running(): break

            data.qpos[:] = case['qpos']
            mujoco.mj_forward(model, data)

            print(f"[{i + 1}] 高度: {case['z_height']:.3f}m | 碰撞: {case['info']}")

            start = time.time()
            while viewer.is_running() and time.time() - start < 10.0:
                viewer.sync()
                time.sleep(0.01)


if __name__ == "__main__":
    # 设定高度限制为 0.1 米 (比基座高 10cm)
    # 只有高于 10cm 的姿态才会被检查是否碰撞
    model, data, records = search_valid_collisions(n_attempts=50000, min_height=0.10)

    if records:
        replay_collisions(model, data, records)