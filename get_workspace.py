import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import random


def load_model(xml_name="mjcf/aubo_i5_withdetector.xml"):
    if os.path.exists(xml_name):
        try:
            print(f"正在加载本地模型: {xml_name}")
            return mujoco.MjModel.from_xml_path(xml_name)
        except Exception as e:
            print(f"加载失败 ({e})")
    else:
        print(f"未找到 {xml_name}")


def check_collision_details(model, data):
    """
    返回: (是否碰撞, 碰撞信息字符串)
    """
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)

    collisions = []
    for i in range(data.ncon):
        contact = data.contact[i]

        # 获取几何体ID和Body ID
        geom1 = contact.geom1
        geom2 = contact.geom2
        body1 = model.geom_bodyid[geom1]
        body2 = model.geom_bodyid[geom2]

        # 排除同体和地面
        if body1 == body2: continue
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1)
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2)
        if name1 in ['world', 'floor'] or name2 in ['world', 'floor']: continue

        # 排除相邻 (简单的父子判断)
        p1 = model.body_parentid[body1]
        p2 = model.body_parentid[body2]
        if p1 == body2 or p2 == body1: continue

        # 记录碰撞对
        pair_str = f"{name1} <-> {name2}"
        if pair_str not in collisions:
            collisions.append(pair_str)

    if len(collisions) > 0:
        return True, ", ".join(collisions)
    return False, None


def collect_collision_cases(n_samples=2000):
    model = load_model()
    data = mujoco.MjData(model)

    collision_cases = []  # 存储结构: {'qpos': np.array, 'info': str}

    print(f"正在采样 {n_samples} 个点以寻找碰撞案例...")

    for _ in range(n_samples):
        # 随机采样 (使用 0.95 避免极端边缘)
        qpos = []
        for i in range(model.nq):
            limit_min = model.jnt_range[i][0]
            limit_max = model.jnt_range[i][1]
            qpos.append(np.random.uniform(limit_min, limit_max) * 0.95)

        data.qpos[:] = np.array(qpos)

        is_col, info = check_collision_details(model, data)

        if is_col:
            # 必须使用 .copy()，否则存的是引用，数据会变
            collision_cases.append({
                'qpos': data.qpos.copy(),
                'info': info
            })

    return model, data, collision_cases


def replay_collisions(model, data, cases):
    if not cases:
        print("未检测到任何碰撞，无法回放。")
        return

    print("\n" + "=" * 40)
    print(f"开始回放！共找到 {len(cases)} 个碰撞姿态。")
    print("查看器将每隔 2 秒切换一个姿态。")
    print("请在弹出的 MuJoCo 窗口中观察红色接触点。")
    print("=" * 40 + "\n")

    # 启动查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 为了演示，我们只随机抽取最多 10 个典型案例展示，避免无限循环
        display_cases = cases if len(cases) < 10 else random.sample(cases, 10)

        for i, case in enumerate(display_cases):
            if not viewer.is_running():
                break

            # 1. 设置姿态
            data.qpos[:] = case['qpos']

            # 2. 必须调用 forward 来更新几何体位置和计算接触点
            mujoco.mj_forward(model, data)

            # 3. 打印信息
            print(f"[{i + 1}/{len(display_cases)}] 当前碰撞: {case['info']}")
            print(f"    关节角: {np.round(case['qpos'], 2)}")

            # 4. 保持这个画面 2.5 秒
            start_time = time.time()
            while viewer.is_running() and time.time() - start_time < 10.0:
                viewer.sync()
                time.sleep(0.01)

    print("\n回放结束。")


if __name__ == "__main__":
    # 1. 采集数据
    model, data, collision_records = collect_collision_cases(n_samples=10000)

    print(f"采样结束，自干涉率: {len(collision_records) / 1000 * 100:.1f}%")

    # 2. 启动回放 (如果找到了碰撞)
    if len(collision_records) > 0:
        replay_collisions(model, data, collision_records)