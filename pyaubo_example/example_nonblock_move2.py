import pyaubo_sdk
import time
import math

robot_ip = "127.0.0.1"  # 机器人IP地址
rpc_port = 30004  # RPC端口号

if __name__ == '__main__':
    rpc_client = pyaubo_sdk.RpcClient()  # RPC客户端
    rpc_client.setRequestTimeout(1000)  # 接口调用: 设置 RPC 超时
    rpc_client.connect(robot_ip, rpc_port)  # 接口调用: 连接 RPC 服务
    rpc_client.login("aubo", "123456")  # 接口调用: 登录

    # 轨迹列表
    traj_q_list = [
        [angle * (math.pi / 180) for angle in angles]
        for angles in [
            [0.0, -15.0, 100.0, 25.0, 90.0, 0.0],
            [35.92, -11.28, 59.96, -18.76, 90.0, 35.92],
            [41.04, -7.65, 98.80, 16.44, 90.0, 11.64]
        ]
    ]

    # 重复添加轨迹的最后一组路点
    traj_q_list.append(traj_q_list[-1])

    # 接口调用: 获取机器人的名字
    robot_name = rpc_client.getRobotNames()[0]
    robot_interface = rpc_client.getRobotInterface(robot_name)
    rm = rpc_client.getRuntimeMachine()
    mc = robot_interface.getMotionControl()

    # 接口调用： 设置运动速度比例
    mc.setSpeedFraction(1)

    # 接口调用: 开启运行时
    rm.start()

    # 增加延时，等待新建线程完成后，
    # 再调用moveJoint，
    # 防止moveJoint被discard
    time.sleep(0.05)

    for q in traj_q_list:
        # 关节运动
        while 2 == mc.moveJoint(q, 80 / 180 * math.pi, 60 / 180 * math.pi, 0, 0):
            time.sleep(0.005)

    # 获取下发的最后一个运动指令的目标运动指针
    final_ptr = rm.getAdvancePtr(-1)
    print(f"目标运动指针：{final_ptr}")

    while True:
        # 轨迹中的倒数第二组和最后一组路点相同
        # 若当前运动指针等于 moveJoint 下发的最后一个运动指令的目标运动指针时，
        # 则表示机械臂执行完所有的轨迹路点
        if rm.getMainPtr(-1) == final_ptr:
            print(f"运动结束")
            # 停止运行时
            rm.abort()
            break
        print(f"当前运动指针:{rm.getMainPtr(-1)}")
        time.sleep(0.005)
