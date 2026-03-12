import sys
import pyaubo_sdk
import time
import math
import threading

robot_ip = "127.0.0.1"  # 机器人IP地址
rpc_port = 30004  # RPC端口号
rtde_port = 30010  # RTDE端口号

rtde_mtx_ = threading.Lock()  # 互斥锁

line_number_ = -1  # 行号


def callback(parser):
    global line_number_
    rtde_mtx_.acquire()
    line_number_ = parser.popInt32()
    rtde_mtx_.release()


if __name__ == '__main__':
    rpc_client = pyaubo_sdk.RpcClient()  # RPC客户端
    rpc_client.setRequestTimeout(1000)  # 接口调用: 设置 RPC 超时
    rpc_client.connect(robot_ip, rpc_port)  # 接口调用: 连接 RPC 服务
    rpc_client.login("aubo", "123456")  # 接口调用: 登录

    rtde_client = pyaubo_sdk.RtdeClient()  # RTDE客户端
    rtde_client.connect(robot_ip, rtde_port)  # 接口调用: 连接 RTDE 服务
    rtde_client.login("aubo", "123456")  # 接口调用: 登录

    # 设置话题
    chanel = rtde_client.setTopic(False, ["line_number"], 50, 1)
    if chanel == -1:
        print(f"setTopic失败！setTopic返回值为{chanel}")
        sys.exit(1)

    # 订阅话题
    rtde_client.subscribe(chanel, callback)

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

    # 接口调用：设置运动速度比例
    mc.setSpeedFraction(1)

    # 接口调用: 开启运行时
    rm.start()
    # 增加延时，等待新建线程完成后，
    # 再调用getPlanContext，
    # 确保获取的当前线程号非-1
    time.sleep(0.05)

    # 获取当前的线程号
    tid = rm.getPlanContext(-1)[0]
    # print(f"{tid}")

    for index, q in enumerate(traj_q_list):
        if index == len(traj_q_list) - 1:
            line_num = -1
            comment = f"运动结束"
        else:
            line_num = index
            comment = f"路点{index}"
        # 设置运行时上下文
        rm.setPlanContext(tid, line_num, comment)  
        # 关节运动
        while 2 == mc.moveJoint(q, 80 / 180 * math.pi, 60 / 180 * math.pi, 0, 0):
            time.sleep(0.005)

    # 获取下发的最后一个运动指令的目标运行时上下文
    final_plan_context = rm.getAdvancePlanContext(-1)

    # RTDE订阅 line_number 更新数据有延时，
    # line_number_ 初始默认值为-1，
    # 故这里增加延时
    while line_number_ == -1:
        time.sleep(0.005)

    while True:
        # 轨迹中的倒数第二组和最后一组路点相同
        # 若当前行号等于 moveJoint 发送最后一组路点前 setPlanContext 设定的行号，
        # 则表示机械臂已执行完所有轨迹路点
        if line_number_ == final_plan_context[1]:
            print(f"{final_plan_context[2]}")
            # 停止运行时
            rm.abort()
            break
        print(f"当前向路点{line_number_}运动")
        time.sleep(0.005)
