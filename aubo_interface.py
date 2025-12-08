import socket
import time
import struct
import logging

"""
==============================================================================
Aubo i5 机械臂 Python 控制接口 (TCP/IP Dashboard 协议)
==============================================================================

[连接指南 / Connection Guide]

1. 硬件连接 (Hardware):
   - 使用网线将电脑的网口连接到 Aubo 控制柜底部的【Ethernet 接口】。
   - 注意：不要连接到 EtherCAT 口，那是给内部总线用的。

2. 网络配置 (Network Config) - Windows 11:
   - 机械臂端 (示教器):
     进入【设置】->【系统】->【网络】。
     查看或设置 IP 地址，例如: 192.168.1.10 (子网掩码 255.255.255.0)

   - 电脑端 (PC):
     进入 Windows 设置 -> 网络和 Internet -> 以太网 -> IP分配(编辑) -> 手动 IPv4。
     IP 地址:   192.168.1.100 (注意：前三段必须与机械臂一致，最后一段不能相同)
     子网掩码: 255.255.255.0
     网关:     留空 或 192.168.1.1

   - 测试: 打开 cmd 输入 `ping 192.168.1.10`，必须能通才能运行代码。

3. 机械臂状态准备 (Robot Status):
   - 在示教器上点击【初始化】，确保所有关节刹车已松开 (状态变绿，听到咔哒声)。
   - 确保机械臂处于【自动模式】。
   - 部分旧版本固件可能需要在 设置->扩展->远程控制 中勾选【启用 TCP 服务器】。

4. 端口说明:
   - Port 8899: Dashboard 接口，用于发送字符串指令 (movej, movel, etc.)，本代码使用此端口。
   - Port 30004: 实时反馈端口，用于高频读取电流、电压、精确位姿等。

==============================================================================
"""
IP = "192.168.1.10"


class AuboRealRobot:
    def __init__(self, IP, dashboard_port=8899):
        """
        初始化机械臂控制对象
        :param ip: 机械臂的 IP 地址 (默认为 192.168.1.10)
        :param dashboard_port: 控制端口 (默认为 8899)
        """
        self.ip = IP
        self.port = dashboard_port
        self.sock = None
        self.connected = False
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("AuboRobot")

    def connect(self):
        """
        建立 TCP 连接
        """
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(3.0)  # 设置3秒超时
            self.sock.connect((self.ip, self.port))
            self.connected = True
            self.logger.info(f"成功连接到机械臂: {self.ip}:{self.port}")
            return True
        except Exception as e:
            self.logger.error(f"连接失败: {e}")
            self.logger.error("请检查: 1.网线是否插好 2.电脑IP是否设置正确(192.168.1.x) 3.能否Ping通机械臂")
            self.connected = False
            return False

    def disconnect(self):
        """
        断开 TCP 连接
        """
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        self.connected = False
        self.logger.info("连接已断开")

    def send_command(self, cmd_str):
        """
        发送字符串指令到 Dashboard 服务器
        """
        if not self.connected:
            self.logger.warning("未连接，无法发送指令")
            return None
        try:
            # 官方协议通常要求以换行符结尾
            if not cmd_str.endswith('\n'):
                msg = cmd_str + "\n"
            else:
                msg = cmd_str

            self.sock.sendall(msg.encode('utf-8'))

            # 接收回执 (非阻塞读取或简易读取)
            try:
                data = self.sock.recv(1024)
                response = data.decode('utf-8').strip()
                # self.logger.debug(f"发送: {cmd_str.strip()} | 回复: {response}")
                return response
            except socket.timeout:
                return None
        except Exception as e:
            self.logger.error(f"发送指令错误: {e}")
            return None

    def move_j(self, joint_angles_rad, v=0.5, a=0.5):
        """
        关节空间运动 (MoveJ) - 最常用的点到点移动
        :param joint_angles_rad: 包含6个关节角的列表/数组，单位必须是【弧度】
        :param v: 关节角速度 (rad/s)
        :param a: 关节角加速度 (rad/s^2)
        """
        if len(joint_angles_rad) != 6:
            self.logger.error("关节数据长度必须为6")
            return

        # 构造官方脚本指令格式: movej([j1,j2,j3,j4,j5,j6], a, v)
        # 注意保留4位小数即可
        joints_str = ",".join([f"{j:.4f}" for j in joint_angles_rad])
        cmd = f"movej([{joints_str}], {a}, {v})"

        self.logger.info(f"执行运动: {cmd}")
        return self.send_command(cmd)

    def move_l(self, pose, v=0.2, a=0.2):
        """
        直线运动 (MoveL) - 末端走直线
        :param pose: [x, y, z, r, p, y] 单位: 米 和 弧度
        """
        # 这一步需要根据 Aubo 具体版本确认是发四元数还是欧拉角
        # 这里仅作示例结构，通常推荐优先用 move_j 防止奇异点
        pass

    def get_current_joints(self):
        """
        查询当前机械臂关节角度 (模拟查询)
        注意：Dashboard端口主要用于发指令。
        如果需要高频(100Hz+)实时读取，建议另开线程连接 30004 端口解析二进制包。
        """
        # 尝试发送查询指令 (指令视固件版本可能不同，如 'get_actual_joint_positions')
        resp = self.send_command("get_actual_joint_positions")
        if resp:
            try:
                # 假设返回格式为字符串 "[0.1, 0.2, ...]" 或 "0.1, 0.2, ..."
                # 清洗字符串
                clean_resp = resp.replace("[", "").replace("]", "").replace(";", ",")
                joints = [float(x) for x in clean_resp.split(",") if x.strip()]
                if len(joints) == 6:
                    return joints
            except:
                pass
        return None


# ==========================================
# 单元测试 (Unit Test)
# ==========================================
if __name__ == "__main__":
    print("--- 开始测试 Aubo 接口 ---")

    # 1. 设置为您机械臂的实际 IP
    ROBOT_IP = "192.168.1.10"

    robot = AuboRealRobot(ROBOT_IP)

    if robot.connect():
        print("连接成功!")

        # 获取当前位置
        joints = robot.get_current_joints()
        if joints:
            print(f"当前关节角: {joints}")
        else:
            print("获取关节角失败 (可能是指令不匹配或未处于远程模式)")

        # 危险操作：移动测试 (请确保周围无人且手在急停上)
        # target_joints = [0, 0, 0, 0, 0, 0] # 零位
        # robot.move_j(target_joints, v=0.1, a=0.1)

        robot.disconnect()
    else:
        print("连接失败。请检查网线和IP设置。")