#! /usr/bin/env python
# coding=utf-8

"""
serial类 串口读写示例

步骤:
第一步: 连接到 RPC 服务
第二步: 打开串口
第三步: 向串口发送字符串
第四步: 向串口发送字符串数组
第五步: 向串口发送带有换行符的字符串
第六步: 从串口读取字符串
第七步: 向串口发送字节
第八步: 从串口监控单字节
第九步: 向串口发送整数
第十步: 从串口监控指定数量字节列表
第十一步: 关闭串口
第十二步: 断开 RPC 连接
"""

import time
import pyaubo_sdk

robot_ip = "127.0.0.1"  # 服务器 IP 地址
robot_port = 30004  # 端口号
robot_rpc_client = pyaubo_sdk.RpcClient()
serial_port = robot_rpc_client.getSerial()
rc = robot_rpc_client.getRegisterControl()
serial_dev = "/dev/ttyS0"
baud_rate = 115200
stop_bits = 1.0
parity = 0
serial_alias = "serial_0"
timeout = 5.0
response = -1


def exampleOpenSerial() -> bool:
    if (
        serial_port.serialOpen(
            serial_dev,
            baud_rate,
            stop_bits,
            parity,
            serial_alias,
        )
        != 0
    ):
        print("[ERROR] 串口打开失败")
        return False
    print("[SUCC] 串口打开成功")
    return True


def exampleCloseSerial() -> bool:
    if serial_port.serialClose(serial_alias) != 0:
        print("[ERROR] 串口关闭失败")
        return False
    print("[SUCC] 串口关闭成功")
    return True


def exampleSerialSendString(data: str) -> bool:
    if serial_port.serialSendString(data, serial_alias) != 0:
        print("[ERROR] 串口发送字符串失败")
        return False
    time.sleep(0.001)
    print("[SUCC] 串口发送字符串成功")
    return True


def exampleSerialSendAllString(data: list[str]) -> bool:
    is_check = False
    if serial_port.serialSendAllString(is_check, data, serial_alias) != 0:
        print("[ERROR] 串口发送字符串数组失败")
        return False
    time.sleep(0.001)
    print("[SUCC] 串口发送字符串数组成功")
    return True


def exampleSerialSendLine(data: str) -> bool:
    if serial_port.serialSendLine(data, serial_alias) != 0:
        print("[ERROR] 串口发送带有换行符的字符串失败")
        return False
    time.sleep(0.001)
    print("[SUCC] 串口发送带有换行符的字符串成功")
    return True


def exampleSerialSendByte(data: bytes) -> bool:
    if serial_port.serialSendByte(data, serial_alias) != 0:
        print("[ERROR] 串口发送字节失败")
        return False
    time.sleep(0.001)
    print("[SUCC] 串口发送字节成功")
    return True


def exampleSerialSendInt(data: int) -> bool:
    if serial_port.serialSendInt(data, serial_alias) != 0:
        print("[ERROR] 串口发送整数失败")
        return False
    time.sleep(0.001)
    print("[SUCC] 串口发送整数成功")
    return True


def exampleSerialReadString() -> str:
    start_time = time.time()
    prefix = "<"
    suffix = ">"
    interpret_escape = False
    value_name = "read_str"
    default_value = ""
    rc.setString(value_name, default_value)  # 初始化寄存器
    while time.time() - start_time < timeout:
        response = serial_port.serialReadString(
            value_name,
            serial_alias,
            prefix,
            suffix,
            interpret_escape,
        )
        if response == 0:
            value = rc.getString(
                value_name,
                default_value,
            )
            if value != default_value:
                print(f"[SUCC] 串口读取字符串成功: {value}")
                return value
        time.sleep(0.01)
    if response != 0:
        print(f"[ERROR] 串口读取字符串超时 返回码{response}")
        return response


def exampleSerialReadByte() -> bytes:
    start_time = time.time()
    value_name = "read_byte"
    default_value = 0
    rc.setInt32(value_name, default_value)  # 初始化寄存器
    while time.time() - start_time < timeout:
        response = serial_port.serialReadByte(
            value_name,
            serial_alias,
        )
        if response == 0:
            value = rc.getInt32(
                value_name,
                default_value,
            )
            value = chr(value)
            print(f"[INFO] 字节监控数据: {value}")
        time.sleep(0.01)
    if response != 0:
        print(f"[ERROR] 串口读取字节超时 返回码{response}")
        return response


def exampleSerialReadByteList(num_bytes: int) -> list[int]:
    start_time = time.time()
    value_name = "read_byte_list"
    default_value = ["0"] * num_bytes
    rc.setVecChar(value_name, default_value)  # 初始化寄存器
    while time.time() - start_time < timeout:
        response = serial_port.serialReadByteList(
            num_bytes,
            value_name,
            serial_alias,
        )
        if response == 0:
            value = rc.getVecChar(
                value_name,
                default_value,
            )
            print(f"[INFO] 字节列表监控数据: {value}")
        time.sleep(0.01)
    if response != 0:
        print(f"[ERROR] 串口读取字节列表超时 返回码{response}")
        return response


if __name__ == "__main__":
    robot_rpc_client.connect(robot_ip, robot_port)  # 接口调用: 连接 RPC 服务
    if robot_rpc_client.hasConnected():
        print("Robot rcp_client connected successfully!")
        robot_rpc_client.login("aubo", "123456")  # 接口调用: 机械臂登录
        if robot_rpc_client.hasLogined():
            print("Robot rcp_client logined successfully!")
            exampleOpenSerial()
            exampleSerialSendString("HELLO ROBOT")
            exampleSerialSendAllString(
                ["H", "E", "L", "L", "O", " ", "R", "O", "B", "O", "T"],
            )
            exampleSerialSendLine("HELLO ROBOT")
            exampleSerialReadString()  # 接受头尾各为"<",">"的字符串例如"<9000>"。头尾字符可自行修改
            exampleSerialSendByte("a")
            exampleSerialReadByte()
            exampleSerialSendInt(9000)
            exampleSerialReadByteList(6)
            exampleCloseSerial()
            robot_rpc_client.logout()  # 退出登录
            robot_rpc_client.disconnect()  # 断开连接
