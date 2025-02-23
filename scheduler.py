import subprocess
import numpy as np
import random
from int_send import send
import subprocess
import threading
import time
import sys
from config import DATAINTERVAL, requirements, priorities, services, intsend_devices, intreceive_devices
from multiprocessing import Process
import iperf3
import concurrent.futures
from scheduling_algorithm import CMAB
from switch_cli import run_simple_switch_cli
from get_data import get_latest_data_to_path_stats
import argparse

# traceroute 可以查看到每一跳的时延

# 调度函数没有问题了，然后是利用CMAB进行决策
# CMAB算法需要处理好输入，能不能直接用globecom的，但那个的问题是路径状态在客户端上获得，路径状态可以用INT。
# 那应用层指标呢，理论上吞可用带宽越大，清晰度越高，下载时间也需要考虑。
# 可以这样，输入是历史路径的时延和吞吐量，以及一个传输时间（？这个得斟酌一下）
# 先跑通一版吧，实现INT测试路径的单向时延和吞吐量以及剩余带宽（其实丢包率也可以考虑进去），服务需求则在奖励函数中实现。
# 奖励函数包括服务的带宽，时延，丢包率需求，还有服务的优先级和路径的带宽利用率等。优先级越高的奖励越大，进而保障获取更多的资源


def run_command(command):
    print(f"正在运行: {command}")
    subprocess.Popen(command, shell=True)


def run_iperf3(server_ip):
    """
    运行 iperf3 UDP 测试并返回性能指标
    参数:
        server_ip (str): 服务器的 IP 地址
    返回:
        dict: 包含吞吐量、抖动和丢包率的字典，或 None 如果测试失败
    """
    client = iperf3.Client()
    client.server_hostname = server_ip  # 设置服务器 IP
    client.port = 5201                  # 默认端口
    client.protocol = 'udp'             # 使用 UDP 协议
    client.bandwidth = 1000000          # 目标带宽 1 Mbps，可根据需要调整
    client.duration = 10                # 测试时长 10 秒

    result = client.run()  # 运行测试
    if result.error:
        print(f"服务器 {server_ip} 测试失败: {result.error}")
        return None
    else:
        # 从 JSON 输出中提取数据
        sum_data = result.json['end']['sum']
        throughput = sum_data['bytes'] * 8 / sum_data['seconds'] / 1000000  # 吞吐量 (Mbps)
        jitter = sum_data['jitter_ms']                                      # 抖动 (ms)
        packet_loss = (sum_data['lost_packets'] / sum_data['packets'] 
                       if sum_data['packets'] > 0 else 0)                   # 丢包率
        return {
            'server_ip': server_ip,
            'throughput': throughput,
            'jitter': jitter,
            'packet_loss': packet_loss
        }


def generate_path_stats():
    np.random.seed(42)  # 固定种子以复现结果
    services = ["service1", "service2", "service3"]
    path_stats = {}
    for service in services:
        path_stats[service] = []
        # Path 0: 较高带宽，低延迟
        path0 = [[np.random.normal(100, 5), np.random.normal(20, 2), np.random.uniform(0.01, 0.03), np.random.uniform(0.7, 0.8)]
                for _ in range(10)]
        # Path 1: 较低带宽，高延迟
        path1 = [[np.random.normal(90, 5), np.random.normal(25, 3), np.random.uniform(0.03, 0.05), np.random.uniform(0.8, 0.9)]
                for _ in range(10)]
        path_stats[service].append(path0)
        path_stats[service].append(path1)
    return services, path_stats


def update_path_stats(path_stats):
    for service in path_stats:
        for path in range(2):
            last_stats = path_stats[service][path][-1]
            new_stats = [
                max(50, last_stats[0] + np.random.normal(0, 2)),  # 带宽波动
                max(10, last_stats[1] + np.random.normal(0, 1)),  # 延迟波动
                max(0, last_stats[2] + np.random.uniform(-0.01, 0.01)),  # 丢包率波动
                min(1, max(0, last_stats[3] + np.random.uniform(-0.05, 0.05)))  # 负载波动
            ]
            path_stats[service][path].append(new_stats)
            path_stats[service][path].pop(0)  # 保持 10 次历史记录


if __name__ == "__main__":
    # int探测开启
    # 生成命令列表
    int_commands = []
    int_threads = []
    for device, numbers in intsend_devices.items():
        command = f"bash /home/sinet/P4/mininet/util/m {device} python3 /home/sinet/gzc/multipath_scheduling/int_send.py {numbers[0]} {numbers[1]} > /dev/null 2>&1"
        int_commands.append(command)
    
    for device in intreceive_devices:
        command = f"bash /home/sinet/P4/mininet/util/m {device} python3 /home/sinet/gzc/multipath_scheduling/int_receive.py {device} > /dev/null 2>&1"
        int_commands.append(command)        

    for command in int_commands:
        thread = threading.Thread(target=run_command, args=(command,))
        int_threads.append(thread)
        thread.start()

    # sudo pkill -9 -f 'int_'
    # sudo pkill -9 -f 'iperf'

    # 开启iperf流，先打180s试试，先开服务器（服务器开300s），iperf客户端打流180s
    # 用iperf搜集数据的就是无法知道流走的路径,iperf其实也可以不知道，决策完后，直接改变方向

    iperf_commands = []
    iperf_threads = []

    # 开启iperf服务器
    for device in intreceive_devices:
        command = f"bash /home/sinet/P4/mininet/util/m {device} python3 /home/sinet/gzc/multipath_scheduling/iperf_handle.py {device}"
        iperf_commands.append(command)
    
    # 开启iperf客户端打流
    for device, numbers in intsend_devices.items():
        command = f"bash /home/sinet/P4/mininet/util/m {device} iperf -c {numbers[2]} -u -t 5"
        iperf_commands.append(command)

    for command in iperf_commands:
        thread = threading.Thread(target=run_command, args=(command,))
        iperf_threads.append(thread)
        # thread.start()
   
    parser = argparse.ArgumentParser(description='选择不同的操作模式')
    parser.add_argument('--mode', 
                       type=str, 
                       default='CMAB',
                       choices=['CMAB', 'RR', 'minRTT'],
                       help='选择操作模式: CMAB, RR 或 minRTT (默认: CMAB)')

    # 解析命令行参数
    args = parser.parse_args()
    
    # 根据选项执行不同的操作
    mode = args.mode.upper()  # 转换为大写以保持一致性
    
    # 开始调度 
    print(f"开始监控，每 {DATAINTERVAL} 秒读取一次数据...")
    while True:
        path_stats, path_delay = get_latest_data_to_path_stats()   
        print(path_delay)         
        if mode == 'RR':
            pass
        elif mode == 'MINRTT':
            pass
        else:
            cmab = CMAB(services, path_stats, priorities, requirements)
            cmab.schedule()
            print("Final Q-Table:")
            for i, service in enumerate(services):
                print(f"{service}: Path 0 = {cmab.q_table[i, 0]:.3f}, Path 1 = {cmab.q_table[i, 1]:.3f}")
        time.sleep(DATAINTERVAL)  # 暂停指定间隔时间

    """
    services, path_stats = generate_path_stats()
    

    # 模拟动态路径状态

    # 运行 1000 次
    cmab = CMAB(services, path_stats, priorities, requirements)
    for _ in range(100):
        cmab.schedule()
        update_path_stats(path_stats)  # 模拟路径状态变化

    # 输出最后的 Q 值表
    print("Final Q-Table:")
    for i, service in enumerate(services):
        print(f"{service}: Path 0 = {cmab.q_table[i, 0]:.3f}, Path 1 = {cmab.q_table[i, 1]:.3f}")
    
    """


    """
    # iperf3 打流
    # 开启服务器
    iperf_server_devices = ['s1', 's2', 's3']
    iperf_server_commands = [
        f"bash /home/sinet/P4/mininet/util/m {device} iperf3 -s "
        for device in iperf_server_devices        
    ]

    iperf_server_threads = []

    for command in iperf_server_commands:
        thread = threading.Thread(target=run_command, args=(command,))
        iperf_server_threads.append(thread)
        thread.start()
    
        

    intreceive_commands = [
        f"bash /home/sinet/P4/mininet/util/m {device} python3 /home/sinet/gzc/multipath_scheduling/int_receive.py"
        for device in client_devices
    ]    

    # 定义服务优先级和服务需求
    services = ['Service 1', 'Service 2', 'Service 3']
    # 加个映射
    # 这个priorities的值也是经验值，需要调试。
    priorities = {'Service 1': 3, 'Service 2': 2, 'Service 3': 1.5}  # 高数值表示高优先级
    requirements = {'Service 1': {'bandwidth': 100, 'latency': 50, 'loss_rate': 0.01},
                    'Service 2': {'bandwidth': 200, 'latency': 30, 'loss_rate': 0.02}, 
                    'Service 3': {'bandwidth': 150, 'latency': 50, 'loss_rate': 0.03}}

    # 模拟路径的带宽、延迟、丢包率和带宽负载，以及历史状态（例如最近几轮的值）,带宽=固有带宽-吞吐量
    # 带宽负载就是 路径吞吐量/路径固有带宽，带宽负载越大越不好
    # 路径和服务是绑定的 
    # 路径状态（剩余带宽，时延，丢包率，带宽负载）


    path_stats = {
        'Service 1': {
            0 : [(100, 50, 0.01, 0.4), (110, 52, 0.015, 0.38)], 
            1: [(120, 55, 0.02, 0.3), (125, 57, 0.025, 0.32)]
        },
        'Service 2': {
            0: [(200, 30, 0.01, 0.5), (190, 32, 0.015, 0.55)],
            1: [(180, 35, 0.02, 0.6), (175, 36, 0.025, 0.58)]
        },
        'Service 3': {
            0: [(150, 40, 0.02, 0.3), (155, 42, 0.022, 0.42)],
            1: [(160, 45, 0.03, 0.5), (165, 47, 0.035, 0.48)]
        }
    } # 路径(吞吐量, 延迟, 丢包率, 带宽负载)，通过索引来区分路径，索引为0为路径1，1为路径2

    
    # 创建CMAB实例
    # 所以输入需要优先级，需求列表，路径状态字典（但需求，优先级这些都是固定的，本质上只要路径状态）
    """
    # cmab = CMAB(services, path_stats, priorities, requirements)
