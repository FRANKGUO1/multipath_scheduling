import numpy as np
from int_send import send
import time
import sys
from config import DATAINTERVAL, requirements, priorities, services, service_port_mapping
import iperf3
from scheduling_algorithm import CMAB
from switch_cli import run_simple_switch_cli
from get_data import get_latest_data_to_path_stats
import argparse


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


# 找出每个Service中时延最大的key
def find_max_delay_key(data):
    result = {}
    for service, delays in data.items():
        # 获取每个key对应的时延值（取列表第一个元素）
        min_key = min(delays.items(), key=lambda x: x[1][0])
        result[service] = {
            'key': min_key[0],
            'delay': min_key[1][0]
        }
    return result

if __name__ == "__main__":
    """
    path1：50M 5ms 1%
    path2：30M 10ms 2%

    Service1 20M
    Service2 10M
    Service3 25M
    """
    parser = argparse.ArgumentParser(description='选择不同的操作模式')
    parser.add_argument('--mode', 
                       type=str, 
                       default='CMAB',
                       choices=['CMAB', 'RR', 'minRTT'],
                       help='选择操作模式: CMAB, RR 或 minRTT (默认: CMAB)')
    args = parser.parse_args()
    mode = args.mode.upper()
    
    # 开始调度 
    while True:
        path_stats, path_delay = get_latest_data_to_path_stats()   
        # print(path_delay)         
        if mode == 'RR':
            for i in range(1, -1, -1):
                for k, v in service_port_mapping.items():
                    run_simple_switch_cli(v[0], i)
                    run_simple_switch_cli(v[1], i)
                time.sleep(DATAINTERVAL)  # 暂停指定间隔时间
        elif mode == 'MINRTT':
            # print(path_delay)
            max_delays = find_max_delay_key(path_delay)
            for service, info in max_delays.items():
                # print(f"{service}: 最小时延key = {info['key']}, 时延值 = {info['delay']}")
                run_simple_switch_cli(service_port_mapping[service][0], info['key'])
                run_simple_switch_cli(service_port_mapping[service][1], info['key'])
            time.sleep(DATAINTERVAL)  # 暂停指定间隔时间
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
