import numpy as np
import time
import sys
from config import DATAINTERVAL, requirements, priorities, services, service_port_mapping
import iperf3
from scheduling_algorithm import CMAB
from switch_cli import run_simple_switch_cli
from get_data import get_latest_path_delay, get_latest_path_stats
import argparse
import signal
import os
from subprocess import call


def signal_handler(sig, frame):
    # 文件重命名
    try:
        for i in range(1, 4):  # For s1, s2, s3
            old_name = f"iperf_s{i}_results.csv"
            new_name = f"iperf_s{i}_{mode}_results.csv"
            if os.path.exists(old_name):
                os.rename(old_name, new_name)
                print(f"重命名 {old_name}  {new_name}")
            else:
                print(f"文件 {old_name} 未找到, 跳过...")
    except Exception as e:
        print(f"寻找文件时错误: {e}")

    # 寄存器设置成默认
    try:
        print("重置寄存器")
        for _, v in service_port_mapping.items():
            run_simple_switch_cli(v[0], 0)
            run_simple_switch_cli(v[1], 0)
    except Exception as e:
        print(f"错误: {e}")

    # 退出程序
    print("退出程序")
    sys.exit(0)


# 找出每个Service中时延最大的key
def find_min_delay_key(data):
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
    parser = argparse.ArgumentParser(description='选择不同的操作模式')
    parser.add_argument('--mode', 
                       type=str, 
                       default='CMAB',
                       choices=['CMAB', 'RR', 'minRTT', 'CMAB_ts', 'CMAB_greedy'],
                       help='选择操作模式: CMAB, RR 或 minRTT (默认: CMAB)')
    args = parser.parse_args()
    mode = args.mode.upper()

    signal.signal(signal.SIGINT, signal_handler)
    
    # 开始调度 
    try:
        while True:   
            if mode == 'RR':
                for i in range(1, -1, -1):
                    for k, v in service_port_mapping.items():
                        print("选择路径：", i)
                        run_simple_switch_cli(v[0], i)
                        run_simple_switch_cli(v[1], i)
                    time.sleep(DATAINTERVAL)  # 暂停指定间隔时间
            elif mode == 'MINRTT':
                # print(path_stats)
                for service_name in services:
                    path_delay = get_latest_path_delay(service_name)
                    print(path_delay)  
                    min_delays = find_min_delay_key(path_delay)
                    for service, info in min_delays.items():
                        print(f"{service}: 最小时延key = {info['key']}, 时延值 = {info['delay']}")
                        run_simple_switch_cli(service_port_mapping[service][0], info['key'])
                        run_simple_switch_cli(service_port_mapping[service][1], info['key'])
                    if service_name == 'Service3':
                        time.sleep(DATAINTERVAL)  
            else:
                cmab = CMAB(priorities, requirements)
                for service_name in services:
                    path_stats = get_latest_path_stats(service_name)
                    # print("各路径状态", path_stats)
                    cmab.schedule(service_name, path_stats)
                    # print(f"{service_name}: Path 0 = {cmab.q_table[services.index(service_name), 0]:.3f}, Path 1 = {cmab.q_table[services.index(service_name), 1]:.3f}")
                    time.sleep(DATAINTERVAL)  # 暂停指定间隔时间
    except KeyboardInterrupt:
        pass
    
