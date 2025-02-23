import iperf3
import concurrent.futures
from switch_cli import send_cli_commands
from config import ip_port_mapping


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
    client.duration = 5               # 测试时长 10 秒

    result = client.run()  # 运行测试
    print(result)
    # 还需要知道走的什么路径，还要读取寄存器
    if result.error:
            print(f"错误: {result.error}")
    else:
        print(f"{client.protocol} 测试成功完成！")
        if client.protocol == 'tcp':
            print(f"TCP吞吐量: {result.Mbps} Mbps")
        elif client.protocol == 'udp':
            print(f"UDP吞吐量: {result.Mbps} Mbps")
            print(f"UDP抖动: {result.jitter_ms} ms")
            print(f"UDP丢包: {result.lost_percent} 包")
    
if __name__ == '__main__':
    run_iperf3('10.1.1.2')