import time
import subprocess
import threading

# 定义目标IP地址（两条路径）
target_ip = "10.0.1.2"
network_interface1 = "eth0"  
network_interface2 = "eth1" 

# 映射
interface_mapping = {
    network_interface1: 1, 
    network_interface2: 2
    }

# 获取网络接口的吞吐量（接收字节数）
def get_throughput(interface):
    # 获取接收字节数
    with open(f"/sys/class/net/{interface}/statistics/rx_bytes") as f:
        rx_bytes_before = int(f.read())
    
    # 获取发送字节数
    with open(f"/sys/class/net/{interface}/statistics/tx_bytes") as f:
        tx_bytes_before = int(f.read())

    time.sleep(4)

    with open(f"/sys/class/net/{interface}/statistics/rx_bytes") as f:
        rx_bytes_after = int(f.read())
    
    with open(f"/sys/class/net/{interface}/statistics/tx_bytes") as f:
        tx_bytes_after = int(f.read())

    # 计算吞吐量（单位：KB/s）
    rx_rate = (rx_bytes_after - rx_bytes_before) / 1024  # 转换为 KB
    tx_rate = (tx_bytes_after - tx_bytes_before) / 1024  # 转换为 KB

    throughput = rx_rate + tx_rate
    return throughput

# 获取路径时延
def get_latency(target_ip, interface):
    # 使用 ping 命令获取时延
    try:
        ping_output = subprocess.check_output(
            ["ping", "-c", "4", "-I", interface, target_ip], stderr=subprocess.STDOUT, universal_newlines=True
        )
        # 从 ping 输出中提取时延
        lines = ping_output.splitlines()
        # 获取平均时延
        for line in lines:
            if "avg" in line:
                avg_latency = line.split("/")[4] 
                return float(avg_latency)
    except subprocess.CalledProcessError:
        return None 


def monitor_path(target_ip, network_interface):
    # 获取路径的时延和吞吐量
    latency = get_latency(target_ip, network_interface)
    throughput = get_throughput(network_interface)

    results = {
        "rtt": None,
        "throughput": None
    }
    
    # 创建线程来收集数据
    def measure_latency():
        latency = get_latency(target_ip, network_interface)
        if latency is not None:
            results["latency"] = latency
            
    def measure_throughput():
        throughput = get_throughput(network_interface)
        results["throughput"] = throughput
    
    # 启动线程
    latency_thread = threading.Thread(target=measure_latency)
    throughput_thread = threading.Thread(target=measure_throughput)
    
    latency_thread.start()
    throughput_thread.start()
    
    # 等待两个线程完成
    latency_thread.join()
    throughput_thread.join()
    
    return results


def main():
    thread1 = threading.Thread(target=monitor_path, args=(target_ip, network_interface1))
    thread2 = threading.Thread(target=monitor_path, args=(target_ip, network_interface2))

    # 启动线程
    thread1.start()
    thread2.start()
    
    # 等待线程完成（这里永远不会完成，因为线程是无限循环）
    thread1.join()
    thread2.join()

    


if __name__ == '__main__':
    main()

