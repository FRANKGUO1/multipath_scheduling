import time
from collections import deque

# 定义三个 device 和对应的文件
DEVICES = ['s1', 's2', 's3']
INTERVAL = 3  # 每隔3秒读取一次（可调整）

# swid 到 服务和路径的映射
SWID_MAPPING = {
    16: ('Service1', 0),
    17: ('Service1', 1),
    26: ('Service2', 0),
    27: ('Service2', 1),
    36: ('Service3', 0),
    37: ('Service3', 1)
}

def read_last_n_rows(filename, n=8):
    """读取 CSV 文件中最后 n 行数据"""
    try:
        with open(filename, mode='r', encoding='utf-8') as f:
            # 使用 deque 高效读取最后 n 行
            lines = deque(f, maxlen=n)
            # 跳过表头（假设第一行是表头）
            data = []
            for line in lines:
                # 手动解析 CSV 行（避免 csv.reader 对空文件报错）
                row = line.strip().split(',')
                if row[0] != '路径':  # 跳过头行
                    data.append(row)
            return data
    except FileNotFoundError:
        print(f"文件 {filename} 未找到")
        return []
    except IOError:
        print(f"读取文件 {filename} 时发生错误")
        return []

def get_latest_data_to_path_stats():
    """获取三个文件中最新的8组数据并转化为 path_stats 格式"""
    path_stats = {
        'Service1': {0: [], 1: []},
        'Service2': {0: [], 1: []},
        'Service3': {0: [], 1: []}
    }

    path_delay = {
        'Service1': {0: [], 1: []},
        'Service2': {0: [], 1: []},
        'Service3': {0: [], 1: []}
    }

    for device in DEVICES:
        filename = f"{device}_int.csv"
        latest_data = read_last_n_rows(filename, n=8)
        last_data = read_last_n_rows(filename, n=2)

        for row in latest_data:
            # 假设 CSV 格式为：路径,delay_ms,jitter_ms,Throughput_MBps,Packet_Loss_Rate_%,瓶颈链路负载_MBps
            swid = int(row[0])  # swid 是整数
            delay = float(row[1])  # delay_ms
            jitter = float(row[2])  # jitter_ms
            throughput = float(row[3])  # Throughput_MBps
            loss_rate = float(row[4]) / 100  # Packet_Loss_Rate_% 转换为小数
            bottleneck_load = float(row[5])  # 瓶颈链路负载_MBps（未使用）

            # 根据 swid 获取服务和路径
            if swid in SWID_MAPPING:
                service, path_id = SWID_MAPPING[swid]
                # 将数据添加到对应服务和路径
                path_stats[service][path_id].append((throughput, jitter, loss_rate, bottleneck_load))
            else:
                print(f"未知的 swid: {swid}")

        for row in last_data:
            swid = int(row[0])  # swid 是整数
            delay = float(row[1])  # delay_ms

            if swid in SWID_MAPPING:
                service, path_id = SWID_MAPPING[swid]
                # 将数据添加到对应服务和路径
                path_delay[service][path_id].append((delay))
            else:
                print(f"未知的 swid: {swid}")
    return path_stats, path_delay

"""
if __name__ == "__main__":
    main()
def main():
    print(f"开始监控，每 {INTERVAL} 秒读取一次数据...")
    while True:
        get_latest_data_to_path_stats()
        time.sleep(INTERVAL)  # 暂停指定间隔时间
"""

