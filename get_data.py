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

service_csv_mapping = {
    "Service1": "s1_int.csv", 
    "Service2": "s2_int.csv", 
    "Service3": "s3_int.csv" 
}

def read_last_n_rows(filename, n=8):
    """读取 CSV 文件中最后 n 行数据"""
    try:
        with open(filename, mode='r', encoding='utf-8') as f:
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


def get_latest_path_stats(service_name):
    path_stats = {
        service_name: {0: [], 1: []}
    }
    filename = service_csv_mapping[service_name]
    n = 8
    while True:
        data = read_last_n_rows(filename, n)
        if not data:
            print(f"警告: {filename} 中没有数据")
        # 解析 CSV 数据
        for row in data:
            swid = int(row[0])
            if swid in SWID_MAPPING:
                service, path_id = SWID_MAPPING[swid]
                delay = float(row[1])
                jitter = float(row[2])
                throughput = float(row[3])
                loss_rate = float(row[4])
                bottleneck_load = float(row[5])
                path_stats[service][path_id].append((delay, throughput, jitter, loss_rate, bottleneck_load))
        enough_data = all(len(path_stats[service_name][i]) >= 4 for i in range(2))
        if enough_data or n >= 1000:  # 设置一个上限，避免无限循环
            break
        n += 8  # 增加读取行数

    for path_id in [0, 1]:
        if len(path_stats[service_name][path_id]) > 3:
            path_stats[service_name][path_id] = path_stats[service_name][path_id][-3:]
        elif len(path_stats[service_name][path_id]) < 3:
            print(f"警告: {service_name} 路径 {path_id} 的 path_stats 数据不足，仅有 {len(path_stats[service_name][path_id])} 个")

    return path_stats
      

def get_latest_path_delay(service_name):
    path_delay = {
        service_name: {0: [], 1: []}
    }

    filename = service_csv_mapping[service_name]
    n = 4
    while True:
        data = read_last_n_rows(filename, n)
        if not data:
            print(f"警告: {filename} 中没有数据")
        # 解析 CSV 数据
        for row in data:
            swid = int(row[0])
            if swid in SWID_MAPPING:
                service, path_id = SWID_MAPPING[swid]
                delay = float(row[1])
                path_delay[service][path_id].append(delay)
        enough_data = all(len(path_delay[service_name][i]) >= 1 for i in range(2))
        if enough_data or n >= 1000:  # 设置一个上限，避免无限循环
            break
        n += 4  # 增加读取行数

    for path_id in [0, 1]:
        if len(path_delay[service_name][path_id]) > 1:
            path_delay[service_name][path_id] = path_delay[service_name][path_id][-1:]
    #     elif len(path_stats[service_name][path_id]) < 4:
    #         print(f"警告: {service_name} 路径 {path_id} 的 path_stats 数据不足，仅有 {len(path_stats[service_name][path_id])} 个")

    return path_delay

"""
if __name__ == "__main__":
    main()
def main():
    print(f"开始监控，每 {INTERVAL} 秒读取一次数据...")
    while True:
        get_latest_data_to_path_stats()
        time.sleep(INTERVAL)  # 暂停指定间隔时间
"""

