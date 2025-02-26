from multiprocessing import Manager


# 优先级相关参数
priority_threshold = 0.6
overlap_factor = 3.0
gama = 3.0 # 时延的奖惩系数

# cmab参数调整
cmab_k_value = 0.3  # 值越大，算法探索性越高，越倾向于选择选择次数少的路径 
cmab_alpha_value = 0.5   # 值越大，学习率越高，越能适应环境的变化

# 读取int数据文件时间间隔，也是调度时间间隔
DATAINTERVAL = 2

# int发包时间间隔
INTINTERVAL = 0.5  # 要为浮点数

DEFAULT_PATH = 0

# iperf流持续时间
iperf_last_time = 180

# 带宽按照正态分布，设定sigma值
bw_sigma = 3

# 瓶颈链路带宽，时延，丢包率设定，当流量吞吐量超过mininet链路带宽的90%左右时，会产生丢包，所以这里设置的带宽是按90%折算的
bottleneck_config = {
    0: [38, 5, 0.01],
    1: [46, 3, 0.005]
}

# 服务需求
requirements = {
    "Service1": {"bandwidth": 25, "jitter": 5, "loss_rate": 0.03},
    "Service2": {"bandwidth": 20, "jitter": 10, "loss_rate": 0.05},
    "Service3": {"bandwidth": 10, "jitter": 15, "loss_rate": 0.07}
}

# 前两个元素是路径号，第三个元素是目的ip，第四个元素是打流大小（M）
intsend_devices = {
    'd1': ['16', '17', '10.1.1.2', f'{str(requirements["Service1"]["bandwidth"])}M'],
    'd2': ['26', '27', '10.2.1.2', f'{str(requirements["Service2"]["bandwidth"])}M'],
    'd3': ['36', '37', '10.3.1.2', f'{str(requirements["Service3"]["bandwidth"])}M']
}

intreceive_devices = ['s1', 's2', 's3']

# 服务与thrift_port映射
service_port_mapping = {
    'Service1': [9090, 9097],
    'Service2': [9091, 9098],
    'Service3': [9092, 9099]
}

priorities = {"Service1": 5, "Service2": 2, "Service3": 1}

services = ["Service1", "Service2", "Service3"]

# 创建进程间共享的字典
manager = Manager()
data_storage = manager.dict()
swid_data = manager.list()

# ip地址与thrift_port映射
ip_port_mapping = {
    '10.1.1.2': 9090,
    '10.2.1.2': 9091,
    '10.3.1.2': 9092
}

if __name__ == '__main__':
    print(intsend_devices)