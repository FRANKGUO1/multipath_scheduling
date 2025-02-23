from multiprocessing import Manager

intsend_devices = {
    'd1': ['16', '17', '10.1.1.2'],
    'd2': ['26', '27', '10.2.1.2'],
    'd3': ['36', '37', '10.3.1.2']
}

intreceive_devices = ['s1', 's2', 's3']

# 服务与thrift_port映射
service_port_mapping = {
    'Service1': [9090, 9097],
    'Service2': [9091, 9098],
    'Service3': [9092, 9099]
}

# 服务需求
requirements = {
    "Service1": {"bandwidth": 100, "latency": 30, "loss_rate": 0.05},
    "Service2": {"bandwidth": 90, "latency": 35, "loss_rate": 0.06},
    "Service3": {"bandwidth": 80, "latency": 40, "loss_rate": 0.07}
}

priorities = {"Service1": 3, "Service2": 2, "Service3": 1}

services = ["Service1", "Service2", "Service3"]

# 读取int数据文件时间间隔
DATAINTERVAL = 4

# int发包时间间隔
INTINTERVAL = 0.5

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