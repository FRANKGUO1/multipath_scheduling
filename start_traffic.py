import threading
import subprocess
from config import intsend_devices, intreceive_devices, iperf_last_time

def run_command(command):
    print(f"正在运行: {command}")
    subprocess.Popen(command, shell=True)

def main():
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

    # 开启iperf流，先打180s试试，先开服务器（服务器开300s），iperf客户端打流180s
    # 用iperf搜集数据的就是无法知道流走的路径,iperf其实也可以不知道，决策完后，直接改变方向

    iperf_server_commands = []
    iperf_client_commands = []
    iperf_threads = []

    # 开启iperf服务器
    for device in intreceive_devices:
        command = f"bash /home/sinet/P4/mininet/util/m {device} python3 /home/sinet/gzc/multipath_scheduling/iperf_handle.py {device} > /dev/null 2>&1"
        iperf_server_commands.append(command)
    
    # 开启iperf客户端打流
    for device, numbers in intsend_devices.items():
        command = f"bash /home/sinet/P4/mininet/util/m {device} iperf -c {numbers[2]} -u -b {numbers[3]} -t {iperf_last_time} > /dev/null 2>&1"
        iperf_client_commands.append(command)

    for command in iperf_server_commands:
        thread = threading.Thread(target=run_command, args=(command,))
        iperf_threads.append(thread)
        thread.start()
   
    for command in iperf_client_commands:
        thread = threading.Thread(target=run_command, args=(command,))
        iperf_threads.append(thread)
        thread.start()


if __name__ == "__main__":
    main()
