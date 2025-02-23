"""                        
          +--+--+     +--+--+       
+--+      |     +-----+     |          
|h1+------+ s1  |     |  h2 |
+--+      |     +-----+     |                        
          +--+--+     +--+--+
            
"""

from p4utils.mininetlib.network_API import NetworkAPI
import time
import numpy as np
from multiprocessing import Process


bw_sigma = 5

bw_high = 30
bw_low = 10

# one way delay
latency_high = "50ms"
latency_low = "10ms"


#-------------控制路径带宽-----------------------------------------
def change_bw_normal_distribution(net, node1, node2):
    while True:
        # 接口带宽遵从sigma为5的正态分布
        bw_value1 = int(np.random.normal(bw_high, bw_sigma))
        bw_value2 = int(np.random.normal(bw_low, bw_sigma))
        net.setBw(node1, node2, bw_value1, 0)
        net.setBw(node1, node2, bw_value2, 1)
        # print(f"Setting bandwidth 0 between {node1} and {node2} to {bw_value1} Mbps")
        # print(f"Setting bandwidth 1 between {node1} and {node2} to {bw_value2} Mbps")
        # 每隔5s变化一次带宽
        time.sleep(5)


#-------------控制路径时延--------------------------------------------
def change_latency_normal_distribution(net, node1, node2):
    while True:
        latency1 = int(np.random.normal(50, 5))
        latency2 = int(np.random.normal(10, 1))
        
        net.setDelay(node1, node2, latency1, 0)
        net.setDelay(node1, node2, latency2, 1)
        # print(f"Setting delay 0 between {node1} and {node2} to {latency1} Mbps")
        # print(f"Setting delay 1 between {node1} and {node2} to {latency2} Mbps")
        # 通过循环每隔1s变化一次
        time.sleep(1)


if '__main__' == __name__:
    net = NetworkAPI()

    # Network general options
    net.setLogLevel('info')
    net.enableCli()

    # Network definition
    net.addP4Switch('s1', cli_input='s1-commands.txt')
    net.setP4Source('s1','ipv4.p4')

    net.addHost('h1')
    net.addHost('h2')

    net.addLink('s1', 'h1', port1=1, port2=0, key=0, intfName1="s1-eth0", intfName2="eth0")
    net.addLink('s1', 'h2', port1=2, port2=0, key=0, intfName1="s1-eth1", intfName2="eth0")
    net.addLink('s1', 'h2', port1=3, port2=1, key=1, intfName1="s1-eth2", intfName2="eth1")

    net.setIntfMac('h1', 's1', 'e8:61:1f:37:b6:83')
    net.setIntfMac('s1', 'h1', 'a0:36:9f:08:d1:28')

    net.setIntfMac('h2', 's1', 'a0:36:9f:08:d1:2b', 0)
    net.setIntfMac('s1', 'h2', 'd0:36:9f:ed:5c:63', 0)
    net.setIntfMac('s1', 'h2', 'a0:36:9f:08:c7:b7', 1)
    net.setIntfMac('h2', 's1', 'e8:61:1f:37:b5:8b', 1)

    net.setIntfIp('h1', 's1', '10.0.1.2/24', 0)
    net.setIntfIp('h2', 's1', '10.0.2.2/24', 0)
    net.setIntfIp('h2', 's1', '10.0.3.2/24', 1)

    net.setDefaultRoute('h1', '10.0.1.1')
    net.setDefaultRoute('h2', '10.0.2.1')
    net.setDefaultRoute('h2', '10.0.3.1')

    # 设置链路带宽和时延，使其变化符合正态分布
    change_bw_process = Process(target=change_bw_normal_distribution, args=(net, 'h2', 's1'))
    change_bw_process.start()

    change_latency_process = Process(target=change_latency_normal_distribution, args=(net, 'h2', 's1'))
    change_latency_process.start()

    # Assignment strategy
    # net.mixed()

    # Nodes general options
    # net.enablePcapDumpAll()

    # net.enableLogAll()

    # Start network
    net.startNetwork()

    change_bw_process.terminate()
    change_latency_process.terminate()
