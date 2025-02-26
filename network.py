from p4utils.mininetlib.network_API import NetworkAPI
from config import bottleneck_config, bw_sigma
import time
import numpy as np
from multiprocessing import Process


"""
六个主机，八台交换机
"""

#-------------控制路径带宽-----------------------------------------
def change_bw_normal_distribution(net, node1, node2, bw):
    while True:
        # 接口带宽遵从sigma为3的正态分布
        bw_value = int(np.random.normal(bw, bw_sigma))
        # bw_value2 = int(np.random.normal(bottleneck_config[1][0], bw_sigma))
        net.setBw(node1, node2, bw_value)
        # net.setBw(node1, node2, bw_value2, 1)
        print(f"Setting bandwidth between {node1} and {node2} to {bw_value} Mbps")
        # print(f"Setting bandwidth 1 between {node1} and {node2} to {bw_value2} Mbps")
        # 每隔5s变化一次带宽
        time.sleep(5)


if __name__ == '__main__':
    net = NetworkAPI()

    # Network general options
    net.setLogLevel('info')
    net.enableCli()

    # Network definition
    net.addHost('s1')
    net.addHost('s2')
    net.addHost('s3')

    net.addHost('d1')
    net.addHost('d2')
    net.addHost('d3')

    # net.addHost('c1')

    net.addP4Switch('r1', cli_input='commands/r1-commands.txt')
    net.addP4Switch('r2', cli_input='commands/r2-commands.txt')
    net.addP4Switch('r3', cli_input='commands/r3-commands.txt')
    net.addP4Switch('r8', cli_input='commands/r8-commands.txt')
    net.addP4Switch('r9', cli_input='commands/r9-commands.txt')
    net.addP4Switch('r10', cli_input='commands/r10-commands.txt')

    net.addP4Switch('r4', cli_input='commands/r4-commands.txt')
    net.addP4Switch('r5', cli_input='commands/r5-commands.txt')
    net.addP4Switch('r6', cli_input='commands/r6-commands.txt')
    net.addP4Switch('r7', cli_input='commands/r7-commands.txt')

    net.setP4Source('r1','p4code/edge_switch.p4')
    net.setP4Source('r2','p4code/edge_switch.p4')
    net.setP4Source('r3','p4code/edge_switch.p4')
    net.setP4Source('r8','p4code/edge_switch.p4')
    net.setP4Source('r9','p4code/edge_switch.p4')
    net.setP4Source('r10','p4code/edge_switch.p4')

    net.setP4Source('r4','p4code/core_switch.p4')
    net.setP4Source('r5','p4code/core_switch.p4')
    net.setP4Source('r6','p4code/core_switch.p4')
    net.setP4Source('r7','p4code/core_switch.p4')

    #-------------------------------------------------
    net.addLink('s1', 'r1')
    net.addLink('s2', 'r2')
    net.addLink('s3', 'r3')

    net.addLink('d1', 'r8')
    net.addLink('d2', 'r9')
    net.addLink('d3', 'r10')

    net.addLink('r1', 'r4')
    net.addLink('r1', 'r5')
    net.addLink('r2', 'r4')
    net.addLink('r2', 'r5')
    net.addLink('r3', 'r4')
    net.addLink('r3', 'r5')

    net.addLink('r4', 'r6')
    net.addLink('r5', 'r7')

    net.addLink('r8', 'r6')
    net.addLink('r8', 'r7')
    net.addLink('r9', 'r6')
    net.addLink('r9', 'r7')
    net.addLink('r10', 'r6')
    net.addLink('r10', 'r7')

    net.setBw('r4', 'r6', bottleneck_config[0][0])
    net.setBw('r5', 'r7', bottleneck_config[1][0])

    net.setDelay('r4', 'r6', bottleneck_config[0][1])
    net.setDelay('r5', 'r7', bottleneck_config[1][1])

    net.setLoss('r4', 'r6', bottleneck_config[0][2])
    net.setLoss('r5', 'r7', bottleneck_config[1][2])

    net.l3()
    net.enablePcapDumpAll()
    # net.enableLogAll()

    # Start network
    net.startNetwork()

"""
net.addLink('r1', 'c1')
net.addLink('r2', 'c1')
net.addLink('r3', 'c1')
net.addLink('r8', 'c1')
net.addLink('r9', 'c1')
net.addLink('r10', 'c1')
"""

"""
net.setIntfMac('h1', 's1', 'e8:61:1f:37:b6:83')
net.setIntfMac('s1', 'h1', 'a0:36:9f:08:d1:28')
net.setIntfMac('h2', 's1', 'a0:36:9f:08:d1:2b')
net.setIntfMac('s1', 'h2', 'd0:36:9f:ed:5c:63')
net.setIntfMac('s1', 'h2', 'a0:36:9f:08:c7:b7')
net.setIntfMac('h2', 's1', 'e8:61:1f:37:b5:8b')

net.setIntfIp('s1', 'r1', '10.0.1.2/24')
net.setIntfIp('s1', 'r2', '10.0.1.3/24')
net.setIntfIp('s2', 'r1', '10.0.2.2/24')
net.setIntfIp('s2', 'r2', '10.0.2.3/24')
net.setIntfIp('s3', 'r1', '10.0.3.2/24')
net.setIntfIp('s3', 'r2', '10.0.3.3/24')

net.setIntfIp('d1', 'r3', '10.1.1.2/24')
net.setIntfIp('d2', 'r4', '10.1.2.2/24')
net.setIntfIp('d3', 'r5', '10.1.3.2/24')

net.setDefaultRoute('h1', '10.0.1.1')
net.setDefaultRoute('h2', '10.0.2.1')
net.setDefaultRoute('h2', '10.0.3.1')

change_bw_process = []
change_bw_process1 = Process(target=change_bw_normal_distribution, args=(net, 'r4', 'r6', bottleneck_config[0][0]))
change_bw_process2 = Process(target=change_bw_normal_distribution, args=(net, 'r5', 'r7', bottleneck_config[1][0]))
change_bw_process.append(change_bw_process1)
change_bw_process.append(change_bw_process2)
for thread in change_bw_process:
    thread.start()

change_bw_normal_distribution(net, 'r4', 'r6', bottleneck_config[0][0])
change_bw_normal_distribution(net, 'r5', 'r7', bottleneck_config[1][0])
"""


