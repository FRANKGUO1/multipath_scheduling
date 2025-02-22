from scapy.all import Ether, IP, get_if_hwaddr, sendp
import sys
import time
from headers_definition import probe_header, get_if, probe_fwd
import argparse
from config import INTINTERVAL


# 目的mac搞一个映射
dstmac_mapping = {
    "d1-eth0": "00:00:0a:01:01:02", 
    "d2-eth0": "00:00:0a:02:01:02", 
    "d3-eth0": "00:00:0a:03:01:02", 
}

dstip_mapping = {
    "d1-eth0": "10.1.1.2", 
    "d2-eth0": "10.2.1.2", 
    "d3-eth0": "10.3.1.2", 
}

# Int包直接用ip转发，但这需要目的ip地址，所以需要映射，该如何映射
"""
1,2,3为服务器，
1 10.8.1.2
2 10.9.1.2
3 10.10.1.2

4,5,6为用户
4 10.1.1.2
5 10.2.1.2
6 10.3.1.2
谁在前谁就是源，谁在后谁就是目的，利用这个来映射
可能组合14，25，36
"""

def send(src_dst_id1, src_dst_id2):
    iface = get_if()
    probe_pkt_1 = Ether(src=get_if_hwaddr(iface), dst=dstmac_mapping[iface]) / \
                  probe_header(num_probe_data=0) / probe_fwd(src_dst_id=src_dst_id1)
    
    probe_pkt_2 = Ether(src=get_if_hwaddr(iface), dst=dstmac_mapping[iface]) / \
                  probe_header(num_probe_data=0) / probe_fwd(src_dst_id=src_dst_id2)

    while True:
        try:
            # probe_pkt_1.show()
            sendp(probe_pkt_1, iface=iface)
            sendp(probe_pkt_2, iface=iface)
            time.sleep(INTINTERVAL)
        except KeyboardInterrupt as e:
            print('Program terminated by user.')
            sys.exit()


if __name__ == '__main__':
    # iface = get_if()
    # print(iface, type(iface))
        # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()

    # 添加命令行选项
    parser.add_argument('src_dst_id1', type=int, help='指定源和目的地址')
    parser.add_argument('src_dst_id2', type=int, help='指定源和目的地址')
    args = parser.parse_args()
    # print(args.src_dst_id)
    send(args.src_dst_id1, args.src_dst_id2)