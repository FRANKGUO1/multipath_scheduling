import sys
from scapy.all import sniff
from headers_definition import probe_data, get_if
import argparse
import csv
from config import INTINTERVAL
# from config import data_storage

# 全局变量保存上一个包的时延
prev_delays = {}

def expand(x):
    yield x
    while x.payload:
        x = x.payload
        yield x

def handle_pkt(pkt, device):
    global prev_delay
    # pkt.show2()
    if pkt.haslayer(probe_data):
        probe_data_layers = [l for l in expand(pkt) if l.name == 'probe_data']
        # print("")
        # l = len(probe_data_layers) # int数据层个数
        # print("probe_data层数为：", l)
        total_delay = round((probe_data_layers[0].egress_cur_time - probe_data_layers[-1].ingress_cur_time) * 0.00001, 4)
        jitter = round((probe_data_layers[0].egress_cur_time - probe_data_layers[-1].ingress_cur_time) * 0.0000001, 4)
        total_throughput = round(1.0 * probe_data_layers[0].egress_byte_cnt / (probe_data_layers[0].egress_cur_time - probe_data_layers[0].egress_last_time), 4)
        total_loss_rate = round(1.0 * (probe_data_layers[0].egress_packet_count - probe_data_layers[-1].ingress_packet_count) / probe_data_layers[0].egress_packet_count * 100, 4)
        bottleneck_load = round(1.0 * probe_data_layers[-2].egress_byte_cnt / (probe_data_layers[-2].egress_cur_time - probe_data_layers[-2].egress_last_time), 4) # 还未除以瓶颈链路带宽
        swid = probe_data_layers[0].swid

        # 计算抖动
        jitter = None
        if swid in prev_delays:  # 如果该 swid 有前一个包的数据
            jitter = round(abs(total_delay - prev_delays[swid]), 4)
        prev_delays[swid] = total_delay  # 更新该 swid 的前一个包时延
        if jitter is not None:
            # 构造 CSV 文件名
            csv_filename = f"{device}_int.csv"
            
            # 定义 CSV 表头和数据
            headers = ['路径', 'delay_ms', 'jitter_ms', 'Throughput_MBps', 'Packet_Loss_Rate_%', '瓶颈链路负载_MBps']
            data = [swid, total_delay, jitter, total_throughput, total_loss_rate, bottleneck_load]

            # 以追加模式打开文件，写入数据
            with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # 如果文件为空，写入表头
                if f.tell() == 0:
                    writer.writerow(headers)
                # 写入数据行
                writer.writerow(data)
            print(f"路径{swid} delay:{total_delay}ms  jitter:{jitter}ms  Throughput:{total_throughput}MBps  Packet_Loss_Rate:{total_loss_rate}%  瓶颈链路负载:{bottleneck_load}MBps")

              
def receive(device):
    iface = get_if()
    if iface:
        print(f"sniffing on {iface}")
        sys.stdout.flush()
        sniff(iface=iface, prn=lambda x: handle_pkt(x, device))
    else:
        print("No suitable interface found.")


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        # 添加命令行选项
        parser.add_argument('device', type=str, help='指定文件名')
        args = parser.parse_args()
        receive(args.device)
    except KeyboardInterrupt:
        print("Stopping...")


