import subprocess
import re
import time
import csv
from dataclasses import dataclass
from typing import Optional, List
import argparse

@dataclass
class NetworkState:
    jitter: Optional[float] = None  # 平均延迟 (ms)
    bandwidth: Optional[float] = None    # 带宽 (Mbps)
    lost: Optional[int] = None           # 丢包数
    total: Optional[int] = None          # 总包数

class IperfServer:
    def __init__(self, csv_filename: str = "iperf_data.csv"):
        self.csv_filename = csv_filename
        cmd = ["iperf", "-s", "-u", "-i", "1", "-e"]
        print("Starting iperf server with command:", " ".join(cmd))
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.stdout = self.process.stdout
        # 初始化 CSV 文件
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "Bandwidth (Mbps)", "jitter (ms)", "Lost", "Total", "Loss Rate (%)"])

    def close(self):
        self.process.terminate()
        self.process.wait()
        print("Iperf server closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def parse_line(self, line: str) -> Optional[NetworkState]:
        print("Raw output:", line)  # 调试：打印原始输出
        # 示例: [  3] 0.0-1.0 sec  1.25 MBytes  10.5 Mbits/sec  0.123 ms    5/  893 (0.56%)
        match = re.search(r'(\d+\.\d+) Mbits/sec\s+(\d+\.\d+) ms\s+(\d+)/\s*(\d+)', line)
        if match:
            state = NetworkState()
            state.bandwidth = float(match.group(1))    # Mbps
            state.jitter = float(match.group(2))  # ms
            state.lost = int(match.group(3))           # 丢包数
            state.total = int(match.group(4))          # 总包数
            return state
        return None

    def write_to_csv(self, state: NetworkState):
        """将数据写入 CSV 文件"""
        with open(self.csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            loss_rate = (state.lost / state.total * 100) if state.total and state.total > 0 else 0
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                state.bandwidth,
                state.jitter,
                state.lost,
                state.total,
                f"{loss_rate:.2f}"
            ])

    def monitor(self, seconds: float) -> List[NetworkState]:
        states = []
        end_time = time.time() + seconds
        while time.time() < end_time:
            line = self.stdout.readline()
            if line:
                line = line.decode().strip()
                state = self.parse_line(line)
                if state:
                    states.append(state)
                    self.write_to_csv(state)  # 写入 CSV
                    print(f"Parsed: Bandwidth={state.bandwidth} Mbps, jitter={state.jitter} ms, "
                          f"Lost/Total={state.lost}/{state.total}")
            else:
                print("No output yet, waiting...")
                time.sleep(0.1)
        return states


def main(device):
    with IperfServer(f"iperf_{device}_results.csv") as server:
        print("Monitoring for 10 seconds...")
        states = server.monitor(10)
        if states:
            avg_jitter = sum(state.jitter for state in states) / len(states)
            total_lost = sum(state.lost for state in states if state.lost is not None)
            total_packets = sum(state.total for state in states if state.total is not None)
            loss_rate = (total_lost / total_packets * 100) if total_packets > 0 else 0
            print(f"Average jitter: {avg_jitter:.3f} ms")
            print(f"Total Loss Rate: {loss_rate:.2f}% ({total_lost}/{total_packets})")
        else:
            print("No data collected")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 添加命令行选项
    parser.add_argument('device', type=str, help='指定文件名')
    args = parser.parse_args()
    # print(args.src_dst_id)
    main(args.device)
