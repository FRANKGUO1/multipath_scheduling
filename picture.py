import pandas as pd
import matplotlib.pyplot as plt
from config import requirements

# 读取 CSV 文件
file1 = pd.read_csv('iperf_s1_CMAB_results.csv')
file2 = pd.read_csv('iperf_s1_RR_results.csv')
file3 = pd.read_csv('iperf_s1_MinR_results.csv')

# 提取每个文件的对应列
columns = ['Bandwidth (Mbps)', 'jitter (ms)', 'Lost', 'Total', 'Loss Rate (%)', 'Latency_avg(ms)']
data1 = file1[columns]
data2 = file2[columns]
data3 = file3[columns]


# 定义基准线值（你可以根据需要修改这些值）
baseline_bandwidth = requirements['Service1']['bandwidth']  # Bandwidth基准线值
baseline_jitter = requirements['Service2']['jitter']      # Jitter基准线值
baseline_loss = requirements['Service3']['loss_rate']        # Loss Rate基准线值


# 创建 4 个子图
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# 绘制 Bandwidth
axs[0].plot(data1['Bandwidth (Mbps)'], label='CMAB', marker='^', linewidth=2)
axs[0].plot(data2['Bandwidth (Mbps)'], label='RR', marker='+', linewidth=1)
axs[0].plot(data3['Bandwidth (Mbps)'], label='MinRTT', marker='.', linewidth=1)
axs[0].set_title('Bandwidth (Mbps)')
axs[0].legend()

# 绘制 Jitter
axs[1].plot(data1['jitter (ms)'], label='CMAB', marker='^', linewidth=2)
axs[1].plot(data2['jitter (ms)'], label='RR', marker='+', linewidth=1)
axs[1].plot(data3['jitter (ms)'], label='MinRTT', marker='.', linewidth=1)
axs[1].set_title('Jitter (ms)')
axs[1].legend()


# 绘制 Loss Rate
axs[2].plot(data1['Loss Rate (%)'], label='CMAB', marker='^', linewidth=2)
axs[2].plot(data2['Loss Rate (%)'], label='RR', marker='+', linewidth=1)
axs[2].plot(data3['Loss Rate (%)'], label='MinRTT', marker='.', linewidth=1)
axs[2].set_title('Loss Rate (%)')
axs[2].legend()


# 调整布局并显示
plt.tight_layout()
plt.show()
