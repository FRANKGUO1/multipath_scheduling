import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
file1 = pd.read_csv('iperf_s1_results.csv')
file2 = pd.read_csv('iperf_s2_results.csv')
file3 = pd.read_csv('iperf_s3_results.csv')

# 提取每个文件的对应列
columns = ['Bandwidth (Mbps)', 'jitter (ms)', 'Lost', 'Total', 'Loss Rate (%)', 'Latency_avg(ms)']
data1 = file1[columns]
data2 = file2[columns]
data3 = file3[columns]

# 设置图形大小
plt.figure(figsize=(10, 8))

# 创建 5 个子图
fig, axs = plt.subplots(3, 2, figsize=(12, 10))

# 绘制 Bandwidth
axs[0, 0].plot(data1['Bandwidth (Mbps)'], label='File 1', marker='o')
axs[0, 0].plot(data2['Bandwidth (Mbps)'], label='File 2', marker='x')
axs[0, 0].plot(data3['Bandwidth (Mbps)'], label='File 3', marker='s')
axs[0, 0].set_title('Bandwidth (Mbps)')
axs[0, 0].legend()

# 绘制 Jitter
axs[0, 1].plot(data1['jitter (ms)'], label='File 1', marker='o')
axs[0, 1].plot(data2['jitter (ms)'], label='File 2', marker='x')
axs[0, 1].plot(data3['jitter (ms)'], label='File 3', marker='s')
axs[0, 1].set_title('Jitter (ms)')
axs[0, 1].legend()

# 绘制 Lost
axs[1, 0].plot(data1['Lost'], label='File 1', marker='o')
axs[1, 0].plot(data2['Lost'], label='File 2', marker='x')
axs[1, 0].plot(data3['Lost'], label='File 3', marker='s')
axs[1, 0].set_title('Lost')
axs[1, 0].legend()

# 绘制 Total
axs[1, 1].plot(data1['Total'], label='File 1', marker='o')
axs[1, 1].plot(data2['Total'], label='File 2', marker='x')
axs[1, 1].plot(data3['Total'], label='File 3', marker='s')
axs[1, 1].set_title('Total')
axs[1, 1].legend()

# 绘制 Loss Rate
axs[2, 0].plot(data1['Loss Rate (%)'], label='File 1', marker='o')
axs[2, 0].plot(data2['Loss Rate (%)'], label='File 2', marker='x')
axs[2, 0].plot(data3['Loss Rate (%)'], label='File 3', marker='s')
axs[2, 0].set_title('Loss Rate (%)')
axs[2, 0].legend()

# 绘制 Latency_avg
axs[2, 1].plot(data1['Latency_avg(ms)'], label='File 1', marker='o')
axs[2, 1].plot(data2['Latency_avg(ms)'], label='File 2', marker='x')
axs[2, 1].plot(data3['Latency_avg(ms)'], label='File 3', marker='s')
axs[2, 1].set_title('Latency_avg (ms)')
axs[2, 1].legend()

# 调整布局并显示
plt.tight_layout()
plt.show()
