import subprocess
import numpy as np
import random

# traceroute 可以查看到每一跳的时延

# 调度函数没有问题了，然后是利用CMAB进行决策
# CMAB算法需要处理好输入，能不能直接用globecom的，但那个的问题是路径状态在客户端上获得，路径状态可以用INT。
# 那应用层指标呢，理论上吞可用带宽越大，清晰度越高，下载时间也需要考虑。
# 可以这样，输入是历史路径的时延和吞吐量，以及一个传输时间（？这个得斟酌一下）
# 先跑通一版吧，实现INT测试路径的单向时延和吞吐量以及剩余带宽（其实丢包率也可以考虑进去），服务需求则在奖励函数中实现。
# 奖励函数包括服务的带宽，时延，丢包率需求，还有服务的优先级和路径的带宽利用率等。优先级越高的奖励越大，进而保障获取更多的资源

class CMAB:
    def __init__(self, services, path_stats, priorities, requirements):
        self.services = services
        self.path_stats = path_stats
        self.priorities = priorities
        self.requirements = requirements
        self.n_services = len(services)
        self.n_paths = 2  # 每个服务有两条路径
        self.q_table = np.zeros((self.n_services, self.n_paths))  # Q表，存储每条路径的预期奖励
        self.counts = np.zeros((self.n_services, self.n_paths))  # 每条路径被选择的次数
        self.total_bandwidth_usage = np.zeros(self.n_services)  # 每个服务的带宽使用情况

    # 基于UCB策略选择路径，这里可以扩展，基于贪婪，基于TS
    def choose_path(self, service):
        service_idx = self.services.index(service)
        total_counts = np.sum(self.counts[service_idx])
        
        if total_counts == 0:
            return np.random.choice(self.n_paths)  # 如果没有选择过路径，则随机选择
        ucb_values = self.q_table[service_idx] + np.sqrt(2 * np.log(total_counts) / (self.counts[service_idx] + 1e-5))
        return np.argmax(ucb_values)  # 选择Q值最大的路径

    def update_q_table(self, service, path, reward):
        service_idx = self.services.index(service)
        self.counts[service_idx, path] += 1
        # 更新Q值
        self.q_table[service_idx, path] += (reward - self.q_table[service_idx, path]) / self.counts[service_idx, path]

    def calculate_reward(self, service, path):
        # 计算路径的奖励，加入路径历史状态和公平性作为上下文
        bandwidth, latency, loss_rate, load = self.path_stats[service][path][-1]
        required_bandwidth = self.requirements[service]['bandwidth']
        required_latency = self.requirements[service]['latency']
        required_loss_rate = self.requirements[service]['loss_rate']

        
        # 服务的优先级作为上下文
        priority = self.priorities[service]
        
        # 计算历史状态的平均值，用于路径的历史表现评估
        historical_bandwidth = np.mean([x[0] for x in path_stats[service][path]])
        historical_latency = np.mean([x[1] for x in path_stats[service][path]])
        historical_loss_rate = np.mean([x[2] for x in path_stats[service][path]])
        historical_load = np.mean([x[3] for x in path_stats[service][path]])

        # 计算奖励，加入历史状态的影响
        bandwidth_reward = min(bandwidth / required_bandwidth, 1) * (1 + historical_bandwidth / bandwidth)
        latency_reward = max(1 - latency / required_latency, 0) * (1 + historical_latency / latency)
    
        loss_penalty = max(loss_rate, required_loss_rate) * (1 + historical_loss_rate / loss_rate)
        load_penalty = max(load, 0.9) * (1 + historical_load / load)

        # **引入公平性**
        # 计算带宽使用的公平性调整，避免某个服务占用过多资源
        total_bandwidth = np.sum(self.total_bandwidth_usage)  # 带宽总体使用是如何计算的？
        if total_bandwidth == 0:
            fairness_factor = 0  # 防止除以零
        else:
            service_bandwidth = self.total_bandwidth_usage[self.services.index(service)]
            fairness_factor = 1 - (service_bandwidth / total_bandwidth)  # 公平性因子
        fairness_penalty = max(0, fairness_factor)  # 如果服务占用过多带宽，公平性惩罚增大
        
        # 总奖励，结合服务的优先级（上下文信息）、历史状态和公平性
        reward = np.nan_to_num(priority * ( bandwidth_reward + latency_reward) - loss_penalty - load_penalty - fairness_penalty)
        
        return reward

    def schedule(self):
        for service in self.services:
            # 为每个服务选择路径
            path = self.choose_path(service)
            reward = self.calculate_reward(service, path)  # 计算奖励
            print(service, ":", reward)
            self.update_q_table(service, path, reward)  # 更新Q表

            # 更新服务的带宽使用情况
            bandwidth, _, _, _ = self.path_stats[service][path][-1]
            self.total_bandwidth_usage[self.services.index(service)] += bandwidth

    def resource_balance(self):
        # 资源平衡：根据当前带宽使用情况调整每个服务的路径选择
        total_bandwidth = np.sum(self.total_bandwidth_usage)
        for i, service in enumerate(self.services):
            service_bandwidth = self.total_bandwidth_usage[i]
            if service_bandwidth > total_bandwidth / len(self.services):  # 如果某服务占用过多带宽
                # 给低带宽使用的服务优先分配资源
                self.total_bandwidth_usage[i] *= 0.8  # 稍微降低占用过多带宽的服务使用
            else:
                self.total_bandwidth_usage[i] *= 1.2  # 提供更多带宽给低带宽使用的服务


def send_cli_commands(commands, thrift_port):
    """
    向 simple_switch_CLI 发送一系列命令，并返回输出。
    
    :param commands: 要发送的命令列表
    :param thrift_port: Thrift 端口号，默认为 9090
    :return: 每个命令的输出结果
    """
    try:
        # 启动 simple_switch_CLI 进程
        with subprocess.Popen(
            ['simple_switch_CLI', '--thrift-port', str(thrift_port)],
            stdin=subprocess.PIPE,  # 允许写入命令
            stdout=subprocess.PIPE, # 获取命令输出
            stderr=subprocess.PIPE,
            text=True  # 自动处理文本（字符串）
        ) as cli_process:
            outputs = []
            
            for command in commands:
                cli_process.stdin.write(f'{command}\n')
                cli_process.stdin.flush()  # 确保命令已发送
            
            stdout, stderr = cli_process.communicate()
            
            if stderr:
                outputs.append(f"Error: {stderr}")
            else:
                outputs.append(stdout)

            return outputs
    except Exception as e:
        return [f"Failed to run CLI: {e}"]

def run_simple_switch_cli(thrift_port, path_id):
    # 要发送的命令列表
    # 0代表第一条路，1代表第二条路
    commands = [
        f'register_write select_path 0 {path_id}',
        'register_read select_path 0'
    ]
    
    outputs = send_cli_commands(commands, thrift_port)
    
    for output in outputs:
        print(output)

if __name__ == "__main__":
    # 定义服务优先级和服务需求
    services = ['Service 1', 'Service 2', 'Service 3']
    # 加个映射
    # 这个priorities的值也是经验值，需要调试。
    priorities = {'Service 1': 3, 'Service 2': 2, 'Service 3': 1.5}  # 高数值表示高优先级
    requirements = {'Service 1': {'bandwidth': 100, 'latency': 50, 'loss_rate': 0.01},
                    'Service 2': {'bandwidth': 200, 'latency': 30, 'loss_rate': 0.02}, 
                    'Service 3': {'bandwidth': 150, 'latency': 50, 'loss_rate': 0.03}}

    # 模拟路径的带宽、延迟、丢包率和带宽负载，以及历史状态（例如最近几轮的值）,带宽=固有带宽-吞吐量
    # 带宽负载就是 路径吞吐量/路径固有带宽，带宽负载越大越不好
    # 路径和服务是绑定的 
    # 路径状态（剩余带宽，时延，丢包率，带宽负载）


    path_stats = {
        'Service 1': {
            0 : [(100, 50, 0.01, 0.4), (110, 52, 0.015, 0.38)], 
            1: [(120, 55, 0.02, 0.3), (125, 57, 0.025, 0.32)]
        },
        'Service 2': {
            0: [(200, 30, 0.01, 0.5), (190, 32, 0.015, 0.55)],
            1: [(180, 35, 0.02, 0.6), (175, 36, 0.025, 0.58)]
        },
        'Service 3': {
            0: [(150, 40, 0.02, 0.3), (155, 42, 0.022, 0.42)],
            1: [(160, 45, 0.03, 0.5), (165, 47, 0.035, 0.48)]
        }
    } # 路径(吞吐量, 延迟, 丢包率, 带宽负载)，通过索引来区分路径，索引为0为路径1，1为路径2

    run_simple_switch_cli(thrift_port=9090,path_id=1)
    # 创建CMAB实例
    # 所以输入需要优先级，需求列表，路径状态字典（但需求，优先级这些都是固定的，本质上只要路径状态）
    cmab = CMAB(services, path_stats, priorities, requirements)
