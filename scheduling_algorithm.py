import numpy as np
from switch_cli import run_simple_switch_cli
from config import service_port_mapping

# CMAB算法
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
        self.reward_history = [[] for _ in range(self.n_services * self.n_paths)]  # 存储奖励历史用于方差计算
        self.alpha = 0.9  # 带宽衰减系数
        self.k = 1.0  # UCB 缩放因子

    # 基于UCB策略选择路径，这里可以扩展，基于贪婪，基于TS
    def choose_path(self, service):
        service_idx = self.services.index(service)
        total_counts = np.sum(self.counts[service_idx])
        
        if total_counts == 0:
            return np.random.choice(self.n_paths)  # 如果没有选择过路径，则随机选择
        path_vars = [np.var(self.reward_history[service_idx * self.n_paths + p]) 
                            if len(self.reward_history[service_idx * self.n_paths + p]) > 1 else 1.0 
                            for p in range(self.n_paths)]
        c_t = self.k * np.sqrt(path_vars)        
        ucb_values = self.q_table[service_idx] + c_t * np.sqrt(np.log(total_counts) / (self.counts[service_idx] + 1e-5))
        # ucb_values = np.maximum(ucb_values, 0)  # 确保 UCB 不为负
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
        priority = self.priorities[service] / max(self.priorities.values())  # 标准化优先级
        
        # 计算历史状态的平均值，用于路径的历史表现评估
        historical_bandwidth = np.mean([x[0] for x in self.path_stats[service][path][:-1]])
        historical_latency = np.mean([x[1] for x in self.path_stats[service][path][:-1]])
        historical_loss_rate = np.mean([x[2] for x in self.path_stats[service][path][:-1]])
        historical_load = np.mean([x[3] for x in self.path_stats[service][path][:-1]])

        # 计算奖励，加入历史状态的影响
        bandwidth_reward = min(bandwidth / required_bandwidth, 1) * (1 + historical_bandwidth / bandwidth)
        latency_reward = max(1 - latency / required_latency, 0) * (1 + historical_latency / latency)

        load_threshold = 0.8 + 0.2 * priority
        loss_penalty = max(loss_rate / required_loss_rate, 1)  # 简单比值，无历史放大
        load_penalty = max((load - load_threshold, 0))  # 标准化到 [0, inf]
        # loss_penalty = max(loss_rate, required_loss_rate) * (1 + historical_loss_rate / loss_rate)
        # load_penalty = max(load - load_threshold, 0) * (1 + historical_load / load)

        # **引入公平性**
        # 计算带宽使用的公平性调整，避免某个服务占用过多资源
        total_bandwidth = np.sum(self.total_bandwidth_usage) + 1e-5
        service_bandwidth = self.total_bandwidth_usage[self.services.index(service)]
        fairness_factor = 1 - (service_bandwidth / total_bandwidth)
        fairness_penalty = max(0, fairness_factor)
        reward = np.nan_to_num(priority * (bandwidth_reward + latency_reward) - loss_penalty - load_penalty - fairness_penalty)
        self.reward_history[self.services.index(service) * self.n_paths + path].append(reward)  # 记录奖励
        
        # 总奖励，结合服务的优先级（上下文信息）、历史状态和公平性
        # 从Q值的更新公式可以看出，Q值变负是r-Q为负数，正常收敛情况是r-Q是一个[-1,1]内的数
        reward = np.nan_to_num(priority * ( bandwidth_reward + latency_reward) - loss_penalty - load_penalty - fairness_penalty)        
        # reward = np.clip(raw_reward, -1, 1)  # 限制在 [-1, 1]
        return reward

    def schedule(self):
        for service in self.services:
            # 为每个服务选择路径
            path = self.choose_path(service)
            # path选择出来后，直接下发流表
            for i in range(len(service_port_mapping[service])):
                run_simple_switch_cli(service_port_mapping[service][i], path)
            reward = self.calculate_reward(service, path)  # 计算奖励
            print(service, ":", reward)
            self.update_q_table(service, path, reward)  # 更新Q表

            # 更新服务的带宽使用情况
            bandwidth, _, _, _ = self.path_stats[service][path][-1]
            self.total_bandwidth_usage[self.services.index(service)] += bandwidth
            idx = self.services.index(service)
            # 指数衰减更新带宽使用
            self.total_bandwidth_usage[idx] = self.alpha * self.total_bandwidth_usage[idx] + (1 - self.alpha) * bandwidth
    
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


# RR


# minRTT
