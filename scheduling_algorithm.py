import numpy as np
from switch_cli import run_simple_switch_cli
from config import service_port_mapping, bottleneck_config, services, cmab_k_value, cmab_alpha_value, priority_threshold, overlap_factor, gama, DEFAULT_PATH

# CMAB算法
class CMAB:
    def __init__(self, priorities, requirements):
        self.priorities = priorities
        self.requirements = requirements
        self.n_services = 3 # 三个服务
        self.n_paths = 2  # 每个服务有两条路径
        self.q_table = np.zeros((self.n_services, self.n_paths))  # Q表，存储每条路径的预期奖励
        self.counts = np.zeros((self.n_services, self.n_paths))  # 每条路径被选择的次数
        self.total_bandwidth_usage = np.zeros(self.n_services)  # 每个服务的带宽使用情况
        self.reward_history = [[] for _ in range(self.n_services * self.n_paths)]  # 存储奖励历史用于方差计算
        self.alpha = cmab_alpha_value  # 带宽衰减系数
        self.k = cmab_k_value  # UCB 缩放因子  ，多尝试一些值看看，如0.5，1.0，2.0
        self.total_reward = 0  # 统计每次调度的总奖励值，看是否逐渐升高
        self.selected_paths = set()  # 记录高优先级服务选择的路径
        self.overlap_factor = overlap_factor
        self.priority_threshold = priority_threshold  # 可调整的阈值
        self.gama = gama
        self.path_choices = {}  # 记录每个服务的路径选择

    # 基于UCB策略选择路径，这里可以扩展，基于贪婪，基于TS
    def choose_path(self, service):
        service_idx = services.index(service)
        total_counts = np.sum(self.counts[service_idx])
        
        if total_counts == 0: 
            if ((self.priorities[service] / max(self.priorities.values())) > self.priority_threshold):
                return 1-DEFAULT_PATH  # 如果服务没有选择过路径，则优先选择非默认路径
            else:
                return np.random.choice(self.n_paths)
        path_vars = [np.var(self.reward_history[service_idx * self.n_paths + p]) 
                            if len(self.reward_history[service_idx * self.n_paths + p]) > 1 else 1.0 
                            for p in range(self.n_paths)]
        c_t = self.k * np.sqrt(path_vars)        
        ucb_values = self.q_table[service_idx] + c_t * np.sqrt(np.log(total_counts) / (self.counts[service_idx] + 1e-5))
        return np.argmax(ucb_values)  # 选择Q值最大的路径
 
    def update_q_table(self, service, path, reward):
        service_idx = services.index(service)
        self.counts[service_idx, path] += 1
        alpha = cmab_alpha_value  # 可调的学习率
        self.q_table[service_idx, path] = (1 - alpha) * self.q_table[service_idx, path] + alpha * reward
    
    def calculate_reward(self, service, path, path_stats):
        delay_current, jitter, throughput, loss_rate_current, load = path_stats[service][path][-1]
        bandwidth = bottleneck_config[path][0] - throughput
        load = throughput / bottleneck_config[path][0]  # [0, 1]
        required_bandwidth = self.requirements[service]['bandwidth']
        # required_loss_rate = self.requirements[service]['loss_rate']
        priority = self.priorities[service] / max(self.priorities.values())

        required_bandwidth = self.requirements[service]['bandwidth']
        required_jitter = self.requirements[service]['jitter']
        # required_loss_rate = self.requirements[service]['loss_rate']

        # 计算历史状态的平均值，用于路径的历史表现评估
        historical_jitter = np.mean([x[1] for x in path_stats[service][path][:-1]])
        historical_bandwidth = np.mean([abs(bottleneck_config[path][0]-x[2]) for x in path_stats[service][path][:-1]])   
        # historical_loss_rate = np.mean([x[3] for x in path_stats[service][path][:-1]])
        # historical_load = np.mean([x[4] for x in path_stats[service][path][:-1]])
       
        other_path = 1 - path

        delay_other, _, _, loss_rate_other, _ = path_stats[service][other_path][-1]
        # 计算时延差
        delay_diff = delay_other - delay_current

        service_idx = services.index(service)

        # 计算奖励，加入历史状态的影响
        bandwidth_reward = 0 if bandwidth <= 0 else round(max(bandwidth / required_bandwidth - 1, 0) * (1 + historical_bandwidth / bandwidth), 4)
        jitter_reward = round(max(1 - jitter / required_jitter, 0) * (1 + historical_jitter / jitter), 4)
        delay_reward = 0 if delay_diff > 0 else self.gama

        # 服务的优先级作为上下文
        priority = self.priorities[service] / max(self.priorities.values())  # 标准化优先级
        load_threshold = 1.0 - 0.2 * priority
        
        # 重叠惩罚
        overlap_penalty = (1 - priority) * self.overlap_factor if path in self.selected_paths and priority < self.priority_threshold else 0
        loss_penalty = int(loss_rate_current <= loss_rate_other)  # 丢包惩罚0或1
        load_penalty = int(load > load_threshold)  # 负载惩罚0或1
        lap_penalty = 3 if service_idx == 1 and self.path_choices.get(services[0], None) == path else 0  # 服务2选择服务1的路径，惩罚为3，否则为0。

        # 总惩罚
        total_penalty = overlap_penalty + loss_penalty + load_penalty + lap_penalty
        # print("总惩罚：", total_penalty)
        # print("各奖励，时延，带宽，抖动：", delay_reward, bandwidth_reward, jitter_reward)

        reward = np.nan_to_num(priority * (delay_reward + bandwidth_reward + jitter_reward) - total_penalty)  
        self.total_reward += reward
        # print("统计每次调度总奖励：", self.total_reward)
        self.reward_history[services.index(service) * self.n_paths + path].append(reward) 
        return reward

    def schedule(self, service, path_stats):
        path = self.choose_path(service)
        print(f"{service}选择路径：", path)
        self.path_choices[service] = path
        # path选择出来后，直接下发流表
        for i in range(len(service_port_mapping[service])):
            run_simple_switch_cli(service_port_mapping[service][i], path)
        
        if self.priorities[service] / max(self.priorities.values()) >= self.priority_threshold:
            self.selected_paths.clear()
            self.selected_paths.add(path)
        reward = self.calculate_reward(service, path, path_stats)  # 计算奖励
        print(service, ":", reward)
        self.update_q_table(service, path, reward)  # 更新Q表



    """
    def choose_path(self, service):
        service_idx = services.index(service)
        total_counts = np.sum(self.counts[service_idx])
        
        if total_counts == 0:
            return np.random.choice(self.n_paths)  # 如果没有选择过路径，则随机选择
        path_vars = [np.var(self.reward_history[service_idx * self.n_paths + p]) 
                            if len(self.reward_history[service_idx * self.n_paths + p]) > 1 else 1.0 
                            for p in range(self.n_paths)]
        c_t = self.k * np.sqrt(path_vars)        
        ucb_values = self.q_table[service_idx] + c_t * np.sqrt(np.log(total_counts) / (self.counts[service_idx] + 1e-5))
        return np.argmax(ucb_values)  # 选择Q值最大的路径

    def update_q_table(self, service, path, reward):
        service_idx = services.index(service)
        self.counts[service_idx, path] += 1
        # 更新Q值
        self.q_table[service_idx, path] += (reward - self.q_table[service_idx, path]) / self.counts[service_idx, path]

    def calculate_reward(self, service, path):
        # 计算路径的奖励，加入路径历史状态和公平性作为上下文
        throughput, jitter, loss_rate, load = path_stats[service][path][-1]
        # 计算剩余带宽和瓶颈链路负载
        bandwidth = bottleneck_config[path][0] - throughput
        load /= bottleneck_config[path][0]
        
        required_bandwidth = self.requirements[service]['bandwidth']
        required_jitter = self.requirements[service]['jitter']
        required_loss_rate = self.requirements[service]['loss_rate']
       
        # 服务的优先级作为上下文
        priority = self.priorities[service] / max(self.priorities.values())  # 标准化优先级
        
        # 计算历史状态的平均值，用于路径的历史表现评估
        historical_bandwidth = np.mean([bottleneck_config[path][0]-x[0] for x in path_stats[service][path][:-1]])
        historical_jitter = np.mean([x[1] for x in path_stats[service][path][:-1]])
        historical_loss_rate = np.mean([x[2] for x in path_stats[service][path][:-1]])
        historical_load = np.mean([x[3] for x in path_stats[service][path][:-1]])

        # 计算奖励，加入历史状态的影响
        bandwidth_reward = min(bandwidth / required_bandwidth, 1) * (1 + historical_bandwidth / bandwidth)
        if jitter == 0:
            jitter_reward = 1.0
        else:
            jitter_reward = max(1 - jitter / required_jitter, 0) * (1 + historical_jitter / jitter)

        load_threshold = 0.8 + 0.2 * priority
        # loss_penalty = max(loss_rate / required_loss_rate, 1)  # 简单比值，无历史放大
        # load_penalty = max((load - load_threshold, 0))  # 标准化到 [0, inf]
        if loss_rate == 0:
            loss_penalty = 0.0
        else:
            loss_penalty = max(loss_rate, required_loss_rate) * (1 + historical_loss_rate / loss_rate)
        if load == 0:
            load_penalty = 0.0
        else:
            load_penalty = max(load - load_threshold, 0) * (1 + historical_load / load)

        # **引入公平性**
        # 计算带宽使用的公平性调整，避免某个服务占用过多资源
        total_bandwidth = np.sum(self.total_bandwidth_usage) + 1e-5
        service_bandwidth = self.total_bandwidth_usage[self.services.index(service)]
        fairness_factor = 1 - (service_bandwidth / total_bandwidth)
        fairness_penalty = max(0, fairness_factor)
        # reward = np.nan_to_num(priority * (bandwidth_reward + jitter_reward) - loss_penalty - load_penalty)
         # 记录奖励
        
        # 总奖励，结合服务的优先级（上下文信息）、历史状态和公平性
        # 从Q值的更新公式可以看出，Q值变负是r-Q为负数，正常收敛情况是r-Q是一个[-1,1]内的数
        # reward = np.nan_to_num(priority * ( bandwidth_reward + jitter_reward) - loss_penalty - load_penalty - fairness_penalty)        
        reward = np.nan_to_num(priority * ( bandwidth_reward + jitter_reward) - loss_penalty - load_penalty)        
        self.reward_history[services.index(service) * self.n_paths + path].append(reward) 
        # reward = np.clip(raw_reward, -1, 1)  # 限制在 [-1, 1]
        return reward
    """



# RR


# minRTT
