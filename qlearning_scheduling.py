import numpy

# 目前还缺就是解析决策结果，将决策结果变为路径id和清晰度，比特率，比特率水平，由parse_predication()函数实现
# 动作空间需要改良，变为和arm一样
# 还需要将任务添加到任务队列中，方便download函数调用
# 还有就是需不需要考虑码率（目前看是需要的，甚至应该用码率替换清晰度）

class MultipathVideoScheduler:
    def __init__(self, num_paths, bitrate_levels, rtt_bins=10, throughput_bins=10):
        """
        初始化Q-learning调度器
        
        参数:
        num_paths: 路径数量
        bitrate_levels: 可用的码率级别列表
        rtt_bins: RTT离散化的区间数
        throughput_bins: 吞吐量离散化的区间数
        """
        self.num_paths = num_paths
        self.bitrate_levels = bitrate_levels
        self.num_bitrates = len(bitrate_levels)
        
        # 状态空间离散化参数
        self.rtt_bins = rtt_bins
        self.throughput_bins = throughput_bins
        
        # 初始化Q表
        # 状态维度: [路径1_RTT, 路径1_throughput, 路径2_RTT, 路径2_throughput]
        # 动作维度: [选择的路径, 选择的码率]
        self.q_table = nump.zeros((
            rtt_bins, throughput_bins,  # 路径1的状态
            rtt_bins, throughput_bins,  # 路径2的状态
            num_paths, self.num_bitrates  # 动作空间
        ))
        
        # 学习参数
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # ε-贪婪策略的参数
        
    def discretize_state(self, path_states):
        """
        将连续的状态值离散化为Q表索引
        
        参数:
        path_states: 列表的列表，每个子列表包含[rtt, throughput]
        """
        discrete_state = []
        for path in path_states:
            rtt, throughput = path
            # 这里需要根据实际情况设置RTT和throughput的范围
            rtt_idx = min(int(rtt / 100 * self.rtt_bins), self.rtt_bins - 1)
            tput_idx = min(int(throughput / 10 * self.throughput_bins), self.throughput_bins - 1)
            discrete_state.extend([rtt_idx, tput_idx])
        return tuple(discrete_state)
    
    def calculate_reward(self, quality, rebuffering_ratio):
        """
        计算奖励值
        
        参数:
        quality: 视频质量评分
        rebuffering_ratio: 重缓冲比率
        """
        # 可以根据实际需求调整权重
        quality_weight = 1.0
        rebuffer_weight = -2.0
        
        return quality_weight * quality + rebuffer_weight * rebuffering_ratio
    
    def choose_action(self, state):
        """
        使用ε-贪婪策略选择动作
        
        参数:
        state: 离散化后的状态元组
        """
        if nump.random.random() < self.epsilon:
            # 探索：随机选择动作
            path = nump.random.randint(self.num_paths)
            bitrate = nump.random.randint(self.num_bitrates)
            return (path, bitrate)
        else:
            # 利用：选择Q值最大的动作
            path, bitrate = nump.unravel_index(
                nump.argmax(self.q_table[state]), 
                (self.num_paths, self.num_bitrates)
            )
            return (path, bitrate)
    
    def update(self, state, action, reward, next_state):
        """
        更新Q表
        
        参数:
        state: 当前状态
        action: 执行的动作 (path, bitrate)
        reward: 获得的奖励
        next_state: 下一个状态
        """
        path, bitrate = action
        current_q = self.q_table[state + (path, bitrate)]
        
        # 计算下一状态的最大Q值
        next_max_q = nump.max(self.q_table[next_state])
        
        # Q-learning更新公式
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        
        self.q_table[state + (path, bitrate)] = new_q
    
    def train_step(self, current_paths_state, action, quality, rebuffering_ratio, next_paths_state):
        """
        训练一个步骤
        
        参数:
        current_paths_state: 当前路径状态
        action: 执行的动作
        quality: 获得的视频质量
        rebuffering_ratio: 重缓冲比率
        next_paths_state: 下一个路径状态
        """
        current_state = self.discretize_state(current_paths_state)
        next_state = self.discretize_state(next_paths_state)
        reward = self.calculate_reward(quality, rebuffering_ratio)
        
        self.update(current_state, action, reward, next_state)


class SimpleStateDiscretizer:
    def __init__(self):
        # 为每条路径定义RTT和throughput的固定范围
        # RTT范围: <10, 10-50, 50-100, >100 (4个区间)
        self.rtt_ranges = [10, 50, 100]
        
        # Throughput范围: <5, 5-10, 10-20, 20-30, >30 (5个区间)
        self.throughput_ranges = [5, 10, 20, 30]
        
        # 计算单条路径的状态数
        self.rtt_states = len(self.rtt_ranges) + 1  # 4个状态
        self.throughput_states = len(self.throughput_ranges) + 1  # 5个状态
        
        # 单条路径的状态数 = RTT状态数 × Throughput状态数 = 4 × 5 = 20
        self.single_path_states = self.rtt_states * self.throughput_states
    
    def discretize_path_state(self, rtt, throughput):
        """
        将单条路径的RTT和throughput转换为离散状态
        
        参数:
        rtt: float, RTT值(ms)
        throughput: float, 吞吐量值(Mbps)
        
        返回:
        tuple: (rtt_state, throughput_state) 离散化后的状态编号
        """
        rtt_state = numpy.digitize(rtt, self.rtt_ranges)
        throughput_state = numpy.digitize(throughput, self.throughput_ranges)
        return (rtt_state, throughput_state)
    
    def get_path_state_number(self, rtt, throughput):
        """
        将单条路径的RTT和throughput转换为单个状态编号
        
        参数:
        rtt: float, RTT值(ms)
        throughput: float, 吞吐量值(Mbps)
        
        返回:
        int: 该路径的状态编号（从0开始）
        """
        rtt_state, throughput_state = self.discretize_path_state(rtt, throughput)
        return rtt_state * self.throughput_states + throughput_state
    
    def get_multi_path_state(self, path1_rtt, path1_throughput, path2_rtt, path2_throughput):
        """
        将两条路径的状态组合为一个总体状态编号
        
        参数:
        path1_rtt: float, 路径1的RTT值(ms)
        path1_throughput: float, 路径1的吞吐量值(Mbps)
        path2_rtt: float, 路径2的RTT值(ms)
        path2_throughput: float, 路径2的吞吐量值(Mbps)
        
        返回:
        int: 组合后的状态编号（从0开始）
        """
        # 获取两条路径各自的状态编号
        path1_state = self.get_path_state_number(path1_rtt, path1_throughput)
        path2_state = self.get_path_state_number(path2_rtt, path2_throughput)
        
        # 将两个路径的状态组合成一个总状态
        # path1_state的范围是0到19，path2_state也是0到19
        # 总状态 = path1_state * 20 + path2_state
        return path1_state * self.single_path_states + path2_state
    
    def get_state_space_size(self):
        """
        返回总的状态空间大小
        
        返回:
        int: 状态空间的总大小 (400 = 20 × 20)
        """
        # 总状态数 = 单条路径的状态数 × 单条路径的状态数
        # = (4 × 5) × (4 × 5) = 20 × 20 = 400
        return self.single_path_states * self.single_path_states


# 使用示例
def example_usage():
    discretizer = SimpleStateDiscretizer()
    
    # 测试不同的输入值
    state = discretizer.get_multi_path_state(
        path1_rtt=1,          # 路径1的RTT
        path1_throughput=1,   # 路径1的吞吐量
        path2_rtt=1,          # 路径2的RTT
        path2_throughput=1    # 路径2的吞吐量
    )

    
    print(f"状态空间总大小: {discretizer.get_state_space_size()}")
    print(state)
    
    
if __name__ == "__main__":
    example_usage() 