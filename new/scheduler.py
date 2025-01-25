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
    def __init__(self, num_paths, num_services, path_states, service_priorities):
        """
        service_priorities: 每个服务的优先级（3为最高优先级，1为最低）
        """
        self.num_paths = num_paths
        self.num_services = num_services
        self.path_states = path_states
        self.service_priorities = service_priorities
        self.q_values = np.zeros((num_services, num_paths))  # Q值初始化
        self.action_counts = np.zeros((num_services, num_paths))  # 每个服务-路径组合的选择次数
    
    def reward_function(self, service_id, path_id):
        """
        根据路径状态计算奖励。
        奖励函数设计：低延迟、高带宽、低丢包和低负载会获得更高的奖励，同时考虑服务优先级。
        """
        delay, bandwidth, loss_rate, load = self.path_states[path_id]
        priority = self.service_priorities[service_id]
        
        # 根据服务优先级调整奖励
        priority_factor = 1 / priority  # 优先级越高，factor越小，奖励越大
        
        # 计算路径的基本奖励
        reward = -delay + bandwidth * 0.5 - loss_rate * 1.0 - load * 0.5
        
        # 调整奖励，优先级较高的服务得到更高的奖励
        reward *= priority_factor
        
        return reward

    def select_action(self, service_id):
        """
        选择路径进行调度，使用CMAB的ε-greedy策略。
        """
        epsilon = 0.1  # 探索概率
        if random.random() < epsilon:
            # 随机选择路径
            return random.randint(0, self.num_paths - 1)
        else:
            # 选择Q值最高的路径
            return np.argmax(self.q_values[service_id])
    
    def update_q_values(self, service_id, path_id, reward):
        """
        更新Q值，采用增量更新公式。
        """
        self.action_counts[service_id, path_id] += 1
        # 使用Q值更新公式
        learning_rate = 1 / self.action_counts[service_id, path_id]
        self.q_values[service_id, path_id] += learning_rate * (reward - self.q_values[service_id, path_id])

    def run(self, num_rounds=1000):
        """
        运行CMAB调度。
        """
        for _ in range(num_rounds):
            for service_id in range(self.num_services):
                # 选择路径
                path_id = self.select_action(service_id)
                # 获取奖励
                reward = self.reward_function(service_id, path_id)
                # 更新Q值
                self.update_q_values(service_id, path_id, reward)
                print(f"Service {service_id} (Priority {self.service_priorities[service_id]}), Path {path_id}, Reward: {reward}, Q-value: {self.q_values[service_id, path_id]}")


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
    run_simple_switch_cli(thrift_port=9090,path_id=1)
