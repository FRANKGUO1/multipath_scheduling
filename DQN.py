import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random

# Define the DQN network architecture
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        
        # Neural network with 3 hidden layers，三个隐藏层
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Define the environment for multipath video streaming
class MultipathStreamingEnv:
    def __init__(self, bitrate_levels, history_length=3):
        self.bitrate_levels = bitrate_levels
        self.history_length = history_length
        self.paths = 2  # Number of available paths
        
        # Metrics history for each path
        self.throughput_history = [deque(maxlen=history_length) for _ in range(self.paths)]
        self.rtt_history = [deque(maxlen=history_length) for _ in range(self.paths)]
        
        # Initialize buffering metrics
        self.rebuffering_time = 0
        self.total_playback_time = 0
        
    def get_state(self):
        state = []
        
        for path in range(self.paths):
            # Current 3-second metrics
            current_throughput = np.mean(list(self.throughput_history[path])[-3:])
            current_rtt = np.mean(list(self.rtt_history[path])[-3:])
            
            # Historical statistics
            throughput_mean = np.mean(list(self.throughput_history[path]))
            throughput_std = np.std(list(self.throughput_history[path]))
            rtt_mean = np.mean(list(self.rtt_history[path]))
            rtt_std = np.std(list(self.rtt_history[path]))
            
            # Rate of change
            throughput_rate = (self.throughput_history[path][-1] - self.throughput_history[path][-2]) / self.throughput_history[path][-2]
            rtt_rate = (self.rtt_history[path][-1] - self.rtt_history[path][-2]) / self.rtt_history[path][-2]
            
            # Combine metrics for this path
            path_state = [
                current_throughput, current_rtt,
                throughput_mean, throughput_std,
                rtt_mean, rtt_std,
                throughput_rate, rtt_rate
            ]
            state.extend(path_state)
            
        return np.array(state)
    
    def calculate_reward(self, selected_bitrate):
        # Calculate rebuffering ratio
        rebuffering_ratio = self.rebuffering_time / (self.total_playback_time + 1e-6)
        
        # Normalize selected bitrate by maximum available bitrate
        bitrate_ratio = selected_bitrate / max(self.bitrate_levels)
        
        # Combine metrics with weights
        alpha = 0.7  # Weight for bitrate
        beta = 0.3   # Weight for rebuffering penalty
        
        reward = alpha * bitrate_ratio - beta * rebuffering_ratio
        return reward

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)
        
        self.batch_size = 64
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.Experience = namedtuple('Experience',
                                   ['state', 'action', 'reward', 'next_state', 'done'])
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.policy_net.network[-1].out_features)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        batch = self.Experience(*zip(*batch))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # Compute Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Example usage for video streaming
def main():
    # Define available bitrate levels (in Kbps)
    bitrate_levels = [4481.84, 22572.278, 35902.455, 55301.205, 71817.751, 87209.263]
    
    # Initialize environment and agent
    env = MultipathStreamingEnv(bitrate_levels)
    state_dim = 16  # 8 metrics per path * 2 paths
    action_dim = len(bitrate_levels) * 2  # bitrate levels * number of paths
    
    agent = DQNAgent(state_dim, action_dim)
    
    # Training loop
    num_episodes = 1000
    target_update = 10
    
    for episode in range(num_episodes):
        state = env.get_state()
        episode_reward = 0
        done = False
        
        while not done:
            # Select and perform action
            action = agent.select_action(state)
            selected_path = action % 2
            selected_bitrate = bitrate_levels[action // 2]
            
            # Get reward and next state
            reward = env.calculate_reward(selected_bitrate)
            next_state = env.get_state()
            
            # Store experience
            agent.memory.append(agent.Experience(state, action, reward, next_state, done))
            
            # Train the network
            agent.train()
            
            state = next_state
            episode_reward += reward
            
        # Update target network periodically
        if episode % target_update == 0:
            agent.update_target_network()
            
        print(f"Episode {episode + 1}, Total Reward: {episode_reward}")

if __name__ == "__main__":
    main()