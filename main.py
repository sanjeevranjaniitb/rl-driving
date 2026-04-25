import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import time # Optional: to control frame rate
import highway_env # Import the highway environment

# 1. The "Brain": Multi-Layer Perceptron
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# 2. The Specialist: RL Agent
class Agent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0 # Exploration rate

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        # Flatten the state before passing to the network
        state = torch.FloatTensor(state).flatten().unsqueeze(0)
        return self.policy_net(state).argmax().item()

    def train(self, batch_size=64):
        if len(self.memory) < batch_size: return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Flatten states and next_states
        states = torch.FloatTensor(states).view(batch_size, -1)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states).view(batch_size, -1)
        dones = torch.FloatTensor(dones)

        # Current Q-values
        curr_q = self.policy_net(states).gather(1, actions)

        # Target Q-values using the Target Network (Stability)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(curr_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# 3. The Execution Loop
env = gym.make("highway-v0", render_mode="human") # Changed render_mode to 'human'
# Configure environment for birdseye view to 'fix' lanes and only move vehicles
env.unwrapped.configure({
    "camera_type": "birdseye",
    "screen_width": 1200, # Increased width for better resolution
    "screen_height": 600, # Increased height for better resolution
    "scaling": 2.5 # Increased scaling for larger objects
})

# Re-initialize the environment to get correct dimensions for the agent
obs, info = env.reset()

# Calculate flattened state dimension
highway_state_dim_flat = obs.flatten().shape[0]
highway_action_dim = env.action_space.n

agent = Agent(highway_state_dim_flat, highway_action_dim)

episode_rewards = [] # List to store total reward for each episode

for episode in range(200):
    state, _ = env.reset()
    total_reward = 0

    # In local setup, render_mode='human' directly opens a window
    # No need for IPython.display or matplotlib in the loop for live visualization

    # The highway-env episodes can be much longer, limit steps to avoid very long runs
    for t in range(500):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        agent.train()

        env.render() # Render the environment for each step

        # Optional: Add a small delay for better visual experience
        # You can adjust or remove this based on your preference
        if episode < 10: # For the first 10 episodes, very slow
            time.sleep(0.2)
        elif episode < 50: # For next 40 episodes, moderate speed
            time.sleep(0.05)
        else: # For later episodes, near real-time
            time.sleep(0.01)

        if done:
            break

    episode_rewards.append(total_reward) # Store reward for this episode
    # Decaying exploration and updating Target Network
    agent.epsilon = max(0.01, agent.epsilon * 0.99) # Ensure epsilon does not go below 0.01
    if episode % 10 == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
    # Print reward for every episode
    print(f"Episode {episode}, Reward: {total_reward}")

env.close()

# Plotting the rewards over episodes (this will still generate a matplotlib window)
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Training on Highway-v0: Rewards per Episode")
plt.grid(True)
plt.show()