import gymnasium as gym
import highway_env  # registers highway-v0
import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import threading
from collections import deque
from flask import Flask, Response, render_template

app = Flask(__name__)

# ── Neural Network ────────────────────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

# ── RL Agent ──────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.memory     = deque(maxlen=10000)
        self.gamma      = 0.99
        self.epsilon    = 1.0
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=1e-3)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        s = torch.FloatTensor(state).flatten().unsqueeze(0)
        return self.policy_net(s).argmax().item()

    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states      = torch.FloatTensor(states).view(batch_size, -1)
        actions     = torch.LongTensor(actions).unsqueeze(1)
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states).view(batch_size, -1)
        dones       = torch.FloatTensor(dones)
        curr_q      = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q   = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        loss = nn.MSELoss()(curr_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ── Shared state ──────────────────────────────────────────────────────────────
ACTION_NAMES = ["LANE LEFT", "IDLE", "LANE RIGHT", "FASTER", "SLOWER"]

sim_state = {
    "episode": 0, "step": 0, "reward": 0.0, "epsilon": 1.0,
    "total_reward": 0.0, "vehicles": [], "action": 1,
    "action_name": "IDLE", "done": False,
    "episode_rewards": [], "running": False, "status": "Starting..."
}
state_lock = threading.Lock()

# ── Simulation thread ─────────────────────────────────────────────────────────
def run_simulation():
    env = gym.make("highway-v0", render_mode=None)
    env.unwrapped.configure({
        "vehicles_count": 15,
        "lanes_count": 4,
        "duration": 40,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy"],
            "normalize": False
        }
    })
    obs, _ = env.reset()
    state_dim  = obs.flatten().shape[0]
    action_dim = env.action_space.n
    agent      = Agent(state_dim, action_dim)

    with state_lock:
        sim_state["running"] = True
        sim_state["status"]  = "Training..."

    episode_rewards = []

    for episode in range(300):
        state, _ = env.reset()
        total_reward = 0.0

        for step in range(600):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.append((state, action, reward, next_state, done))
            state        = next_state
            total_reward += reward
            agent.train()

            # Extract vehicle data from the environment
            vehicles = []
            road = env.unwrapped.road
            for i, v in enumerate(road.vehicles[:12]):
                lane_idx = v.lane_index[2] if v.lane_index else 0
                vehicles.append({
                    "x":    float(v.position[0]),
                    "y":    float(v.position[1]),
                    "vx":   float(v.speed),
                    "lane": int(lane_idx),
                    "ego":  (v is env.unwrapped.vehicle)
                })

            with state_lock:
                sim_state.update({
                    "episode":         episode,
                    "step":            step,
                    "reward":          round(reward, 3),
                    "total_reward":    round(total_reward, 2),
                    "epsilon":         round(agent.epsilon, 3),
                    "vehicles":        vehicles,
                    "action":          action,
                    "action_name":     ACTION_NAMES[action] if action < len(ACTION_NAMES) else str(action),
                    "done":            done,
                    "episode_rewards": episode_rewards[-60:]
                })

            if done:
                break

        episode_rewards.append(round(total_reward, 2))
        agent.epsilon = max(0.01, agent.epsilon * 0.99)
        if episode % 10 == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    env.close()
    with state_lock:
        sim_state["running"] = False
        sim_state["status"]  = "Training complete"

threading.Thread(target=run_simulation, daemon=True).start()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/stream")
def stream():
    def event_stream():
        import time
        while True:
            with state_lock:
                data = json.dumps(sim_state)
            yield f"data: {data}\n\n"
            time.sleep(0.08)
    return Response(event_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

if __name__ == "__main__":
    app.run(debug=False, threaded=True, port=5051)
