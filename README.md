# 🚗 Deep Reinforcement Learning — Autonomous Highway Driving

> A Deep Q-Network (DQN) agent trained from scratch to navigate a multi-lane highway, maximising velocity while avoiding collisions — visualised live in the browser.

---

## Table of Contents

1. [What is Reinforcement Learning?](#1-what-is-reinforcement-learning)
2. [The Markov Decision Process](#2-the-markov-decision-process)
3. [Q-Learning and the Bellman Equation](#3-q-learning-and-the-bellman-equation)
4. [Deep Q-Networks (DQN)](#4-deep-q-networks-dqn)
5. [Key Stabilisation Techniques](#5-key-stabilisation-techniques)
6. [The Highway Environment](#6-the-highway-environment)
7. [Architecture of This Implementation](#7-architecture-of-this-implementation)
8. [How to Run](#8-how-to-run)
9. [What to Observe in the UI](#9-what-to-observe-in-the-ui)
10. [Results and Learning Curve](#10-results-and-learning-curve)

---

## 1. What is Reinforcement Learning?

Reinforcement Learning (RL) is a paradigm of machine learning in which an **agent** learns to make sequential decisions by interacting with an **environment**. Unlike supervised learning — where a model is trained on labelled input-output pairs — RL provides no explicit ground truth. Instead, the agent receives a **reward signal** after each action and must discover, through trial and error, which sequence of actions maximises its long-term cumulative reward.

The core loop is elegantly simple:

```
Agent  ──action──▶  Environment
Agent  ◀──state──   Environment
Agent  ◀──reward──  Environment
```

At every discrete timestep `t`, the agent observes the current state `sₜ`, selects an action `aₜ`, transitions to a new state `sₜ₊₁`, and receives a scalar reward `rₜ`. The objective is to learn a **policy** `π(a | s)` — a mapping from states to actions — that maximises the expected discounted return:

```
G_t = Σ_{k=0}^{∞}  γᵏ · r_{t+k+1}
```

where `γ ∈ [0, 1)` is the **discount factor** that controls how much the agent values future rewards relative to immediate ones. A value of `γ = 0.99` (as used here) means the agent is highly forward-looking, planning many steps ahead.

---

## 2. The Markov Decision Process

RL problems are formally modelled as a **Markov Decision Process (MDP)**, defined by the tuple:

```
M = (S, A, P, R, γ)
```

| Symbol | Meaning |
|--------|---------|
| `S`    | State space — all possible observations of the environment |
| `A`    | Action space — all possible actions the agent can take |
| `P(s'│s,a)` | Transition probability — likelihood of reaching state `s'` from `s` via action `a` |
| `R(s,a,s')` | Reward function — scalar signal received after each transition |
| `γ`    | Discount factor |

The **Markov property** asserts that the future is conditionally independent of the past given the present state:

```
P(sₜ₊₁ | sₜ, aₜ, sₜ₋₁, aₜ₋₁, ...) = P(sₜ₊₁ | sₜ, aₜ)
```

This is a powerful assumption — it means the agent only needs to know *where it is now*, not the entire history of how it got there.

---

## 3. Q-Learning and the Bellman Equation

The **action-value function** (Q-function) quantifies the expected return when taking action `a` in state `s` and thereafter following policy `π`:

```
Q^π(s, a) = E_π [ G_t | sₜ = s, aₜ = a ]
           = E_π [ rₜ + γ · Q^π(sₜ₊₁, aₜ₊₁) | sₜ = s, aₜ = a ]
```

The optimal Q-function `Q*(s, a)` satisfies the **Bellman Optimality Equation**:

```
Q*(s, a) = E [ r + γ · max_{a'} Q*(s', a') | s, a ]
```

This recursive identity is the cornerstone of Q-learning. The optimal policy is then simply:

```
π*(s) = argmax_a  Q*(s, a)
```

In tabular Q-learning, we maintain a lookup table and update it via:

```
Q(s, a) ← Q(s, a) + α · [ r + γ · max_{a'} Q(s', a') − Q(s, a) ]
```

where `α` is the learning rate. The term in brackets is the **Temporal Difference (TD) error** — the discrepancy between our current estimate and the bootstrapped target.

---

## 4. Deep Q-Networks (DQN)

Tabular Q-learning breaks down when the state space is continuous or high-dimensional — as is the case in autonomous driving, where the state encodes positions, velocities, and headings of multiple vehicles simultaneously.

**DQN** (Mnih et al., 2015) replaces the Q-table with a deep neural network parameterised by weights `θ`:

```
Q(s, a ; θ) ≈ Q*(s, a)
```

The network takes a state vector `s` as input and outputs a Q-value for every possible action simultaneously. The agent selects:

```
aₜ = argmax_a  Q(sₜ, a ; θ)
```

The network is trained by minimising the **mean-squared Bellman error** over a mini-batch of transitions `(s, a, r, s', done)` sampled from a replay buffer:

```
L(θ) = E [ ( yₜ − Q(s, a ; θ) )² ]

where  yₜ = r + γ · (1 − done) · max_{a'} Q(s', a' ; θ⁻)
```

`θ⁻` denotes the **target network** weights (explained below). Gradients flow back through the network via standard backpropagation, and weights are updated with Adam optimiser.

### Network Architecture

```
Input (state vector, flattened)
        │
   Linear(state_dim → 128)  +  ReLU
        │
   Linear(128 → 128)        +  ReLU
        │
   Linear(128 → action_dim)
        │
Output: Q-value for each action
```

---

## 5. Key Stabilisation Techniques

Naïvely applying gradient descent to the Bellman equation is notoriously unstable. DQN introduced two critical innovations:

### 5.1 Experience Replay

Rather than learning from consecutive, highly correlated transitions, the agent stores every `(s, a, r, s', done)` tuple in a **replay buffer** `D` of capacity `N = 10,000`. At each training step, a random mini-batch of size `B = 64` is sampled:

```
(sᵢ, aᵢ, rᵢ, s'ᵢ, doneᵢ) ~ Uniform(D)
```

This breaks temporal correlations, dramatically reducing variance in gradient estimates and improving sample efficiency.

### 5.2 Target Network

The training target `yₜ` depends on the same network being updated — creating a moving target problem analogous to chasing a shadow. DQN resolves this by maintaining a **frozen copy** of the network `θ⁻` that is only synchronised with the live network every `C` steps:

```
θ⁻ ← θ   (every C = 10 episodes)
```

This decouples the target from the gradient update, providing a stable learning signal.

### 5.3 ε-Greedy Exploration

The agent faces the classic **exploration-exploitation dilemma**: should it exploit its current best-known action, or explore new actions that might yield higher long-term reward?

We use an **ε-greedy** policy:

```
aₜ = { random action          with probability ε
      { argmax_a Q(sₜ, a ; θ)  with probability 1 − ε
```

`ε` is annealed exponentially over training:

```
εₜ = max(0.01,  ε₀ × 0.99ᵗ)
```

Early in training (ε ≈ 1.0), the agent explores almost entirely at random. As training progresses, it increasingly exploits its learned policy. By episode ~460, ε reaches 0.01 — meaning the agent acts greedily 99% of the time.

---

## 6. The Highway Environment

This project uses [`highway-env`](https://github.com/Farama-Foundation/HighwayEnv) — a lightweight, physics-based autonomous driving simulator built on top of `gymnasium`.

### State Space

The observation is a **kinematic matrix** of shape `(10, 5)` — representing up to 10 vehicles (ego + 9 nearest neighbours), each described by:

```
[presence, x, y, vx, vy]
```

This is flattened to a vector of dimension `50` before being fed to the network.

### Action Space

The agent selects from 5 discrete meta-actions:

| ID | Action | Description |
|----|--------|-------------|
| 0  | LANE LEFT  | Change to the left lane |
| 1  | IDLE       | Maintain current speed and lane |
| 2  | LANE RIGHT | Change to the right lane |
| 3  | FASTER     | Increase target speed |
| 4  | SLOWER     | Decrease target speed |

### Reward Function

```
r = v / v_max  −  collision_penalty
```

The agent is rewarded proportionally to its normalised speed and penalised heavily upon collision. This incentivises fast, safe driving — exactly the behaviour we want to emerge.

---

## 7. Architecture of This Implementation

```
rl-driving/
├── app.py                  # Flask server + RL training loop (background thread)
├── main.py                 # Standalone training script (pygame window)
├── templates/
│   └── index.html          # Browser UI — highway canvas + live stats panel
└── README.md
```

### Data Flow

```
[highway-env simulation]
        │  vehicle positions, rewards
        ▼
[Python training thread]  ──  DQN forward/backward pass
        │  JSON state snapshot
        ▼
[Flask SSE endpoint /stream]
        │  Server-Sent Events (~12 fps)
        ▼
[Browser — index.html]
        │  Canvas rendering + stats update
        ▼
[You — watching the agent learn in real time]
```

The simulation runs in a **daemon thread** completely decoupled from the web server. A shared dictionary protected by a `threading.Lock` carries the latest state snapshot. The `/stream` endpoint pushes this snapshot to the browser every 80 ms via **Server-Sent Events (SSE)** — a lightweight, unidirectional HTTP streaming protocol that requires no WebSocket handshake.

---

## 8. How to Run

### Prerequisites

```bash
pip install flask gymnasium highway-env torch
```

### Start the server

```bash
python app.py
```

### Open the UI

Navigate to **[http://localhost:5051](http://localhost:5051)** in your browser.

The RL agent begins training immediately in the background. The browser UI updates in real time — no refresh needed.

---

## 9. What to Observe in the UI

| What you see | What it means |
|---|---|
| Orange car driving erratically | Early episodes — agent is exploring randomly (ε ≈ 1.0) |
| Orange car changing lanes frequently | Agent discovering that lane changes help avoid slow traffic |
| ε decreasing on the HUD | Exploration rate decaying — agent becoming more confident |
| Reward chart trending upward | Agent learning a better policy over time |
| Fewer collision flashes | Agent has learned to anticipate and avoid obstacles |
| Orange car consistently in fast lane | Convergence — agent has found a near-optimal policy |

---

## 10. Results and Learning Curve

A well-trained DQN agent on `highway-v0` typically achieves:

- **Episode reward > 25** after ~100 episodes
- **Collision rate < 10%** after ~150 episodes
- **Near-optimal lane-keeping and overtaking** after ~200 episodes

The reward curve follows a characteristic shape:

```
Reward
  │                                          ·····
  │                                    ·····
  │                              ·····
  │                        ·····
  │                  ·····
  │            ·····
  │      ·····
  │·····
  └──────────────────────────────────────── Episode
   0    50   100   150   200   250   300
```

Early variance is high due to random exploration. As ε decays and the replay buffer fills with quality transitions, the learning signal stabilises and performance improves monotonically.

---

## References

- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518, 529–533.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Leurent, E. (2018). *An Environment for Autonomous Driving Decision-Making*. [highway-env](https://github.com/Farama-Foundation/HighwayEnv).
- Watkins, C. J. C. H., & Dayan, P. (1992). *Q-learning*. Machine Learning, 8(3–4), 279–292.

---

*Built with PyTorch · highway-env · Flask · HTML5 Canvas*
