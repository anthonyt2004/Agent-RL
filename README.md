# Reinforcement Learning From Scratch - Q-Learning vs REINFORCE

Implementation and comparison of two reinforcement learning algorithms **from scratch** (no RL library), applied to a grid navigation task. Built as a preliminary step of a larger research project on autonomous cyber defense of telecom infrastructures (in partnership with Orange S.A.).

---

## The Problem

A mouse navigates a 10×10 grid to collect cheese while avoiding poison cells. The agent learns purely from rewards - no hardcoded rules, no supervised signal.

| Cell type | Reward |
|-----------|--------|
| Cheese    | +50    |
| Poison    | −1000 (terminal) |
| Empty     | −5 + proximity bonus (up to +5) |

The proximity bonus encourages exploration toward cheese without revealing its location directly.

---

## Algorithms Implemented

Both algorithms are implemented **entirely from scratch using only NumPy** - no Stable Baselines, no PyTorch, no Gymnasium.

### Q-Learning (Sarsamax - Off-policy)

Learns a Q-table `Q(s, a)` estimating the value of each action in each state. Uses an **ε-greedy exploration policy** with exponential decay (`ε` from 1.0 → 0.01).

**Update rule (Bellman equation):**
$Q(s, a) \leftarrow Q(s, a) + \alpha \cdot [r + \gamma \cdot \max_{a'} Q(s', a') − Q(s, a)]$

Key implementation details:
- Invalid actions (wall collisions) are masked before selecting the greedy action
- Tie-breaking is randomized among equally-valued actions to avoid deterministic loops
- Board state is copied per episode to allow cheese depletion tracking

### REINFORCE (Policy Gradient - On-policy)

Directly optimizes a stochastic policy $\pi_\theta(a|s)$ parameterized by a preference table $\theta$. Uses a **softmax policy** with invalid action masking (set to −1e9 before softmax).

**Update rule:**
$\theta(s, a) \leftarrow \theta(s, a) + \alpha \cdot G_t \cdot \nabla \log \pi_\theta(a_t | s_t)$

Which simplifies to:
- $+\alpha \cdot G \cdot (1 − \pi(a))$ for the taken action
- $+\alpha \cdot G \cdot (−\pi(a'))$ for all other actions

Trajectories are rolled out completely before updating (Monte Carlo return `G`).

---

## Results

Both algorithms were trained for **500,000 iterations** on the same environment.

| Metric | Q-Learning | REINFORCE |
|--------|-----------|-----------|
| Training time | ~22 min | ~56 min |
| Convergence speed | Fast (~100k iterations) | Slow (~300k+ iterations) |
| Final policy stability | High (near-deterministic) | Moderate (high variance) |
| Scalability | Limited (tabular) | Better (parametric) |

**Key observations:**

- **Q-Learning** converges significantly faster in this discrete environment. The off-policy nature of Sarsamax allows efficient reuse of past transitions, and the tabular Q-table provides an exact representation for a 10×10 grid.

- **REINFORCE** suffers from high gradient variance inherent to Monte Carlo estimation. Without a baseline or variance reduction technique, convergence requires many more episodes. However, its parametric nature makes it more naturally extensible to continuous or large state spaces.

- These limitations motivated the transition to **PPO (Proximal Policy Optimization)** and **Deep RL** with neural network function approximators in the next phase of the project.

---

## Project Structure

```
.
├── main.py             # Entry point - choose algorithm and launch training + visualization
├── config.py           # Hyperparameters (grid size, learning rate, discount, iterations...)
├── env.py              # MDP logic: state transitions, reward function, terminal conditions
├── board.py            # Board generation and state management
├── q_learning.py       # Q-Learning (Sarsamax) implementation
├── policy_gradient.py  # REINFORCE implementation
├── visualize.py        # PyGame visualization of training and learned policy
└── requirements.txt
```

---

## Usage

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run (default: Q-Learning)
python main.py
```

To switch to REINFORCE, open `main.py` and uncomment:
```python
visualize(learn_reinforce, policy)
```

Hyperparameters can be tuned in `config.py`:
```python
ROWS, COLS = 10, 10       # Grid size
ITERATIONS = 500_000      # Training iterations
LEARNING_RATE = 0.0005
DISCOUNT = 0.99
POISON_PROB = 0.1
```

---

## Context & Next Steps

This repository is a preliminary implementation built as part of a **Projet Scientifique Collectif (PSC)** at École polytechnique, in partnership with **Orange S.A.**, on the topic of *Reinforcement Learning for Automated Cyber Defense of Telecom Infrastructures*.

The goal of this phase was to gain a deep operational understanding of RL fundamentals before scaling to complex environments. The project is now moving toward:

- **Deep RL with PPO** via Stable Baselines3 for continuous/large state spaces
- **Multi-agent scenarios** (Red/Blue/Green agents) on the [PrimAITE](https://github.com/dstl/PrimAITE) simulator
- **Graph Neural Networks (GNN)** for representing dynamic network topologies
- **MARL** (Multi-Agent Reinforcement Learning) for cooperative defense strategies

---

## Dependencies

```
numpy
pygame
```
