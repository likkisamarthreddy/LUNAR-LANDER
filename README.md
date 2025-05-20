# 🚀 LunarLander with PSO (Particle Swarm Optimization)

A smart controller for LunarLander environment using Particle Swarm Optimization (PSO). The agent learns to land the lunar module safely by evolving the best actions using swarm intelligence.

## 🎯 Features

- Solves LunarLander-v2 using PSO instead of traditional RL
- No neural networks or backpropagation — just evolutionary optimization
- Lightweight and easy to understand
- Fun demo of swarm intelligence in action

## 🧠 How It Works

1. A population (swarm) of candidate solutions (particles) is initialized.
2. Each particle represents a set of weights for a simple policy.
3. The policy is tested in the LunarLander environment.
4. PSO updates particle positions based on personal and global best scores.
5. The process repeats to find a high-performing landing strategy.

## 🔧 Requirements

- Python 3.7+
- NumPy
- gym[box2d]

Install dependencies:
```bash
pip install numpy gym[box2d]


📄 License
This project is licensed under the MIT License.
