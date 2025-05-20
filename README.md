🚀 Lunar Lander AI Agent
This project implements an AI agent to solve the LunarLander-v3 environment from OpenAI Gym using Particle Swarm Optimization (PSO). It also includes a human-playable interface using Pygame and a modular evaluation system.

📁 Project Structure
graphql
Copy
Edit
.
├── train_agent.py       # Trains an agent using PSO and saves the best policy
├── evaluate_agent.py    # Evaluates a saved policy over multiple episodes
├── my_policy.py         # Defines the policy_action function (linear policy)
├── play_lunar_lander.py # Allows manual play of Lunar Lander using keyboard
└── best_policy.npy      # [Optional] Pretrained policy file (generated after training)
🧠 Algorithm
The agent uses a linear policy (observation → logits → argmax) and is optimized via Particle Swarm Optimization (PSO). PSO parameters can be customized for experimentation.

🛠️ Installation
Install dependencies:

bash
Copy
Edit
pip install gymnasium[box2d] pygame numpy

🏋️‍♂️ Training the Agent
To train the agent using PSO:

bash
Copy
Edit
python train_agent.py --train --num_particles 100 --num_iterations 500 --filename best_policy.npy
You can also resume training from a saved session using:

bash
Copy
Edit
python train_agent.py --train --load --filename best_policy.npy
🎮 Playing with the Trained Agent
Evaluate the trained agent visually over a few episodes:

bash
Copy
Edit
python train_agent.py --play --filename best_policy.npy
Or use the dedicated evaluation script (renders first 5 episodes):

bash
Copy
Edit
python evaluate_agent.py --filename best_policy.npy --policy_module my_policy
👤 Human Gameplay
You can manually play Lunar Lander using the keyboard:

bash
Copy
Edit
python play_lunar_lander.py
Controls:
W: Fire main engine

A: Fire left engine

D: Fire right engine

S: Do nothing

Q: Quit game after episode

📊 Policy Representation
The policy is a simple linear mapping:

python
Copy
Edit
logits = np.dot(observation, W) + b
action = np.argmax(logits)
Where:

observation: 8-dimensional input from the environment.

W: Weights matrix (8×4).

b: Bias vector (4,).

📈 Optimization Details
Algorithm: Particle Swarm Optimization (PSO)

Features:

Velocity and position updates

Mutation for diversity

Partial random restarts

Multi-process evaluation

🧪 Example Results
After training for 500–1000 iterations with 100 particles, the agent can typically achieve average rewards above 200, which is considered solved in LunarLander-v3.

📄 License
This project is licensed under the MIT License. See LICENSE for details.

🙌 Acknowledgments
OpenAI Gym

Pygame

NumPy
