# # traffic_rl/main.py
# import numpy as np
# import torch
# from env import TrafficEnv
# from agent import DQNAgent

# def train_agent(episodes=500):
#     print("Starting Training...")
#     # Initialize without rendering for speed
#     env = TrafficEnv()
    
#     # State space is 12 dimensions (6 per intersection). Action space is 4.
#     obs_dim = 12 
#     n_actions = 4
#     agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions)

#     best_reward = -float('inf')

#     for ep in range(episodes):
#         state, _ = env.reset()
#         total_reward = 0
#         done = False

#         while not done:
#             action = agent.act(state)
#             next_state, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated

#             agent.store(state, action, reward, next_state, done)
#             agent.learn()

#             state = next_state
#             total_reward += reward

#         agent.decay_epsilon()

#         if ep % 10 == 0:
#             print(f"Episode {ep:03d} | Epsilon: {agent.epsilon:.2f} | Reward: {total_reward:.2f}")
            
#         # Save best model
#         if total_reward > best_reward:
#             best_reward = total_reward
#             agent.save("best_traffic_agent.pth")

#     return agent

# def evaluate_agent(agent_path="best_traffic_agent.pth"):
#     print("Starting Evaluation (Visual Mode)...")
#     # Initialize WITH rendering
#     env = TrafficEnv(render_mode="human")
#     agent = DQNAgent(obs_dim=12, n_actions=4)
#     agent.load(agent_path)
#     agent.epsilon = 0.0 # Pure exploitation

#     state, _ = env.reset()
#     done = False
    
#     while not done:
#         action = agent.act(state)
#         state, _, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated

#     env.close()

# if __name__ == "__main__":
#     # 1. Train the model
#     #train_agent(episodes=300)
    
#     # 2. Watch the result
#     evaluate_agent("best_traffic_agent.pth")


# traffic_rl/main.py
# import numpy as np
# import torch
# from env import TrafficEnv
# from agent import DQNAgent

# def train_agent(episodes=500):
#     print("Starting Training...")
#     # Initialize without rendering for speed
#     env = TrafficEnv()
    
#     # DYNAMICALLY fetch the dimensions. 
#     # Our new env has an obs_dim of 22 because of the corridors.
#     obs_dim = env.observation_space.shape[0] 
#     n_actions = int(env.action_space.n)
    
#     agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions)

#     best_reward = -float('inf')

#     for ep in range(episodes):
#         state, _ = env.reset()
#         total_reward = 0
#         done = False

#         while not done:
#             action = agent.act(state)
#             next_state, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated

#             agent.store(state, action, reward, next_state, done)
#             agent.learn()

#             state = next_state
#             total_reward += reward

#         agent.decay_epsilon()

#         if ep % 10 == 0:
#             print(f"Episode {ep:03d} | Epsilon: {agent.epsilon:.2f} | Reward: {total_reward:.2f}")
            
#         # Save best model
#         if total_reward > best_reward:
#             best_reward = total_reward
#             agent.save("best_traffic_agent.pth")

#     return agent

# def evaluate_agent(agent_path="best_traffic_agent.pth"):
#     print("Starting Evaluation (Visual Mode)...")
#     # Initialize WITH rendering
#     env = TrafficEnv(render_mode="human")
    
#     # DYNAMICALLY fetch dimensions here too!
#     obs_dim = env.observation_space.shape[0] 
#     n_actions = int(env.action_space.n)
    
#     agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions)
#     agent.load(agent_path)
#     agent.epsilon = 0.0 # Pure exploitation

#     state, _ = env.reset()
#     done = False
    
#     while not done:
#         action = agent.act(state)
#         state, _, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated

#     env.close()

# if __name__ == "__main__":
#     # 1. Train the model (Uncomment this! You MUST train again on the new physics)
#     train_agent(episodes=500)
    
#     # 2. Watch the result
#     #evaluate_agent("best_traffic_agent.pth")


###### above used to train on env.py, below trains directly on sumo

import os
import numpy as np
import torch
from sumo_env import SumoTrafficEnv
from agent import DQNAgent

def train_agent(episodes=1000):
    print("Starting SUMO Training...")

    # Detect GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    env = SumoTrafficEnv(gui=False)   # no GUI during training — much faster

    obs_dim   = env.observation_space.shape[0]   # 30
    n_actions = int(env.action_space.n)           # 4

    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, device=device)

    # Load existing checkpoint if available so you can resume
    weights_path = "best_traffic_agent.pth"
    if os.path.exists(weights_path):
        agent.load(weights_path)
        print(f"Resuming from {weights_path}")

    best_reward = -float('inf')

    for ep in range(episodes):
        print(f"Episode {ep + 1}/{episodes}", end="\r", flush=True)
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        if ep % 10 == 0:
            print(f"Episode {ep:03d} | Epsilon: {agent.epsilon:.2f} | Reward: {total_reward:.2f}")

        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(weights_path)
            print(f"  >> New best: {best_reward:.2f}")

    env.close()
    print("Training complete.")


def evaluate_agent(weights_path="best_traffic_agent.pth"):
    print("Starting Evaluation...")

    # Detect GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    env = SumoTrafficEnv(gui=True)   # GUI on for watching

    obs_dim   = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, device=device)
    agent.load(weights_path)
    agent.epsilon = 0.0   # pure exploitation

    state, _ = env.reset()
    done = False

    while not done:
        action = agent.act(state)
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()
    print("Evaluation done.")


if __name__ == "__main__":
    train_agent(episodes=169)
    # After training finishes, uncomment below to watch:
    # evaluate_agent()