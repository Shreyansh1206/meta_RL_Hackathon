from sumo_env import SumoTrafficEnv
from agent import DQNAgent
import os

def main():
    # 1. Initialize the Environment
    env = SumoTrafficEnv(gui=True)
    
    # 2. Setup Agent to match your 22D Pygame Weights
    agent = DQNAgent(obs_dim=30, n_actions=4)
    
    # 3. Load your trained weights
    weights_path = "best_traffic_agent.pth"
    if os.path.exists(weights_path):
        agent.load(weights_path)
        print(f"Successfully loaded weights: {weights_path}")
    else:
        print(f"ERROR: Could not find {weights_path}")
        return

    agent.epsilon = 0.0 # Pure exploitation mode
    
    # 4. Run the simulation
    state, _ = env.reset()
    try:
        while True:
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        env.close()
        print("Simulation Closed.")

if __name__ == "__main__":
    main()