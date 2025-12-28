import os
import sys
import neat
import pickle
import numpy as np
import gymnasium as gym
from glob import glob

# --- PATH SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..')) 

if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    print("Error: ale-py not installed.")
    sys.exit(1)

ENV_ID = "ALE/Skiing-v5"
RESULTS_DIR = os.path.join(project_root, "evolution_results", "baseline_run")

def load_latest_winner():
    """Finds the most recent pickle file in baseline_run."""
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Directory not found {RESULTS_DIR}")
        sys.exit(1)

    all_files = glob(os.path.join(RESULTS_DIR, "*.pkl"))
    
    if not all_files:
        print(f"No .pkl files found in {RESULTS_DIR}")
        sys.exit(1)
    
    latest_file = max(all_files, key=os.path.getctime)
    print(f"Loading genome from: {os.path.basename(latest_file)}")
    
    with open(latest_file, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, tuple):
        return data[0]
    else:
        return data

def visualize():
    genome = load_latest_winner()
    
    config_path = os.path.join(project_root, "config", "config_baseline.txt")
    
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)
        
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    env = gym.make(ENV_ID, obs_type="ram", render_mode="human")
    
    input_size = config.genome_config.num_inputs
    
    if input_size != 128:
        print(f"Warning: Config specifies {input_size} inputs. Baseline requires 128.")

    observation, info = env.reset(seed=14)
    
    # Baseline uses FeedForward
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    done = False
    total_reward = 0.0
    
    print("--- STARTING REPLAY ---")
    
    try:
        while not done:
            # Manual normalization for baseline
            inputs = observation / 255.0
            
            output = net.activate(inputs)
            action = np.argmax(output)
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
    except KeyboardInterrupt:
        print("Interrupted by user.")
        
    print(f"Game Over. Total Fitness: {total_reward}")
    env.close()

if __name__ == "__main__":
    visualize()