import os
import sys
import neat
import pickle
import numpy as np
import gymnasium as gym
import ale_py
from glob import glob
import time

# --- PATH SETUP ---
# Calculate paths relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

# Register Atari environments
try:
    gym.register_envs(ale_py)
except Exception:
    pass

# --- CONFIGURATION ---
# Path to the directory where your trained .pkl winners are saved
RESULTS_DIR = os.path.join(project_root, "results", "neat_freeway_baseline")
CONFIG_FILENAME = "neat_freeway_config.txt"
TEST_SEEDS = range(100)  # Standard test range: Seeds 0-99
ENV_ID = "ALE/Freeway-v5"

def load_latest_winner():
    """Locates and loads the most recently created winner genome file."""
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Directory {RESULTS_DIR} not found.")
        sys.exit(1)

    # CHANGE: Specifically look for files starting with "winner_" 
    # to avoid loading the "stats_" files by mistake.
    all_files = glob(os.path.join(RESULTS_DIR, "winner_*.pkl"))
    
    if not all_files:
        print(f"No winner .pkl files found in {RESULTS_DIR}")
        sys.exit(1)
    
    # Sort files by creation time to get the newest one
    latest_file = max(all_files, key=os.path.getctime)
    print(f"Loading winner genome: {os.path.basename(latest_file)}")
    
    with open(latest_file, 'rb') as f:
        data = pickle.load(f)
    
    return data[0] if isinstance(data, tuple) else data

def run_simulation(genome, config, seed, render=False):
    """Runs a full episode of Freeway and returns the total score."""
    
    render_mode = "human" if render else None
    
    # Initialize Freeway with RAM observations
    env = gym.make(ENV_ID, obs_type="ram", render_mode=render_mode)

    # Reset with specific seed for reproducibility
    observation, info = env.reset(seed=seed)
    
    # Reconstruct the Neural Network from the genome
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    done = False
    total_score = 0.0
    steps = 0
    max_steps = 1500 # Standard round limit for Freeway

    try:
        while not done and steps < max_steps:
            # Normalize RAM bytes (0-255) to (0.0-1.0)
            inputs = observation.astype(np.float32) / 255.0

            # Get action from network (0: NOOP, 1: UP, 2: DOWN)
            output = net.activate(inputs)
            action = np.argmax(output)
            
            # Execute step
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_score += float(reward)
            done = terminated or truncated
            steps += 1
            
            if render:
                time.sleep(0.01) # Slow down visualization slightly

    except Exception as e:
        print(f"Simulation Error (Seed {seed}): {e}")
    finally:
        env.close()

    return total_score

def main():
    # 1. Load trained model and config
    genome = load_latest_winner()
    config_path = os.path.join(project_root, "config", CONFIG_FILENAME)
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    print(f"\nSTARTING BENCHMARK (SEEDS 0-99)...")
    print("-" * 60)

    results = []
    
    # 2. Run testing loop
    for seed in TEST_SEEDS:
        score = run_simulation(genome, config, seed, render=False)
        results.append({
            'seed': seed,
            'score': score
        })
        # Optional: print progress every 10 seeds
        if (seed + 1) % 10 == 0:
            print(f"Processed {seed + 1}/100 seeds...")

    # 3. Data Analysis
    scores = [r['score'] for r in results]
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    best_run = max(results, key=lambda x: x['score'])
    worst_run = min(results, key=lambda x: x['score'])

    # 4. Final Performance Report
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"Average Score over {len(TEST_SEEDS)} seeds: {avg_score:.2f} (std: {std_score:.2f})")
    print("-" * 30)
    print(f"Best Performance:  Score {best_run['score']} (Seed {best_run['seed']})")
    print(f"Worst Performance: Score {worst_run['score']} (Seed {worst_run['seed']})")
    print("="*60)

    # 5. Visual Feedback Prompt
    choice = input(f"\nWatch the best performing game (Seed {best_run['seed']})? [y/N]: ").strip().lower()
    if choice == 'y':
        print(f"Replaying Seed {best_run['seed']} with rendering...")
        run_simulation(genome, config, best_run['seed'], render=True)
        print("Visualization complete.")
    else:
        print("Test script finished.")

if __name__ == "__main__":
    main()