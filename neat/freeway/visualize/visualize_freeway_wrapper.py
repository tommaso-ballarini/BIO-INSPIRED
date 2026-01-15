import os
import sys
import neat
import pickle
import numpy as np
import gymnasium as gym
import time
from pathlib import Path
from glob import glob

# --- ENVIRONMENT REGISTRATION ---
try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass

# --- PATH SETUP ---
current_file = Path(__file__).resolve()
freeway_root = current_file.parent.parent

if str(freeway_root) not in sys.path:
    sys.path.insert(0, str(freeway_root))

# --- CONFIGURATION ---
RESULTS_DIR = freeway_root / "results" / "neat_freeway_wrapper"
CONFIG_FILENAME = "neat_freeway_config.txt"
TEST_SEEDS = range(100) 
ENV_ID = "ALE/Freeway-v5"

def load_latest_winner():
    """Locates and loads the most recently created winner genome file."""
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Directory {RESULTS_DIR} not found.")
        sys.exit(1)

    all_files = glob(os.path.join(RESULTS_DIR, "winner_*.pkl"))
    
    if not all_files:
        print(f"No winner .pkl files found in {RESULTS_DIR}")
        sys.exit(1)
    
    latest_file = max(all_files, key=os.path.getctime)
    print(f"Loading winner genome: {os.path.basename(latest_file)}")
    
    with open(latest_file, 'rb') as f:
        data = pickle.load(f)
    
    return data[0] if isinstance(data, tuple) else data

def run_simulation(genome, config, seed, render=False):
    """Runs a full episode using the FreewaySpeedWrapper features."""
    from wrapper.freeway_wrapper import FreewaySpeedWrapper
    
    render_mode = "human" if render else None
    
    # Initialize environment
    raw_env = gym.make(ENV_ID, obs_type="ram", render_mode=render_mode)
    env = FreewaySpeedWrapper(raw_env, normalize=True, mirror_last_5=True)

    observation, info = env.reset(seed=seed)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    done = False
    total_score = 0.0
    steps = 0
    max_steps = 1500 

    try:
        while not done and steps < max_steps:
            output = net.activate(observation)
            action = np.argmax(output)
            
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_score += float(reward)
            done = terminated or truncated
            steps += 1
            
            if render:
                time.sleep(0.01) 

    except Exception as e:
        print(f"Simulation Error (Seed {seed}): {e}")
    finally:
        env.close()

    return total_score

def main():
    genome = load_latest_winner()
    config_path = freeway_root / "config" / CONFIG_FILENAME
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(config_path))

    print(f"\nSTARTING BENCHMARK (SEEDS 0-99)...")
    print("-" * 60)

    results = []
    
    for seed in TEST_SEEDS:
        score = run_simulation(genome, config, seed, render=False)
        results.append({'seed': seed, 'score': score})
        if (seed + 1) % 10 == 0:
            print(f"Processed {seed + 1}/100 seeds...")

    # --- ANALYSIS ---
    scores = [r['score'] for r in results]
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    
    # Find the best run (seed and score)
    best_run = max(results, key=lambda x: x['score'])

    # --- FINAL REPORT ---
    print("\n" + "="*60)
    print("FINAL TEST RESULTS (WRAPPER)")
    print("="*60)
    print(f"Average Score over {len(TEST_SEEDS)} seeds: {avg_score:.2f} (std: {std_score:.2f})")
    print("-" * 30)
    print(f"Best Run:       Score {best_run['score']} (Seed {best_run['seed']})")
    print("="*60)

    choice = input(f"\nWatch the best performing game (Seed {best_run['seed']})? [y/N]: ").strip().lower()
    if choice == 'y':
        print(f"Replaying Seed {best_run['seed']}...")
        run_simulation(genome, config, best_run['seed'], render=True)

if __name__ == "__main__":
    main()
