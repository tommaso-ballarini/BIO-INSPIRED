import os
import sys
import neat
import pickle
import numpy as np
import gymnasium as gym
import time
from pathlib import Path
from glob import glob

# --- PATH SETUP ---
current_file = Path(__file__).resolve()
freeway_root = current_file.parent.parent
if str(freeway_root) not in sys.path:
    sys.path.insert(0, str(freeway_root))

try:
    import ale_py
    gym.register_envs(ale_py)
except Exception: pass

# --- CONFIGURATION ---
RESULTS_DIR = freeway_root / "results" / "neat_freeway_shaped"
CONFIG_FILENAME = "neat_freeway_config.txt"
TEST_SEEDS = range(100)
ENV_ID = "ALE/Freeway-v5"

def load_latest_winner():
    all_files = glob(os.path.join(RESULTS_DIR, "winner_shaped_*.pkl"))
    if not all_files:
        print(f"Error: No winner files in {RESULTS_DIR}")
        sys.exit(1)
    latest_file = max(all_files, key=os.path.getctime)
    print(f"Loading winner: {os.path.basename(latest_file)}")
    with open(latest_file, 'rb') as f:
        data = pickle.load(f)
    return data[0] if isinstance(data, tuple) else data

def run_simulation(genome, config, seed, render=False):
    from wrapper.freeway_wrapper import FreewaySpeedWrapper
    
    render_mode = "human" if render else None
    env = gym.make(ENV_ID, obs_type="ram", render_mode=render_mode)
    env = FreewaySpeedWrapper(env, normalize=True, mirror_last_5=True)

    obs, info = env.reset(seed=seed)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    done = False
    total_score = 0.0 # Test strictly on native Atari points
    
    while not done:
        action = np.argmax(net.activate(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        total_score += float(reward)
        done = terminated or truncated
        if render: time.sleep(0.01)

    env.close()
    return total_score

def main():
    genome = load_latest_winner()
    config_path = str(freeway_root / "config" / CONFIG_FILENAME)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    print("\nSTARTING SHAPED BENCHMARK (SEEDS 0-99)")
    print("-" * 60)

    results = []
    for seed in TEST_SEEDS:
        score = run_simulation(genome, config, seed)
        results.append({'seed': seed, 'score': score})
        if (seed + 1) % 20 == 0: print(f"Tested {seed + 1}/100 seeds...")

    scores = [r['score'] for r in results]
    print("\n" + "="*60)
    print("FINAL TEST RESULTS (NATIVE SCORE)")
    print("="*60)
    print(f"Average: {np.mean(scores):.2f} (std: {np.std(scores):.2f})")
    print(f"Best:    {max(scores)} (Seed {max(results, key=lambda x:x['score'])['seed']})")
    print("="*60)

    if input("\nWatch the best game? [y/N]: ").strip().lower() == 'y':
        best_seed = max(results, key=lambda x: x['score'])['seed']
        run_simulation(genome, config, best_seed, render=True)

if __name__ == "__main__":
    main()