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
except Exception:
    pass

# --- CONFIGURATION ---
RESULTS_DIR = freeway_root / "results" / "neat_freeway_rnn_shaped_timepenalty"
CONFIG_FILENAME = "neat_freeway_config.txt"
TEST_SEEDS = range(100)
ENV_ID = "ALE/Freeway-v5"

def load_latest_winner():
    all_files = glob(os.path.join(RESULTS_DIR, "winner_rnn_shaped_*.pkl"))
    if not all_files:
        print(f"Error: No RNN shaped winner files in {RESULTS_DIR}")
        sys.exit(1)
    
    latest_file = max(all_files, key=os.path.getctime)
    print(f"Loading latest RNN Shaped winner: {os.path.basename(latest_file)}")
    with open(latest_file, 'rb') as f:
        data = pickle.load(f)
    return data[0] if isinstance(data, tuple) else data

def run_simulation(genome, config, seed, render=False):
    from wrapper.freeway_wrapper import FreewaySpeedWrapper
    
    render_mode = "human" if render else None
    raw_env = gym.make(ENV_ID, obs_type="ram", render_mode=render_mode)
    env = FreewaySpeedWrapper(raw_env, normalize=True, mirror_last_5=True)

    observation, info = env.reset(seed=seed)
    net = neat.nn.RecurrentNetwork.create(genome, config)
    
    done = False
    total_native_score = 0.0
    
    while not done:
        output = net.activate(observation)
        action = np.argmax(output)
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_native_score += float(reward)
        done = terminated or truncated
        
        if render:
            time.sleep(0.01)

    env.close()
    return total_native_score

def main():
    genome = load_latest_winner()
    config_path = freeway_root / "config" / CONFIG_FILENAME
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(config_path))

    print("\nSTARTING RNN SHAPED BENCHMARK (SEEDS 0-99)")
    print("-" * 60)

    results = []
    for seed in TEST_SEEDS:
        score = run_simulation(genome, config, seed)
        results.append({'seed': seed, 'score': score})
        if (seed + 1) % 10 == 0:
            print(f"Processed {seed + 1}/100 seeds...")

    scores = [r['score'] for r in results]
    print("\n" + "="*60)
    print("FINAL RNN SHAPED TEST RESULTS")
    print("="*60)
    print(f"Average Score: {np.mean(scores):.2f} (std: {np.std(scores):.2f})")
    print(f"Best Score:    {max(scores)} (Seed {max(results, key=lambda x: x['score'])['seed']})")
    print("="*60)

    if input("\nWatch the best performing game? [y/N]: ").strip().lower() == 'y':
        best_seed = max(results, key=lambda x: x['score'])['seed']
        run_simulation(genome, config, best_seed, render=True)

if __name__ == "__main__":
    main()