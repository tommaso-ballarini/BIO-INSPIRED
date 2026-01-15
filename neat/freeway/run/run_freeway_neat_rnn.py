import os
import sys
import neat
import numpy as np
import pickle
import datetime
import multiprocessing
import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import random
from pathlib import Path

# Environment Registration
try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass

# --- CONFIGURATION ---
ENV_ID = "ALE/Freeway-v5"
CONFIG_FILE_NAME = "neat_freeway_config.txt" 
NUM_GENERATIONS = 50 
TRAINING_SEED_MIN = 100
TRAINING_SEED_MAX = 1000000
MAX_STEPS = 1500
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# Path Setup
SCRIPT_DIR = Path(__file__).resolve().parent
# IMPORTANT: Adjust parent levels based on your exact folder structure
FREEWAY_ROOT = SCRIPT_DIR.parent 
OUTPUT_DIR = FREEWAY_ROOT / "results" / "neat_freeway_rnn"
CONFIG_PATH = FREEWAY_ROOT / "config" / CONFIG_FILE_NAME

os.makedirs(OUTPUT_DIR, exist_ok=True)

def eval_genome(genome, config):
    """
    Evaluation using Recurrent Neural Network (RNN) and Speed Wrapper.
    Fitness is the raw Atari score.
    """
    import sys
    from pathlib import Path
    
    # Path fix for workers to find the wrapper
    current_file = Path(__file__).resolve()
    freeway_root = current_file.parent.parent
    if str(freeway_root) not in sys.path:
        sys.path.insert(0, str(freeway_root))

    from wrapper.freeway_wrapper import FreewaySpeedWrapper
    import gymnasium as gym

    raw_env = gym.make("ALE/Freeway-v5", obs_type="ram", render_mode=None)
    env = FreewaySpeedWrapper(raw_env, normalize=True, mirror_last_5=True)
    
    seed = random.randint(TRAINING_SEED_MIN, TRAINING_SEED_MAX)
    obs, info = env.reset(seed=seed)
    
    # IMPORTANT: Use RecurrentNetwork instead of FeedForward
    net = neat.nn.RecurrentNetwork.create(genome, config)
    
    total_reward = 0.0
    steps = 0
    done = False
    
    while not done and steps < MAX_STEPS:
        # RNN activation maintains internal state between steps
        output = net.activate(obs)
        action = np.argmax(output) 
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += float(reward)
        steps += 1
        done = terminated or truncated
        
    env.close()
    return total_reward

def plot_results(stats, save_dir):
    print(f"Generating plots in: {save_dir}")
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if stats.most_fit_genomes:
        gen = range(len(stats.most_fit_genomes))
        plt.figure(figsize=(10, 6))
        plt.plot(gen, stats.get_fitness_mean(), 'b-', label="Average Fitness")
        plt.plot(gen, [c.fitness for c in stats.most_fit_genomes], 'r-', label="Best Fitness")
        plt.title("Freeway RNN Evolution (Raw Score)")
        plt.xlabel("Generations")
        plt.ylabel("Atari Points")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.savefig(save_dir / f"rnn_fitness_{ts}.png")
        plt.close()

def run_experiment():
    print("STARTING FREEWAY RNN EXPERIMENT")
    
    if not CONFIG_PATH.exists():
        print(f"Error: Config file not found at {CONFIG_PATH}")
        return

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(CONFIG_PATH))
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    pe = neat.ParallelEvaluator(NUM_WORKERS, eval_genome)
    
    try:
        winner = p.run(pe.evaluate, NUM_GENERATIONS)
        
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(OUTPUT_DIR / f"winner_rnn_{ts}.pkl", "wb") as f:
            pickle.dump(winner, f)
        with open(OUTPUT_DIR / f"stats_rnn_{ts}.pkl", "wb") as f:
            pickle.dump(stats, f)
            
        print(f"\nEvolution complete. Best Score: {winner.fitness}")
        plot_results(stats, OUTPUT_DIR)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    run_experiment()