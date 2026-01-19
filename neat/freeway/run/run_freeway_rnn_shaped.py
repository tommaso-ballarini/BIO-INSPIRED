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
FREEWAY_ROOT = SCRIPT_DIR.parent 
OUTPUT_DIR = FREEWAY_ROOT / "results" / "neat_freeway_rnn_shaped"
CONFIG_PATH = FREEWAY_ROOT / "config" / CONFIG_FILE_NAME

os.makedirs(OUTPUT_DIR, exist_ok=True)

def eval_genome(genome, config):
    """
    RNN Evaluation with Shaped Fitness:
    - Rewards: Crossings (High), Vertical Progress (Medium)
    - Penalties: Collisions (Small), Time/Steps (Very Small)
    """
    import sys
    from pathlib import Path
    
    # Path fix for workers to find the wrapper (2 levels up)
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
    
    # Use RecurrentNetwork for RNN architectures
    net = neat.nn.RecurrentNetwork.create(genome, config)
    
    total_reward = 0.0
    max_y = 0.0
    steps = 0
    collisions = 0
    prev_y = obs[0]
    done = False
    
    while not done and steps < MAX_STEPS:
        output = net.activate(obs)
        action = np.argmax(output) 
        
        obs, native_reward, terminated, truncated, info = env.step(action)
        
        current_y = obs[0]
        
        # 1. Reward Native Crossings (multiplied for importance)
        total_reward += float(native_reward) * 100.0
        
        # 2. Reward Vertical Progress (Max height reached)
        if current_y > max_y:
            total_reward += (current_y - max_y) * 20.0
            max_y = current_y
            
        # 3. Penalty for Collisions (Heuristic: Y decreases suddenly)
        if current_y < prev_y - 0.05:
            total_reward -= 2.0
            collisions += 1
            
        # 4. Time Penalty (Step-based)
        total_reward -= 0.01
        
        prev_y = current_y
        steps += 1
        done = terminated or truncated
        
    env.close()
    return max(0.0, total_reward)

def plot_results(stats, save_dir):
    print(f"Generating plots in: {save_dir}")
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if stats.most_fit_genomes:
        gen = range(len(stats.most_fit_genomes))
        plt.figure(figsize=(10, 6))
        plt.plot(gen, stats.get_fitness_mean(), 'b-', label="Average Fitness")
        plt.plot(gen, [c.fitness for c in stats.most_fit_genomes], 'r-', label="Best Fitness")
        plt.title("RNN Shaped + Time Penalty Evolution")
        plt.xlabel("Generations")
        plt.ylabel("Fitness Value")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.savefig(save_dir / f"rnn_shaped_fitness_{ts}.png")
        plt.close()

def run_experiment():
    print("STARTING FREEWAY RNN SHAPED EXPERIMENT")
    
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
        with open(OUTPUT_DIR / f"winner_rnn_shaped_{ts}.pkl", "wb") as f:
            pickle.dump(winner, f)
            
        plot_results(stats, OUTPUT_DIR)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    run_experiment()