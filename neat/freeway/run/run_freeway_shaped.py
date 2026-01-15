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
CONFIG_FILE_NAME = "neat_freeway_config.txt" # Must have num_inputs = 22
NUM_GENERATIONS = 50
TRAINING_SEED_MIN = 100
TRAINING_SEED_MAX = 1000000
MAX_STEPS = 1500
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# Path Setup (neat/freeway/run/...)
SCRIPT_DIR = Path(__file__).resolve().parent
FREEWAY_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = FREEWAY_ROOT / "results" / "neat_freeway_shaped"
CONFIG_PATH = FREEWAY_ROOT / "config" / CONFIG_FILE_NAME

os.makedirs(OUTPUT_DIR, exist_ok=True)

def eval_genome(genome, config):
    """
    Shaped Fitness Evaluation: Rewards crossing AND vertical progress.
    """
    import sys
    from pathlib import Path
    # Ensure worker finds the wrapper (2 levels up from run folder)
    freeway_root = Path(__file__).resolve().parent.parent
    if str(freeway_root) not in sys.path:
        sys.path.insert(0, str(freeway_root))

    from wrapper.freeway_wrapper import FreewaySpeedWrapper
    import gymnasium as gym

    raw_env = gym.make("ALE/Freeway-v5", obs_type="ram", render_mode=None)
    env = FreewaySpeedWrapper(raw_env, normalize=True, mirror_last_5=True)
    
    seed = random.randint(TRAINING_SEED_MIN, TRAINING_SEED_MAX)
    obs, info = env.reset(seed=seed)
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    total_reward = 0.0
    max_y = 0.0 # Track vertical progress
    steps = 0
    done = False
    
    while not done and steps < MAX_STEPS:
        output = net.activate(obs)
        action = np.argmax(output) 
        
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # --- SHAPED FITNESS LOGIC ---
        # reward = native crossing points (usually 1.0 per crossing)
        # obs[0] often represents the normalized Y position in many Freeway wrappers
        current_y = obs[0] 
        if current_y > max_y:
            # Reward incremental vertical progress to encourage moving UP
            total_reward += (current_y - max_y) * 0.1 
            max_y = current_y
            
        total_reward += float(reward) # Add crossing points
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
        plt.plot(gen, stats.get_fitness_mean(), 'b-', label="Avg Fitness")
        plt.plot(gen, [c.fitness for c in stats.most_fit_genomes], 'r-', label="Best Fitness")
        plt.title("Freeway Shaped Fitness Evolution")
        plt.xlabel("Generations")
        plt.ylabel("Fitness (Shaped)")
        plt.legend()
        plt.savefig(save_dir / f"shaped_fitness_{ts}.png")
        plt.close()

def run_experiment():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, str(CONFIG_PATH))
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    pe = neat.ParallelEvaluator(NUM_WORKERS, eval_genome)
    
    try:
        winner = p.run(pe.evaluate, NUM_GENERATIONS)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with open(OUTPUT_DIR / f"winner_shaped_{ts}.pkl", "wb") as f:
            pickle.dump(winner, f)
        
        plot_results(stats, OUTPUT_DIR)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    try: multiprocessing.set_start_method('spawn')
    except RuntimeError: pass
    run_experiment()