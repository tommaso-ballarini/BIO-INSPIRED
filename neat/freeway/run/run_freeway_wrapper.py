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

# Register Atari environments
try:
    gym.register_envs(ale_py)
except Exception:
    pass

# --- EXPERIMENT CONFIGURATION ---
ENV_ID = "ALE/Freeway-v5"
CONFIG_FILE_NAME = "neat_freeway_config.txt" # Must have num_inputs = 22
NUM_GENERATIONS = 50
TRAINING_SEED_MIN = 100      # Seeds reserved for testing are < 100
TRAINING_SEED_MAX = 1000000  
MAX_STEPS = 1500
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "neat_freeway_wrapper"
CONFIG_PATH = PROJECT_ROOT / "config" / CONFIG_FILE_NAME

os.makedirs(OUTPUT_DIR, exist_ok=True)

def eval_genome(genome, config):
    """
    Evaluation using the FreewaySpeedWrapper (22 features).
    Updated for: neat/freeway/wrapper/
    """
    # --- PATH SETUP FOR WORKERS ---
    # Current file: PROJECT_ROOT/neat/freeway/run/run_freeway_wrapper.py
    # Level 1 up: PROJECT_ROOT/neat/freeway/run/
    # Level 2 up: PROJECT_ROOT/neat/freeway/ (This contains the 'wrapper' folder)
    current_file = Path(__file__).resolve()
    freeway_root = current_file.parent.parent
    
    # Add freeway_root to sys.path so 'import wrapper' works
    if str(freeway_root) not in sys.path:
        sys.path.insert(0, str(freeway_root))

    # --- WRAPPER IMPORT ---
    try:
        from wrapper.freeway_wrapper import FreewaySpeedWrapper
    except ImportError as e:
        print(f"Worker Error: Module 'wrapper' not found in {freeway_root}")
        # Print path for debugging if it fails again
        print(f"Current sys.path: {sys.path}")
        return 0.0

    # --- ENVIRONMENT SETUP ---
    try:
        # We use the native ALE Freeway environment
        raw_env = gym.make("ALE/Freeway-v5", obs_type="ram", render_mode=None)
        # Apply the wrapper from the local folder
        env = FreewaySpeedWrapper(raw_env, normalize=True, mirror_last_5=True)
    except Exception as e:
        print(f"Worker Error: Env setup failed: {e}")
        return 0.0
    
    # --- EVALUATION ---
    # Random seed for training (reserving 0-99 for test)
    current_seed = random.randint(100, 1000000)
    observation, info = env.reset(seed=current_seed)
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    total_reward = 0.0
    steps = 0
    max_steps_limit = 1500 
    done = False
    
    while not done and steps < max_steps_limit:
        # Activate network with the 22 features provided by the wrapper
        output = net.activate(observation)
        
        # Action map: 0: NOOP, 1: UP, 2: DOWN
        action = np.argmax(output) 
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += float(reward)
        steps += 1
        done = terminated or truncated
        
    env.close()
    return total_reward

def plot_results(stats, save_dir):
    """
    Plots Fitness history and Speciation.
    """
    print(f"Generating plots in: {save_dir}")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # --- 1. FITNESS PLOT ---
    if stats.most_fit_genomes:
        generation = range(len(stats.most_fit_genomes))
        best_fitness = [c.fitness for c in stats.most_fit_genomes]
        avg_fitness = np.array(stats.get_fitness_mean())
        stdev_fitness = np.array(stats.get_fitness_stdev())

        plt.figure(figsize=(10, 6))
        plt.plot(generation, avg_fitness, 'b-', label="Average Fitness")
        plt.fill_between(generation, avg_fitness - stdev_fitness, avg_fitness + stdev_fitness, color='blue', alpha=0.1)
        plt.plot(generation, best_fitness, 'r-', label="Best Fitness")
        
        plt.title(f"Freeway Wrapper Evolution\n(Seeds >= {TRAINING_SEED_MIN})")
        plt.xlabel("Generations")
        plt.ylabel("Fitness (Raw Score)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        plt.savefig(save_dir / f"wrapper_fitness_{timestamp}.png")
        plt.close()

    # --- 2. SPECIATION PLOT ---
    try:
        species_sizes = stats.get_species_sizes()
        if species_sizes:
            plt.figure(figsize=(10, 6))
            plt.stackplot(range(len(species_sizes)), np.array(species_sizes).T)
            plt.title("Speciation History")
            plt.xlabel("Generations")
            plt.ylabel("Population Size")
            plt.savefig(save_dir / f"wrapper_speciation_{timestamp}.png")
            plt.close()
    except Exception as e:
        print(f"Error plotting speciation: {e}")

def run_experiment():
    print("STARTING FREEWAY WRAPPER EXPERIMENT")
    
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
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save Winner and Stats
        with open(OUTPUT_DIR / f"winner_freeway_wrapper_{timestamp}.pkl", "wb") as f:
            pickle.dump(winner, f)
        with open(OUTPUT_DIR / f"stats_freeway_wrapper_{timestamp}.pkl", "wb") as f:
            pickle.dump(stats, f)
            
        print(f"\nEvolution completed. Best Fitness: {winner.fitness}")
        plot_results(stats, OUTPUT_DIR)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    run_experiment()
