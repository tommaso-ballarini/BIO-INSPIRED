import sys
import os
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
CONFIG_FILE_NAME = "neat_freeway_config.txt"
NUM_GENERATIONS = 30
TRAINING_SEED_MIN = 100      # Seeds >= 100 reserved for training
TRAINING_SEED_MAX = 1000000  
MAX_STEPS = 1500             # Standard Freeway round limit
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# Path configuration
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "neat_freeway_baseline"
CONFIG_PATH = PROJECT_ROOT / "config" / CONFIG_FILE_NAME

os.makedirs(OUTPUT_DIR, exist_ok=True)

def eval_genome(genome, config):
    """
    Evaluates a genome using raw RAM observations.
    The seed is randomly sampled from the training range (> 100).
    """
    # obs_type="ram" provides the 128 Atari RAM bytes
    env = gym.make(ENV_ID, obs_type="ram", render_mode=None)
    
    # Freeway action map: 0: NOOP, 1: UP, 2: DOWN
    current_seed = random.randint(TRAINING_SEED_MIN, TRAINING_SEED_MAX)
    observation, info = env.reset(seed=current_seed)
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    total_reward = 0.0
    steps = 0
    done = False
    
    while not done and steps < MAX_STEPS:
        # Normalize RAM bytes (0-255) to (0.0-1.0)
        inputs = observation.astype(np.float32) / 255.0
        
        output = net.activate(inputs)
        action = np.argmax(output) 
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        done = terminated or truncated
        
    env.close()
    # Freeway fitness uses the raw crossing score
    return float(total_reward)

def plot_results(stats, save_dir):
    """
    Generates fitness and speciation plots.
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
        plt.plot(generation, best_fitness, 'r-', label="Best Fitness (Points)")
        
        plt.title(f"Freeway Baseline - Fitness Evolution\n(Training Seeds >= {TRAINING_SEED_MIN})")
        plt.xlabel("Generations")
        plt.ylabel("Atari Score")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        fit_path = save_dir / f"fitness_{timestamp}.png"
        plt.savefig(fit_path)
        plt.close()

    # --- 2. SPECIATION PLOT ---
    try:
        species_sizes = stats.get_species_sizes()
        if species_sizes:
            plt.figure(figsize=(10, 6))
            plt.stackplot(range(len(species_sizes)), np.array(species_sizes).T)
            plt.title("Speciation Evolution")
            plt.xlabel("Generations")
            plt.ylabel("Population Size")
            spec_path = save_dir / f"speciation_{timestamp}.png"
            plt.savefig(spec_path)
            plt.close()
    except Exception as e:
        print(f"Error plotting speciation: {e}")

def run_experiment():
    print("=== STARTING NEAT FREEWAY EXPERIMENT (BASELINE) ===")
    print(f"Config: {CONFIG_PATH}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Seed Range Training: {TRAINING_SEED_MIN} - {TRAINING_SEED_MAX}")
    
    if not CONFIG_PATH.exists():
        print(f"ERROR: Config file not found at {CONFIG_PATH}")
        return

    # Load config
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
        winner_path = OUTPUT_DIR / f"winner_freeway_{timestamp}.pkl"
        stats_path = OUTPUT_DIR / f"stats_freeway_{timestamp}.pkl"
        
        with open(winner_path, "wb") as f:
            pickle.dump(winner, f)
        with open(stats_path, "wb") as f:
            pickle.dump(stats, f)
            
        print("\nEvolution completed.")
        print(f"Best Fitness: {winner.fitness}")
        print(f"Winner saved to: {winner_path}")

        plot_results(stats, OUTPUT_DIR)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Multiprocessing start method for macOS/Windows
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
        
    run_experiment()
