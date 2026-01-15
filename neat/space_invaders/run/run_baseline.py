import sys
import os
import pickle
import multiprocessing
import numpy as np
import neat
import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py 
from pathlib import Path

# --- PATH CONFIGURATION ---
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

CONFIG_PATH = project_root / 'config' / 'config_baseline.txt'
RESULTS_DIR = project_root / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- GLOBAL SETTINGS ---
GAME_NAME = "SpaceInvadersNoFrameskip-v4"
GENERATIONS = 30
FIXED_SEED = 42

print(f"✅ Env Config: {GAME_NAME}")
print(f"🔒 Fixed Seed: {FIXED_SEED} (Determinism Enabled)")

# Ensure ALE environments are registered
gym.register_envs(ale_py)

def eval_genome(genome, config):
    """ Evaluates a single genome. """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Create env for this specific process
    try:
        env = gym.make(GAME_NAME, obs_type="ram", render_mode=None)
    except Exception as e:
        return 0.0

    # --- FIXED SEED FOR DETERMINISM ---
    observation, info = env.reset(seed=FIXED_SEED)

    # RAM check (Space Invaders RAM is 128 bytes)
    if len(observation) != 128:
        env.close()
        return 0.0

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    max_steps = 10000 

    while not (terminated or truncated) and steps < max_steps:
        # Normalize RAM (0-255 -> 0.0-1.0)
        inputs = observation / 255.0 
        if isinstance(inputs, np.ndarray):
            inputs = inputs.flatten()

        # Network activation
        outputs = net.activate(inputs)
        action = np.argmax(outputs)
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    env.close()
    return total_reward

def plot_stats(statistics):
    """ Plots Average and Best Fitness history. """
    print("📊 Generating Fitness Plot...")
    if not statistics.most_fit_genomes:
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())

    plt.figure(figsize=(10, 6))
    plt.plot(generation, best_fitness, 'r-', label="Best Fitness")
    plt.plot(generation, avg_fitness, 'b-', label="Avg Fitness")
    plt.title(f"Baseline Training - Raw RAM (Seed {FIXED_SEED})")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend()
    try:
        output_path = RESULTS_DIR / "fitness_baseline.png"
        plt.savefig(output_path)
        print(f"✅ Fitness plot saved: {output_path.name}")
    except Exception as e:
        print(f"❌ Fitness plot error: {e}")
    plt.close()

def plot_species(statistics):
    """ Generates Speciation Stackplot. """
    print("📊 Generating Speciation Plot...")
    
    # Get species sizes per generation
    species_sizes = statistics.get_species_sizes()
    
    if not species_sizes:
        print("⚠️ No speciation data found.")
        return

    num_generations = len(species_sizes)
    
    # Transpose for stackplot (Rows=Species, Cols=Generations)
    curves = np.array(species_sizes).T

    plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    
    try:
        ax.stackplot(range(num_generations), *curves)
        
        plt.title("Evolution of Species (Speciation)")
        plt.ylabel("Number of Genomes per Species")
        plt.xlabel("Generations")
        plt.margins(0, 0)
        
        output_path = RESULTS_DIR / "speciation.png"
        plt.savefig(output_path)
        print(f"✅ Speciation plot saved: {output_path.name}")
        
    except Exception as e:
        print(f"❌ Plotting error: {e}")
    
    plt.close()

def run_baseline():
    print(f"📂 Loading Config: {CONFIG_PATH}")
    if not CONFIG_PATH.exists():
        print(f"❌ Config not found: {CONFIG_PATH}")
        return

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(CONFIG_PATH))

    p = neat.Population(config)
    
    # Reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # Checkpoints
    checkpoint_prefix = RESULTS_DIR / "neat-checkpoint-"
    p.add_reporter(neat.Checkpointer(10, filename_prefix=str(checkpoint_prefix)))

    # Multiprocessing Setup
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"🚀 Starting Baseline on {num_workers} workers...")
    
    pe = neat.ParallelEvaluator(num_workers, eval_genome)
    
    try:
        winner = p.run(pe.evaluate, GENERATIONS)
        
        print(f"\n🏆 Training Complete.")
        print(f"💎 Best Ever Fitness: {winner.fitness}")
        
        with open(RESULTS_DIR / 'baseline_winner.pkl', 'wb') as f:
            pickle.dump(winner, f)
        
        # Generate Plots
        plot_stats(stats)
        plot_species(stats)
        
        print("✅ Baseline finished successfully.")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    print(f"Gymnasium version: {gym.__version__}")
    run_baseline()