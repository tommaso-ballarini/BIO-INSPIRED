import sys
import os
import pickle
import multiprocessing
import numpy as np
import neat
import matplotlib.pyplot as plt
import gymnasium as gym
from ocatari.core import OCAtari
import random
import time
from pathlib import Path

# --- PATH CONFIGURATION ---
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- IMPORT WRAPPER ---
try:
    from wrapper.wrapper_si_ego import SpaceInvadersEgocentricWrapper
except ImportError:
    print("❌ ERROR: 'wrapper_si_ego.py' not found!")
    sys.exit(1)

# --- GLOBAL SETTINGS ---
CONFIG_PATH = project_root / 'config' / 'config_si_ego.txt'
RESULTS_DIR = project_root / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GAME_NAME = "ALE/SpaceInvaders-v5"
GENERATIONS = 50
TRAINING_SEED_MIN = 100
TRAINING_SEED_MAX = 100000
EPISODES_PER_GENOME = 3

print(f"✅ Config: {GAME_NAME} (Egocentric RNN Mode)")
print(f"🔄 Training: Average of {EPISODES_PER_GENOME} episodes (Seeds {TRAINING_SEED_MIN}+)")

# --- PLOTTING FUNCTIONS ---

def plot_stats(statistics):
    """ Plots Average and Best Fitness. """
    if not statistics.most_fit_genomes: return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())

    plt.figure(figsize=(10, 6))
    plt.plot(generation, best_fitness, 'r-', label="Best Fitness")
    plt.plot(generation, avg_fitness, 'b-', label="Avg Fitness")
    plt.title(f"Egocentric RNN Training (Avg {EPISODES_PER_GENOME} eps)")
    plt.grid()
    plt.legend()
    
    try:
        output_path = RESULTS_DIR / "fitness_rnn.png"
        plt.savefig(output_path)
    except Exception as e:
        print(f"Plot saving error: {e}")
    plt.close()

def plot_species(statistics):
    """ Generates Speciation Stackplot. """
    print("📊 Generating Speciation Plot...")
    
    species_sizes = statistics.get_species_sizes()
    
    if not species_sizes:
        print("⚠️ No speciation data found.")
        return

    num_generations = len(species_sizes)
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
        print(f"✅ Speciation plot saved to: {output_path.name}")
        
    except Exception as e:
        print(f"❌ Plotting error: {e}")
    
    plt.close()

# --- EVALUATION LOGIC ---

def eval_genome(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)
    
    try:
        from ocatari.core import OCAtari
        env = OCAtari(GAME_NAME, mode="ram", hud=False, render_mode=None)
    except Exception:
        return 0.0
    
    env = SpaceInvadersEgocentricWrapper(env, skip=4)
    
    total_fitness_acc = 0.0
    
    random.seed(os.getpid() + time.time())
    
    for _ in range(EPISODES_PER_GENOME):
        current_seed = random.randint(TRAINING_SEED_MIN, TRAINING_SEED_MAX)
        observation, info = env.reset(seed=current_seed)
        
        # Desynchronize aliens by waiting 0-30 frames
        random_delay = random.randint(0, 30)
        for _ in range(random_delay):
            observation, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated: break
        
        if len(observation) != 19:
            env.close()
            return 0.0
        
        episode_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        max_steps = 6000 
        
        net.reset()

        while not (terminated or truncated) and steps < max_steps:
            outputs = net.activate(observation)
            action = np.argmax(outputs)
            
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
        if episode_reward == 0:
            episode_reward += (steps / 10000.0)
            
        total_fitness_acc += episode_reward

    env.close()

    avg_fitness = total_fitness_acc / EPISODES_PER_GENOME
    return max(0.001, avg_fitness)

# --- MAIN ---

def run_training():
    print(f"📂 Loading Config: {CONFIG_PATH}")
    if not CONFIG_PATH.exists():
        print(f"❌ Config not found: {CONFIG_PATH}")
        return

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(CONFIG_PATH))

    if config.genome_config.num_inputs != 19:
        print(f"❌ CONFIG ERROR: num_inputs must be 19!")
        return

    p = neat.Population(config)
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    checkpoint_prefix = RESULTS_DIR / "neat-rnn-chk-"
    p.add_reporter(neat.Checkpointer(10, filename_prefix=str(checkpoint_prefix)))

    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"🚀 Starting RNN Training on {num_workers} workers...")
    
    pe = neat.ParallelEvaluator(num_workers, eval_genome)
    
    try:
        winner = p.run(pe.evaluate, GENERATIONS)
        
        with open(RESULTS_DIR / 'winner_ego.pkl', 'wb') as f:
            pickle.dump(winner, f)
        
        all_genomes = list(p.population.values())
        all_genomes.sort(key=lambda g: g.fitness if g.fitness else 0.0, reverse=True)
        top_3 = all_genomes[:3]
        
        top3_path = RESULTS_DIR / 'top3_list.pkl'
        with open(top3_path, 'wb') as f:
            pickle.dump(top_3, f)
            
        print(f"💾 Saved winner_ego.pkl")
        print(f"💾 Saved Top 3 list to: {top3_path.name}")

        plot_stats(stats)
        plot_species(stats)
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_training()