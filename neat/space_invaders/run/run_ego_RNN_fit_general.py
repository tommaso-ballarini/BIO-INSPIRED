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

# --- CONFIGURATION ---
CONFIG_PATH = project_root / 'config' / 'config_si_ego.txt'
RESULTS_DIR = project_root / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GAME_NAME = "ALE/SpaceInvaders-v5"
GENERATIONS = 300
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# --- TRAINING SEED CONFIG ---
TRAINING_SEED_MIN = 100      # Seeds < 100 reserved for testing
TRAINING_SEED_MAX = 100000   # Broad range for training
EPISODES_PER_GENOME = 3      # Average over 3 games for robustness

print(f"✅ Config: {GAME_NAME} (RNN + Survival Logic + Seed {TRAINING_SEED_MIN}+)")

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
    plt.title(f"Egocentric RNN Training (Survival)")
    plt.grid()
    plt.legend()
    
    try:
        output_path = RESULTS_DIR / "fitness_ego_rnn_fit.png"
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
        
        output_path = RESULTS_DIR / "speciation_ego_rnn_fit.png"
        plt.savefig(output_path)
        print(f"✅ Speciation plot saved to: {output_path.name}")
        
    except Exception as e:
        print(f"❌ Plotting error: {e}")
    
    plt.close()

# --- EVALUATION LOGIC ---

def eval_genome(genome, config):
    # 1. Create RNN Network
    net = neat.nn.RecurrentNetwork.create(genome, config)
    
    # 2. Env Setup (Once per genome for efficiency)
    try:
        # render_mode=None for maximum speed
        env = OCAtari(GAME_NAME, mode="ram", hud=False, render_mode=None)
    except Exception:
        return 0.0
    
    env = SpaceInvadersEgocentricWrapper(env, skip=4)
    
    # Ensure randomness in parallel processes
    random.seed(os.getpid() + time.time())
    
    fitness_history = []

    # --- EPISODE LOOP (3 Different Games) ---
    for episode in range(EPISODES_PER_GENOME):
        
        # A. Pick a random seed from the "safe" training range
        current_seed = random.randint(TRAINING_SEED_MIN, TRAINING_SEED_MAX)
        
        # B. Reset with specific seed
        observation, info = env.reset(seed=current_seed)
        
        # Random Delay (0-30 frames) to desync start conditions
        random_delay = random.randint(0, 30)
        for _ in range(random_delay):
            observation, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated: break
        
        # Wrapper check
        if len(observation) != 19:
            break 
        
        episode_fitness = 0.0  
        steps = 0
        terminated = False
        truncated = False
        max_steps = 6000 
        
        # IMPORTANT: Reset RNN internal state at start of episode
        net.reset()

        while not (terminated or truncated) and steps < max_steps:
            outputs = net.activate(observation)
            action = np.argmax(outputs)
            
            # --- FITNESS LOGIC ---
            danger_level = observation[3]
            is_safe = danger_level < 0.25 
            
            # Shot Penalty (Anti-Spam)
            if action == 1:
                episode_fitness -= 0.05 
            
            if is_safe:
                # Aim Bonus
                rel_x = observation[11]
                if abs(rel_x) < 0.15: 
                    episode_fitness += 0.02 
            else:
                # Danger Penalty
                episode_fitness -= (danger_level * 0.2) 

            # Env Step
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Kill Reward
            if reward > 0:
                episode_fitness += reward         
                episode_fitness += (reward * 0.5)

            steps += 1
            
        # Minimal survival bonus for the episode
        if episode_fitness <= 0:
            episode_fitness = max(0.001, steps / 10000.0)
            
        fitness_history.append(episode_fitness)

    env.close()
    
    # Return AVERAGE of episodes
    if not fitness_history:
        return 0.0
    return np.mean(fitness_history)

# --- MAIN ---

def run_training():
    print(f"📂 Loading Config: {CONFIG_PATH}")
    if not CONFIG_PATH.exists():
        print(f"❌ Config not found: {CONFIG_PATH}")
        return

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(CONFIG_PATH))

    # Verify Config
    if config.genome_config.num_inputs != 19:
        print(f"❌ CONFIG ERROR: num_inputs must be 19!")
        return

    p = neat.Population(config)
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    checkpoint_prefix = RESULTS_DIR / "neat-rnn-chk-"
    p.add_reporter(neat.Checkpointer(10, filename_prefix=str(checkpoint_prefix)))

    # Use defined workers
    pe = neat.ParallelEvaluator(NUM_WORKERS, eval_genome)
    
    try:
        print(f"🚀 Starting RNN Training (Survival + Aiming) on {EPISODES_PER_GENOME} seeds per genome...")
        winner = p.run(pe.evaluate, GENERATIONS)
        
        # Save Winner
        with open(RESULTS_DIR / 'winner_ego.pkl', 'wb') as f:
            pickle.dump(winner, f)
        
        # Save Top 3
        all_genomes = list(p.population.values())
        all_genomes.sort(key=lambda g: g.fitness if g.fitness else 0.0, reverse=True)
        top_3 = all_genomes[:3]
        
        top3_path = RESULTS_DIR / 'top3_list.pkl'
        with open(top3_path, 'wb') as f:
            pickle.dump(top_3, f)
            
        print(f"💾 Saved winner_ego.pkl and top3_list.pkl")
        
        print("\n📊 Generating Plots...")
        plot_stats(stats)
        plot_species(stats)
        print("✅ Plots generated in 'results/'")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_training()