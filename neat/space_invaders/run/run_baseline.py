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

# --- IMPORTS ---
try:
    from ocatari.core import OCAtari
except ImportError:
    print("❌ ERROR: OCAtari not installed.")
    sys.exit(1)

try:
    from wrapper.wrapper_si_columns import SpaceInvadersColumnWrapper
except ImportError:
    print("❌ ERROR: 'wrapper_si_columns.py' not found in wrapper folder!")
    sys.exit(1)

# --- GLOBAL SETTINGS ---
CONFIG_PATH = project_root / 'config' / 'config_si_columns.txt'
RESULTS_DIR = project_root / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GAME_NAME = "ALE/SpaceInvaders-v5"
GENERATIONS = 50
FIXED_SEED = 42 

print(f"✅ Env Config: {GAME_NAME} with Column Wrapper (FFNN Mode)")
print(f"🔒 Fixed Seed: {FIXED_SEED}")

# --- PLOTTING FUNCTIONS ---

def plot_stats(statistics):
    """ Plots Average and Best Fitness. """
    print("📊 Generating Fitness Plot...")
    if not statistics.most_fit_genomes:
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())

    plt.figure(figsize=(10, 6))
    plt.plot(generation, best_fitness, 'r-', label="Best Fitness")
    plt.plot(generation, avg_fitness, 'b-', label="Avg Fitness")
    plt.title(f"Columns FFNN Training (Seed {FIXED_SEED})")
    plt.xlabel("Generations")
    plt.ylabel("Fitness (Score)")
    plt.grid()
    plt.legend()
    
    try:
        output_path = RESULTS_DIR / "fitness_columns_ffnn.png"
        plt.savefig(output_path)
    except Exception as e:
        print(f"❌ Fitness plot error: {e}")
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
    # 1. Network: FeedForward (FFNN)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # 2. Env Setup (OCAtari + Wrapper)
    try:
        env = OCAtari(GAME_NAME, mode="ram", hud=False, render_mode=None)
        # Apply Column Wrapper (10 columns, skip 4 frames)
        env = SpaceInvadersColumnWrapper(env, n_columns=10, skip=4)
    except Exception:
        return 0.0

    # 3. Deterministic Reset
    observation, info = env.reset(seed=FIXED_SEED)
    
    # Check Input Size (Must match config: 96)
    if len(observation) != 96:
        print(f"⚠️ SIZE ERROR: Expected 96, got {len(observation)}")
        env.close()
        return 0.0

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    max_steps = 10000 

    while not (terminated or truncated) and steps < max_steps:
        inputs = observation
        
        # FFNN Activation
        outputs = net.activate(inputs)
        action = np.argmax(outputs)
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    env.close()
    return total_reward

# --- MAIN ---

def run_columns():
    print(f"📂 Loading Config: {CONFIG_PATH}")
    if not CONFIG_PATH.exists():
        print(f"❌ Config not found! Create {CONFIG_PATH} with num_inputs=96.")
        return

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(CONFIG_PATH))

    # Verify Config
    if config.genome_config.num_inputs != 96:
        print(f"❌ CONFIG ERROR: num_inputs is {config.genome_config.num_inputs}, must be 96!")
        return

    p = neat.Population(config)
    
    # Reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    checkpoint_prefix = RESULTS_DIR / "neat-col-ffnn-checkpoint-"
    p.add_reporter(neat.Checkpointer(10, filename_prefix=str(checkpoint_prefix)))

    # Multiprocessing
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"🚀 Starting Columns FFNN Training on {num_workers} workers...")
    
    pe = neat.ParallelEvaluator(num_workers, eval_genome)
    
    try:
        winner = p.run(pe.evaluate, GENERATIONS)
        
        print(f"\n🏆 Training Complete.")
        print(f"💎 Best Ever Fitness: {winner.fitness}")
        
        # Save Winner
        with open(RESULTS_DIR / 'columns_winner_ffnn.pkl', 'wb') as f:
            pickle.dump(winner, f)
        print(f"💾 Saved to: columns_winner_ffnn.pkl")

        # Generate Plots
        plot_stats(stats)
        plot_species(stats)
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_columns()