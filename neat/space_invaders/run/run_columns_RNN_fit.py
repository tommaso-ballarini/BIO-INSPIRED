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
    print("❌ ERROR: OCAtari library not installed.")
    sys.exit(1)

try:
    from wrapper.wrapper_si_columns_RNN import SpaceInvadersColumnWrapper
except ImportError:
    print("❌ ERROR: 'wrapper_si_columns_RNN.py' not found in wrapper folder!")
    sys.exit(1)

# --- GLOBAL SETTINGS ---
CONFIG_PATH = project_root / 'config' / 'config_si_columns_RNN.txt'
RESULTS_DIR = project_root / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GAME_NAME = "SpaceInvadersNoFrameskip-v4"
GENERATIONS = 30
FIXED_SEED = 42 

print(f"✅ Env Config: {GAME_NAME} with Column Wrapper (RNN Mode)")
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
    plt.title(f"Columns RNN Training (Seed {FIXED_SEED})")
    plt.xlabel("Generations")
    plt.ylabel("Fitness (Score)")
    plt.grid()
    plt.legend()
    
    try:
        output_path = RESULTS_DIR / "fitness_columns_rnn_fit.png"
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
    # 1. Network: Recurrent (RNN)
    net = neat.nn.RecurrentNetwork.create(genome, config)
    
    # 2. Env Setup (OCAtari + Wrapper)
    try:
        import ale_py # Re-import for safety in subprocess
        env = OCAtari(GAME_NAME, mode="ram", hud=False, render_mode=None)
    except Exception as e:
        print(f"⚠️ OCAtari init error: {e}")
        return 0.0
    
    # Apply Column Wrapper (10 columns, skip 4 frames)
    env = SpaceInvadersColumnWrapper(env, n_columns=10, skip=4)
    
    # 3. Deterministic Reset
    observation, info = env.reset(seed=FIXED_SEED)
    
    # Check Input Size (Expected 32 for RNN config)
    if len(observation) != 32:
        print(f"⚠️ SIZE ERROR: Expected 32, got {len(observation)}")
        env.close()
        return 0.0

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    max_steps = 10000 
    
    # --- TRACKING ---
    x_positions = [] 
    shots_fired = 0 

    while not (terminated or truncated) and steps < max_steps:
        inputs = observation
        
        # RNN Activation
        outputs = net.activate(inputs)
        action = np.argmax(outputs)
        
        # Action 1 = FIRE in Space Invaders
        if action == 1:
            shots_fired += 1
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Sample position (every 10 steps)
        if steps % 10 == 0:
            try:
                # RAM[28] is usually player X position in Space Invaders
                player_x = env.unwrapped.ale.getRAM()[28]
                x_positions.append(player_x)
            except:
                pass

        steps += 1

    env.close()

    # --- FITNESS CALCULATION (Modified) ---
    
    fitness = total_reward
    
    # 1. SPAM PENALTY
    # Subtract 0.2 points per shot. Example: 100 misses = -20 fitness.
    # Killing an alien (10-30 pts) is still net positive if efficient.
    fitness -= (shots_fired * 0.2)

    # 2. ANTI-CAMPING PENALTY
    if len(x_positions) > 5:
        min_x = np.min(x_positions)
        max_x = np.max(x_positions)
        coverage = max_x - min_x 
        
        # Screen width ~160. Require covering at least 40px (25%)
        if coverage < 40:
            # Drastic penalty for staying in one spot (50% reduction)
            fitness = fitness * 0.5
        else:
            # Bonus for movement/exploration
            fitness += 20 

    # Avoid negative fitness (NEAT sometimes breaks with fitness < 0)
    return max(0.1, fitness)

# --- MAIN ---

def run_columns():
    print(f"📂 Loading Config: {CONFIG_PATH}")
    if not CONFIG_PATH.exists():
        print(f"❌ Config not found! Create {CONFIG_PATH} with num_inputs=32 and feed_forward=False.")
        return

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(CONFIG_PATH))

    # Verify Config
    if config.genome_config.num_inputs != 32:
        print(f"❌ CONFIG ERROR: num_inputs is {config.genome_config.num_inputs}, must be 32!")
        return

    p = neat.Population(config)
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    checkpoint_prefix = RESULTS_DIR / "neat-col-rnn-checkpoint-"
    p.add_reporter(neat.Checkpointer(10, filename_prefix=str(checkpoint_prefix)))

    # Multiprocessing
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"🚀 Starting Columns RNN Training on {num_workers} workers...")
    
    pe = neat.ParallelEvaluator(num_workers, eval_genome)
    
    try:
        winner = p.run(pe.evaluate, GENERATIONS)
        
        best_ever = stats.best_genome()
        print(f"\n🏆 Training Complete.")
        print(f"💎 Best Ever Fitness: {best_ever.fitness}")
        
        # Save Winner
        with open(RESULTS_DIR / 'columns_winner_RNN_fit.pkl', 'wb') as f:
            pickle.dump(best_ever, f)
        print(f"💾 Saved to: columns_winner_RNN_fit.pkl")

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