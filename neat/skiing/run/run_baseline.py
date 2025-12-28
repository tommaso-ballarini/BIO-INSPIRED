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

# Register Atari environments
gym.register_envs(ale_py)

# --- EXPERIMENT CONFIGURATION ---
ENV_ID = "ALE/Skiing-v5"
CONFIG_FILE_NAME = "config_baseline.txt"
NUM_GENERATIONS = 20
MAX_STEPS = 2000
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
OUTPUT_DIR = os.path.join(project_root, "evolution_results", "baseline_run")
CONFIG_PATH = os.path.join(project_root, "config", CONFIG_FILE_NAME)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def eval_genome(genome, config):
    """
    Baseline evaluation using raw RAM state.
    Fitness is the native game reward.
    """
    env = gym.make(ENV_ID, obs_type="ram", render_mode=None)
    observation, info = env.reset()
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    total_reward = 0.0
    steps = 0
    done = False
    
    while not done and steps < MAX_STEPS:
        # Normalize RAM bytes (0-255) to float (0.0-1.0)
        inputs = observation / 255.0
        
        # Activate network
        output = net.activate(inputs)
        action = np.argmax(output) 
        
        # Step environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        done = terminated or truncated
        
    env.close()
    return float(total_reward)

def plot_results(stats, save_dir):
    """
    Plots Fitness history and Speciation safely.
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
        
        plt.title("Baseline Fitness Evolution")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        fit_path = os.path.join(save_dir, f"baseline_fitness_{timestamp}.png")
        plt.savefig(fit_path)
        plt.close()
        print(f"Fitness plot saved: {fit_path}")

    # --- 2. SPECIATION PLOT (ROBUST) ---
    try:

        if not stats.generation_statistics:
            print("No speciation data available.")
            return

        # 1. Collect all unique species IDs
        all_species = set()
        for gen_data in stats.generation_statistics:
            all_species.update(gen_data.keys())
        all_species = sorted(list(all_species))
        
        # 2. Build population history matrix
        species_history = []
        for gen_data in stats.generation_statistics:
            row = []
            for s_id in all_species:
                if s_id in gen_data:
                    species_obj = gen_data[s_id]
                    try:
                        if hasattr(species_obj, 'members'):
                            row.append(len(species_obj.members))
                        else:
                            row.append(len(species_obj))
                    except:
                        row.append(0)
                else:
                    row.append(0)
            species_history.append(row)
        
        # 3. Plot
        species_history = np.array(species_history).T
        
        if len(species_history) > 0:
            plt.figure(figsize=(10, 6))
            plt.stackplot(range(len(stats.generation_statistics)), species_history, labels=[f"ID {i}" for i in all_species])
            plt.title("Baseline Speciation")
            plt.xlabel("Generations")
            plt.ylabel("Population")
            
            if len(all_species) < 15:
                plt.legend(loc='upper left')
            
            spec_path = os.path.join(save_dir, f"baseline_speciation_{timestamp}.png")
            plt.savefig(spec_path)
            plt.close()
            print(f"Speciation plot saved: {spec_path}")
            
    except Exception as e:
        print(f"Error plotting speciation: {e}")
        import traceback
        traceback.print_exc()

def run_baseline():
    print("STARTING BASELINE EXPERIMENT")
    print(f"Config: {CONFIG_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Config file not found at {CONFIG_PATH}")
        return

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)
    
    p = neat.Population(config)
    
    # Add Reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # Parallel Execution
    pe = neat.ParallelEvaluator(NUM_WORKERS, eval_genome)
    
    try:
        winner = p.run(pe.evaluate, NUM_GENERATIONS)
        
        # Save Winner
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(OUTPUT_DIR, f"baseline_winner_{timestamp}.pkl")
        
        with open(save_path, "wb") as f:
            pickle.dump(winner, f)
            
        print(f"Baseline run completed. Winner saved to: {save_path}")
        print(f"Best Fitness: {winner.fitness}")

        # Generate Plots
        plot_results(stats, OUTPUT_DIR)
        
    except Exception as e:
        print(f"Critical Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
        
    run_baseline()