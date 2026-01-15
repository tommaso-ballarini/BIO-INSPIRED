import os
import sys
from datetime import datetime
import neat
import pickle
import multiprocessing
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random

# Add path to import the wrapper
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from wrapper.wrapper_ffnn_dynamic import BioSkiingOCAtariWrapper
except ImportError:
    print("Error: 'wrapper/wrapper_ffnn_dynamic.py' not found")
    sys.exit(1)

# --- CONFIGURATION ---
NUM_GENERATIONS = 150
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)
TRAINING_SEED_MIN = 100      # Seed < 100 reserved for testing
TRAINING_SEED_MAX = 100000 
CONFIG_FILENAME = "config_wrapper_ffnn_dynamic.txt"
CHECKPOINT_PREFIX = "neat-checkpoint-"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join(parent_dir, 'evolution_results', 'wrapper_ffnn_dynamic_run')

def eval_genome(genome, config, curriculum_mode="beginner"):
    """
    Function executed by each worker in parallel.
    Evaluates a single genome and returns fitness.
    """
    try:
        env = BioSkiingOCAtariWrapper(render_mode="rgb_array", curriculum_mode=curriculum_mode)
    except:
        env = BioSkiingOCAtariWrapper(render_mode=None, curriculum_mode=curriculum_mode)
          
    current_seed = random.randint(TRAINING_SEED_MIN, TRAINING_SEED_MAX)

    observation, info = env.reset(seed=current_seed)
    
    # Create Neural Network (Feed-Forward for memory)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    done = False
    total_reward = 0.0
    steps = 0
    
    while not done and steps < 4000: 
        inputs = observation
        
        # Check input dimensions
        if len(inputs) != config.genome_config.num_inputs:
             print(f"Mismatch: Wrapper gives {len(inputs)}, Config expects {config.genome_config.num_inputs}")
             return 0.0

        # Network Activation
        output = net.activate(inputs)
        action = np.argmax(output)
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        done = terminated or truncated
        steps += 1
        
    env.close()
    return total_reward

def plot_results(stats, save_dir):
    """
    Generates plots by reading generation_statistics directly.
    """
    print(f"\nGenerating plots in: {save_dir}")
    
    # --- 1. FITNESS GRAPH ---
    if stats.most_fit_genomes:
        generation = range(len(stats.most_fit_genomes))
        best_fitness = [c.fitness for c in stats.most_fit_genomes]
        avg_fitness = np.array(stats.get_fitness_mean())
        stdev_fitness = np.array(stats.get_fitness_stdev())

        plt.figure(figsize=(10, 6))
        plt.plot(generation, avg_fitness, 'b-', label="Average Fitness", alpha=0.6)
        plt.fill_between(generation, avg_fitness - stdev_fitness, avg_fitness + stdev_fitness, 
                         color='blue', alpha=0.1)
        plt.plot(generation, best_fitness, 'r-', label="Best Fitness", linewidth=2)
        
        plt.title("Fitness Evolution Skiing")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(os.path.join(save_dir, f"fitness_history_{timestamp}.png"), dpi=100)
        plt.close()
        print("Fitness graph saved.")

    # --- 2. SPECIATION GRAPH (FIXED) ---
    
    try:
        if not stats.generation_statistics:
            print("No speciation data available.")
            return
        
        # Find all species IDs
        all_species = set()
        for gen_data in stats.generation_statistics:
            all_species.update(gen_data.keys())
        
        all_species = sorted(list(all_species))
        
        # Build history matrix (Species x Generations)
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
            plt.stackplot(range(len(stats.generation_statistics)), 
                          species_history, labels=[f"ID {i}" for i in all_species])
            plt.title("Species Evolution")
            plt.xlabel("Generations")
            plt.ylabel("Population")
            
            # Legend only if few species
            if len(all_species) < 15:
                plt.legend(loc='upper left')
            
            plt.savefig(os.path.join(save_dir, f"speciation_{timestamp}.png"), dpi=100)
            plt.close()
            print("Speciation graph saved.")
            
    except Exception as e:
        print(f"Error plotting speciation: {e}")
        import traceback
        traceback.print_exc()

def run_training():
    # 1. Setup Paths
    config_path = os.path.join(parent_dir, 'config', CONFIG_FILENAME)
    results_dir = os.path.join(parent_dir, 'evolution_results', 'wrapper_ffnn_dynamic_run')
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Starting Wrapper FFNN Training")
    print(f"Config: {config_path}")
    print(f"Output: {results_dir}")
    print(f"Workers: {NUM_WORKERS}")

    # 2. Load Configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    # Quick input check
    if config.genome_config.num_inputs != 9:
        print(f"WARNING: OCAtari wrapper uses 9 inputs.")
        print(f"    Config has {config.genome_config.num_inputs}.")
        print("    Update config to 'num_inputs = 9'!")

    # 3. Initialize Population
    p = neat.Population(config)

    # 4. Add Reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # 5. Run Parallel Evolution
    
    class CurriculumEvaluator:
        def __init__(self, num_workers):
            self.num_workers = num_workers
            self.curriculum = "beginner"
            
        def evaluate(self, genomes, config):

            if hasattr(self, 'best_fitness'):
                if self.best_fitness > 9000 and self.avg_fitness > 1600:
                    if self.curriculum == "beginner":
                        self.curriculum = "advanced"
                        print("\n🎓 CURRICULUM UPGRADE: Advanced Mode!")
                            
            jobs = []
            for genome_id, genome in genomes:
                jobs.append((genome, config, self.curriculum))
            
            with multiprocessing.Pool(self.num_workers) as pool:
                results = pool.starmap(eval_genome, jobs)
            
            for (genome_id, genome), fitness in zip(genomes, results):
                genome.fitness = fitness
            
            fitnesses = [g.fitness for _, g in genomes]
            self.best_fitness = max(fitnesses)
            self.avg_fitness = np.mean(fitnesses)
    
    curriculum_eval = CurriculumEvaluator(NUM_WORKERS)
    
    print(f"\n🎿 Starting evolution with Curriculum Learning")
    print(f"   Initial mode: {curriculum_eval.curriculum}")
    winner = p.run(curriculum_eval.evaluate, NUM_GENERATIONS)
    
    # 6. Save Winner
    print(f"\nBest genome found! Fitness: {winner.fitness}")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(results_dir, f"best_agent_{timestamp}.pkl")
    
    with open(save_path, 'wb') as f:
        pickle.dump(winner, f)
    
    print(f"Saved in: {save_path}")
    plot_results(stats, results_dir)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_training()