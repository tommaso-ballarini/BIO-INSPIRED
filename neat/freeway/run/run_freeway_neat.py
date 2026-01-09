import os
import pickle
import sys
import numpy as np
import neat
import matplotlib.pyplot as plt
import gymnasium as gym
from pathlib import Path


# --- PATH CONFIGURATION ---
# Calculate project base directory
CURRENT_DIR = Path(__file__).resolve().parent 
FREEWAY_DIR = CURRENT_DIR.parent             
# Default paths for config and results
DEFAULT_CONFIG = str(FREEWAY_DIR / "config" / "neat_freeway_config.txt")
DEFAULT_OUTDIR = FREEWAY_DIR / "results" / "neat_freeway_baseline"

# Add base directory to sys.path to allow internal imports
if str(FREEWAY_DIR) not in sys.path:
    sys.path.insert(0, str(FREEWAY_DIR))

# ALE Atari Environment Setup
try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass

def plot_results(stats, save_dir):
    """
    Generates Fitness and Speciation plots based on training statistics.
    """
    print(f"\nGenerating plots in: {save_dir}")
    
    # --- 1. FITNESS EVOLUTION GRAPH ---
    if stats.most_fit_genomes:
        generation = range(len(stats.most_fit_genomes))
        best_fitness = [c.fitness for c in stats.most_fit_genomes]
        avg_fitness = np.array(stats.get_fitness_mean())
        stdev_fitness = np.array(stats.get_fitness_stdev())

        plt.figure(figsize=(12, 7))
        # Plotting the mean fitness with a shaded area for standard deviation
        plt.plot(generation, avg_fitness, 'b-', label="Average Fitness", alpha=0.6)
        plt.fill_between(generation, avg_fitness - stdev_fitness, avg_fitness + stdev_fitness, 
                         color='blue', alpha=0.1)
        # Plotting the best fitness in red (Raw Atari Score)
        plt.plot(generation, best_fitness, 'r-', label="Best Fitness (Raw Score)", linewidth=2)
        
        plt.title("NEAT Fitness Evolution - Freeway Baseline (RAW)")
        plt.xlabel("Generations")
        plt.ylabel("Game Points (Raw Score)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left")
        plt.savefig(save_dir / "fitness_history.png")
        plt.close()

    # --- 2. SPECIATION EVOLUTION GRAPH ---
    try:
        # get_species_sizes() returns a list of sizes for each species per generation
        species_sizes = stats.get_species_sizes()
        if species_sizes:
            plt.figure(figsize=(12, 7))
            # Stackplot to visualize how population is divided into species over time
            plt.stackplot(range(len(species_sizes)), np.array(species_sizes).T)
            plt.title("NEAT Species Evolution")
            plt.xlabel("Generations")
            plt.ylabel("Population Size")
            plt.savefig(save_dir / "speciation_history.png")
            plt.close()
            print("Visual reports successfully saved.")
    except Exception as e:
        print(f"Error plotting speciation: {e}")

def eval_genomes_factory(env_id, max_steps, episodes_per_genome, seed_base):
    """
    Factory that returns the evaluation function for NEAT.
    No wrapper is used here; only raw 128-byte RAM observations.
    """
    def eval_genomes(genomes, config):
        # Initialize RAW environment with RAM observation type
        env = gym.make(env_id, obs_type="ram")
        meanings = env.unwrapped.get_action_meanings()
        # Map indices for NOOP, UP, DOWN
        action_map = [meanings.index("NOOP"), meanings.index("UP"), meanings.index("DOWN")]

        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            scores = []

            for ep in range(episodes_per_genome):
                # Using a variable seed for training to handle environment stochasticity
                seed = seed_base + genome_id + ep
                obs, _ = env.reset(seed=seed)
                
                total_game_score = 0.0

                for t in range(max_steps):
                    # Normalize raw RAM bytes [0, 255] to [0.0, 1.0] for the Neural Network
                    obs_norm = obs.astype(np.float32) / 255.0
                    
                    # Network activation: choose action with highest output
                    action_idx = np.argmax(net.activate(obs_norm))
                    action = action_map[action_idx]
                    
                    # Environment step: ALE/Freeway-v5 includes sticky actions (stochastic)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_game_score += float(reward)

                    if terminated or truncated:
                        break

                scores.append(total_game_score)

            # Fitness is strictly defined as the average Raw Score (Atari points)
            genome.fitness = np.mean(scores)
        env.close()

    return eval_genomes

def main():
    # Create results directory
    output_path = Path(DEFAULT_OUTDIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not Path(DEFAULT_CONFIG).exists():
        print(f"ERROR: Config file not found at {DEFAULT_CONFIG}. Check your project structure.")
        return

    # Load NEAT configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, DEFAULT_CONFIG)

    # Initialize the population
    p = neat.Population(config)
    
    # Add reporters to track progress in terminal and store stats
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    try:
        # Run evolution for N generations
        # In Raw Baseline, it might take 100+ generations to see progress
        winner = p.run(eval_genomes_factory("ALE/Freeway-v5", 1500, 1, 42), n=30)
    except KeyboardInterrupt:
        print("\nEvolution interrupted by user. Saving current results...")
        winner = p.best_genome # Use the best genome found so far

    # --- SAVE RESULTS ---
    with open(output_path / "winner.pkl", "wb") as f:
        pickle.dump(winner, f)
    with open(output_path / "stats.pkl", "wb") as f:
        pickle.dump(stats, f)
    
    # Generate Plots
    plot_results(stats, output_path)
    print(f"\nFinal results saved in: {output_path}")

if __name__ == "__main__":
    main()