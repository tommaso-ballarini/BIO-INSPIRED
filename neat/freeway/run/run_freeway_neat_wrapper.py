import os
import pickle
import sys
import numpy as np
import neat
import matplotlib.pyplot as plt
import gymnasium as gym
from pathlib import Path

# --- PATH CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
FREEWAY_DIR = SCRIPT_DIR.parent
# IMPORTANT: Ensure your config file has num_inputs = 22
DEFAULT_CONFIG = str(FREEWAY_DIR / "config" / "neat_freeway_config.txt")
DEFAULT_OUTDIR = FREEWAY_DIR / "results" / "neat_freeway_ff_wrapper"

if str(FREEWAY_DIR) not in sys.path:
    sys.path.insert(0, str(FREEWAY_DIR))

# Force ALE Registration for Gymnasium
try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass

def plot_results(stats, save_dir):
    """
    Generates Fitness and Speciation plots to visualize learning progress.
    """
    print(f"\nGenerating plots in: {save_dir}")
    if stats.most_fit_genomes:
        generation = range(len(stats.most_fit_genomes))
        best_fitness = [c.fitness for c in stats.most_fit_genomes]
        avg_fitness = np.array(stats.get_fitness_mean())
        stdev_fitness = np.array(stats.get_fitness_stdev())

        plt.figure(figsize=(12, 7))
        plt.plot(generation, avg_fitness, 'b-', label="Average Fitness", alpha=0.6)
        plt.fill_between(generation, avg_fitness - stdev_fitness, avg_fitness + stdev_fitness, 
                         color='blue', alpha=0.1)
        plt.plot(generation, best_fitness, 'r-', label="Best Fitness (Raw Score)", linewidth=2)
        plt.title("NEAT Fitness Evolution - Freeway Wrapper (22 Inputs)")
        plt.xlabel("Generations")
        plt.ylabel("Game Points (Raw Score)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left")
        plt.savefig(save_dir / "fitness_history.png")
        plt.close()

    try:
        species_sizes = stats.get_species_sizes()
        if species_sizes:
            plt.figure(figsize=(12, 7))
            plt.stackplot(range(len(species_sizes)), np.array(species_sizes).T)
            plt.title("NEAT Species Evolution")
            plt.xlabel("Generations")
            plt.ylabel("Population Size")
            plt.savefig(save_dir / "speciation_history.png")
            plt.close()
    except Exception as e:
        print(f"Error plotting speciation: {e}")

def eval_genomes_factory(env_id, max_steps, episodes_per_genome, seed_base):
    """
    Factory for NEAT evaluation using the Speed Wrapper (22 features).
    """
    def eval_genomes(genomes, config):
        # Import the specific Speed Wrapper
        from wrapper.freeway_wrapper import FreewaySpeedWrapper
        
        # Initialize environment with RAM observations and apply the wrapper
        raw_env = gym.make(env_id, obs_type="ram")
        env = FreewaySpeedWrapper(raw_env, normalize=True, mirror_last_5=True)
        
        meanings = env.unwrapped.get_action_meanings()
        action_map = [meanings.index("NOOP"), meanings.index("UP"), meanings.index("DOWN")]

        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            scores = []

            for ep in range(episodes_per_genome):
                # Stochastic seed for training robustness
                seed = seed_base + genome_id + ep
                obs, _ = env.reset(seed=seed)
                
                total_game_score = 0.0

                for t in range(max_steps):
                    # Activate network with the 22 features from the wrapper
                    action_idx = np.argmax(net.activate(obs))
                    action = action_map[action_idx]
                    
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_game_score += float(reward)

                    if terminated or truncated:
                        break

                scores.append(total_game_score)

            # Fitness is defined as the Raw Game Score
            genome.fitness = np.mean(scores)
        env.close()

    return eval_genomes

def main():
    output_path = Path(DEFAULT_OUTDIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not Path(DEFAULT_CONFIG).exists():
        print(f"ERROR: Config file not found at {DEFAULT_CONFIG}. Set num_inputs = 22.")
        return

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, DEFAULT_CONFIG)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    try:
        # Run for 50 generations as a starting point for Speed Wrapper
        winner = p.run(eval_genomes_factory("ALE/Freeway-v5", 1500, 1, 42), n=30)
    except KeyboardInterrupt:
        print("\nSaving current progress...")
        winner = p.best_genome

    # Save Winner and Stats
    with open(output_path / "winner.pkl", "wb") as f:
        pickle.dump(winner, f)
    with open(output_path / "stats.pkl", "wb") as f:
        pickle.dump(stats, f)
    
    plot_results(stats, output_path)
    print(f"\nTraining Complete. Results saved in: {output_path}")

if __name__ == "__main__":
    main()