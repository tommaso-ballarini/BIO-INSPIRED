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
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = str(PROJECT_ROOT / "config" / "neat_freeway_config.txt")
DEFAULT_OUTDIR = PROJECT_ROOT / "results" / "neat_freeway_rnn_shaped_parallel"

# Add the project root so workers can import the wrapper
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import ale_py
    gym.register_envs(ale_py)
except Exception: pass

# --- WORKER CONFIG ---
ENV_ID = "ALE/Freeway-v5"
MAX_STEPS = 1500
EPISODES_PER_GENOME = 3        # Average over 3 episodes
TRAINING_SEED_MIN = 100        # Seeds < 100 reserved for test
TRAINING_SEED_MAX = 100000     # Training seeds
NUM_WORKERS = 28               # Parallel workers

def plot_results(stats, save_dir):
    """Generates plots including fitness intervals."""
    print(f"\nGenerating plots in: {save_dir}")
    
    if stats.most_fit_genomes:
        gen = range(len(stats.most_fit_genomes))
        best_fit = [c.fitness for c in stats.most_fit_genomes]
        avg_fit = np.array(stats.get_fitness_mean())
        stdev_fit = np.array(stats.get_fitness_stdev())

        plt.figure(figsize=(12, 7))
        plt.plot(gen, avg_fit, 'b-', label="Average Fitness", alpha=0.8)
        plt.fill_between(gen, avg_fit - stdev_fit, avg_fit + stdev_fit, 
                         color='blue', alpha=0.2, label="Fitness Std Dev")
        plt.plot(gen, best_fit, 'r-', label="Best Fitness", linewidth=2)
        
        plt.title("NEAT RNN - Shaped Fitness Evolution (Parallel Run)")
        plt.xlabel("Generations")
        plt.ylabel("Fitness Value")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc="upper left")
        plt.savefig(save_dir / "fitness_history.png")
        plt.close()

    try:
        species_sizes = stats.get_species_sizes()
        if species_sizes:
            plt.figure(figsize=(12, 7))
            plt.stackplot(range(len(species_sizes)), np.array(species_sizes).T)
            plt.title("NEAT RNN - Speciation History")
            plt.xlabel("Generations")
            plt.ylabel("Population Size")
            plt.savefig(save_dir / "speciation_history.png")
            plt.close()
    except Exception: pass

def eval_genome(genome, config):
    """
    Evaluates a SINGLE genome.
    This function is pickled and sent to worker processes.
    """
    # Worker-local imports to support multiprocessing
    import random
    from wrapper.freeway_wrapper import FreewaySpeedWrapper
    
    try:
        raw_env = gym.make(ENV_ID, obs_type="ram")
        env = FreewaySpeedWrapper(raw_env, normalize=True, mirror_last_5=True)
    except Exception as e:
        print(f"Error creating env in worker: {e}")
        return 0.0
    
    meanings = env.unwrapped.get_action_meanings()
    action_map = [meanings.index("NOOP"), meanings.index("UP"), meanings.index("DOWN")]

    net = neat.nn.RecurrentNetwork.create(genome, config)
    fitness_history = []

    for ep in range(EPISODES_PER_GENOME):
        # Randomize seed to reduce overfitting to traffic patterns
        seed = random.randint(TRAINING_SEED_MIN, TRAINING_SEED_MAX)
        
        obs, _ = env.reset(seed=seed)
        
        total_atari_score = 0.0
        max_y_reached = 0.0
        collision_count = 0
        prev_y = obs[0] # Initial normalized Y

        for t in range(MAX_STEPS):
            outputs = net.activate(obs)
            action_idx = np.argmax(outputs)
            
            obs, reward, terminated, truncated, _ = env.step(action_map[action_idx])
            
            current_y = obs[0]
            total_atari_score += float(reward)

            # Track progress
            if current_y > max_y_reached:
                max_y_reached = current_y
            
            # Collision heuristic (sudden Y drop)
            if current_y < prev_y - 0.05:
                collision_count += 1
            
            prev_y = current_y
            if terminated or truncated: break

        # Shaped fitness weights: score (50), progress (10), collisions (-0.5)
        shaped_fit = (total_atari_score * 50.0) + (max_y_reached * 10.0) - (collision_count * 0.5)
        fitness_history.append(shaped_fit)

    env.close()
    return np.mean(fitness_history)

def main():
    out_path = Path(DEFAULT_OUTDIR)
    out_path.mkdir(parents=True, exist_ok=True)
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, DEFAULT_CONFIG)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    print(f"Initializing ParallelEvaluator with {NUM_WORKERS} workers...")
    pe = neat.ParallelEvaluator(NUM_WORKERS, eval_genome)

    try:
        # Evolution loop
        winner = p.run(pe.evaluate, n=250)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current best.")
        winner = p.best_genome

    # Save results
    with open(out_path / "winner.pkl", "wb") as f: pickle.dump(winner, f)
    with open(out_path / "stats.pkl", "wb") as f: pickle.dump(stats, f)
    plot_results(stats, out_path)
    print(f"\nRNN Shaped Training Finished (Parallel). Results in: {out_path}")

if __name__ == "__main__":
    main()
