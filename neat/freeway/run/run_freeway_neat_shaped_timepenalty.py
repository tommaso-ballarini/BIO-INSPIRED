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
# Ensure your config has: feed_forward = True, num_inputs = 22
DEFAULT_CONFIG = str(PROJECT_ROOT / "config" / "neat_freeway_config.txt")
DEFAULT_OUTDIR = PROJECT_ROOT / "results" / "neat_freeway_ff_shaped_timepenalty"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import ale_py
    gym.register_envs(ale_py)
except Exception: pass

def plot_results(stats, save_dir):
    """ Generates Fitness and Speciation plots with standard deviation intervals. """
    print(f"\nGenerating plots in: {save_dir}")
    if stats.most_fit_genomes:
        gen = range(len(stats.most_fit_genomes))
        best_fit = [c.fitness for c in stats.most_fit_genomes]
        avg_fit = np.array(stats.get_fitness_mean())
        stdev_fit = np.array(stats.get_fitness_stdev())

        plt.figure(figsize=(12, 7))
        plt.plot(gen, avg_fit, 'b-', label="Average Fitness", alpha=0.8)
        plt.fill_between(gen, avg_fit - stdev_fit, avg_fit + stdev_fit, color='blue', alpha=0.2)
        plt.plot(gen, best_fit, 'r-', label="Best Fitness", linewidth=2)
        plt.title("NEAT FFNN - Shaped Fitness + Time Penalty")
        plt.xlabel("Generations")
        plt.ylabel("Fitness Value")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.savefig(save_dir / "fitness_history.png")
        plt.close()

def eval_genomes_factory(env_id, max_steps, episodes_per_genome, seed_base):
    def eval_genomes(genomes, config):
        from wrapper.freeway_wrapper import FreewaySpeedWrapper
        
        raw_env = gym.make(env_id, obs_type="ram")
        env = FreewaySpeedWrapper(raw_env, normalize=True, mirror_last_5=True)
        
        meanings = env.unwrapped.get_action_meanings()
        action_map = [meanings.index("NOOP"), meanings.index("UP"), meanings.index("DOWN")]

        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            fitness_history = []

            for ep in range(episodes_per_genome):
                seed = seed_base + genome_id + ep
                obs, _ = env.reset(seed=seed)
                
                total_atari_score = 0.0
                max_y_reached = 0.0
                collision_count = 0
                prev_y = obs[0]
                step_count = 0

                for t in range(max_steps):
                    action_idx = np.argmax(net.activate(obs))
                    obs, reward, terminated, truncated, _ = env.step(action_map[action_idx])
                    
                    current_y = obs[0]
                    total_atari_score += float(reward)
                    step_count += 1

                    if current_y > max_y_reached:
                        max_y_reached = current_y
                    
                    # Collision detection
                    if current_y < prev_y - 0.05:
                        collision_count += 1
                    
                    prev_y = current_y
                    if terminated or truncated: break

                # --- SHAPED FITNESS WITH TIME PENALTY ---
                # Rewards: Point (50), Progress (20)
                # Penalties: Collision (-2.0), Time (-0.1 per step)
                fitness = (total_atari_score * 100.0) + \
                          (max_y_reached * 20.0) - \
                          (collision_count * 2.0) - \
                          (step_count * 0.01)
                
                fitness_history.append(max(0.0, fitness))

            genome.fitness = np.mean(fitness_history)
        env.close()

    return eval_genomes

def main():
    out_path = Path(DEFAULT_OUTDIR)
    out_path.mkdir(parents=True, exist_ok=True)
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, DEFAULT_CONFIG)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # We run for 50 generations with a fixed seed base for training
    winner = p.run(eval_genomes_factory("ALE/Freeway-v5", 1500, 1, 42), n=50)

    with open(out_path / "winner.pkl", "wb") as f: pickle.dump(winner, f)
    with open(out_path / "stats.pkl", "wb") as f: pickle.dump(stats, f)
    plot_results(stats, out_path)
    print(f"\nTraining complete. Graphs and winner saved in: {out_path}")

if __name__ == "__main__":
    main()