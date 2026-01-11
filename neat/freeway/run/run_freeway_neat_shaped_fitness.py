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
# Ensure config has num_inputs = 22
DEFAULT_CONFIG = str(PROJECT_ROOT / "config" / "neat_freeway_config.txt")
DEFAULT_OUTDIR = PROJECT_ROOT / "results" / "neat_freeway_shaped_fitness"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force ALE Registration
try:
    import ale_py
    gym.register_envs(ale_py)
except Exception: pass

def plot_results(stats, save_dir):
    """ Generates performance and speciation plots in English. """
    print(f"\nGenerating plots in: {save_dir}")
    if stats.most_fit_genomes:
        gen = range(len(stats.most_fit_genomes))
        best_fit = [c.fitness for c in stats.most_fit_genomes]
        avg_fit = np.array(stats.get_fitness_mean())
        
        plt.figure(figsize=(12, 7))
        plt.plot(gen, avg_fit, 'b-', label="Average Shaped Fitness", alpha=0.6)
        plt.plot(gen, best_fit, 'r-', label="Best Shaped Fitness", linewidth=2)
        plt.title("NEAT Evolution - Shaped Fitness (Score + Y + Collision Penalty)")
        plt.xlabel("Generations")
        plt.ylabel("Fitness Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_dir / "fitness_history.png")
        plt.close()

def eval_genomes_factory(env_id, max_steps, episodes_per_genome, seed_base):
    """
    Evaluation factory with Fitness Shaping.
    Rewards: Atari Points (High), Chicken Y (Continuous), Collision (Penalty).
    """
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
                collisions = 0
                
                # Feature index 0 is normalized Y (0.0 = start, 1.0 = goal)
                prev_y = obs[0] 

                for t in range(max_steps):
                    action_idx = np.argmax(net.activate(obs))
                    obs, reward, terminated, truncated, _ = env.step(action_map[action_idx])
                    
                    current_y = obs[0]
                    total_atari_score += float(reward)

                    # Update max Y reached (progress reward)
                    if current_y > max_y_reached:
                        max_y_reached = current_y
                    
                    # Detect collision (Y drops significantly)
                    if current_y < prev_y - 0.05: # Threshold for collision detection
                        collisions += 1
                    
                    prev_y = current_y
                    if terminated or truncated: break

                # --- FITNESS SHAPING LOGIC ---
                # 1. Big reward for each point (Atari Score)
                # 2. Medium reward for the highest Y achieved (encourages moving up)
                # 3. Small penalty for collisions (encourages dodging)
                shaped_fitness = (total_atari_score * 20.0) + (max_y_reached * 10.0) - (collisions * 0.5)
                fitness_history.append(shaped_fitness)

            # Assign average fitness across episodes
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

    eval_func = eval_genomes_factory("ALE/Freeway-v5", 1500, 1, 42)
    
    try:
        # Shaped fitness usually converges much faster (30-50 generations)
        winner = p.run(eval_func, n=50)
    except KeyboardInterrupt:
        winner = p.best_genome

    with open(out_path / "winner.pkl", "wb") as f: pickle.dump(winner, f)
    with open(out_path / "stats.pkl", "wb") as f: pickle.dump(stats, f)
    plot_results(stats, out_path)
    print(f"\nTraining Complete. Results in: {out_path}")

if __name__ == "__main__":
    main()