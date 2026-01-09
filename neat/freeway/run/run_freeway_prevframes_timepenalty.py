import argparse
import pickle
import sys
import numpy as np
import neat
import gymnasium as gym
from pathlib import Path
from collections import deque

# --- CONFIGURAZIONE PERCORSI ---
CURRENT_DIR = Path(__file__).resolve().parent
FREEWAY_DIR = CURRENT_DIR.parent
if str(FREEWAY_DIR) not in sys.path:
    sys.path.insert(0, str(FREEWAY_DIR))

# Setup ALE Atari
try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass

def reset_env(env, seed=None):
    obs, info = env.reset(seed=seed)
    return obs

def step_env(env, action):
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    return obs, reward, done

def get_action_mapping(env):
    try:
        meanings = env.unwrapped.get_action_meanings()
        return [meanings.index("NOOP"), meanings.index("UP"), meanings.index("DOWN")]
    except:
        return [0, 1, 2]

def eval_genomes_factory(env_id, max_steps, episodes_per_genome, seed_base, stack_frames):
    # --- PARAMETRI FITNESS (Anti-Kamikaze) ---
    SCORE_BONUS = 100.0      # Bonus enorme per ogni punto vero
    PROGRESS_WEIGHT = 20.0   # Da 0 a 20 punti per la salita (0.0 a 1.0)
    COLLISION_PENALTY = 2.0  # MALUS PESANTE: ogni incidente "brucia" molto progresso
    TIME_PENALTY_RATE = 0.01 # Toglie 0.01 ad ogni singolo frame (incoraggia velocità pulita)

    def eval_genomes(genomes, config):
        from wrapper.freeway_wrapper_prevframes import FreewayRAM11Wrapper
        
        raw_env = gym.make(env_id, obs_type="ram")
        env = FreewayRAM11Wrapper(raw_env, normalize=True, mirror_last_5=True)
        action_map = get_action_mapping(env)

        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            fitnesses = []

            for ep in range(episodes_per_genome):
                obs = reset_env(env, seed=seed_base + int(genome_id) + ep)
                
                # Setup Buffer per Stacking
                k = max(1, int(stack_frames))
                buf = deque([obs.copy() for _ in range(k)], maxlen=k)
                
                total_reward = 0.0
                collisions = 0
                max_y_reached = 0.0
                prev_y = obs[0] # Y normalizzata dal wrapper
                
                step_count = 0

                for t in range(max_steps):
                    # Concatenazione frame (44 input se stack=4)
                    stacked_inp = np.concatenate(list(buf), axis=0)
                    out = net.activate(stacked_inp)
                    
                    action = action_map[np.argmax(out)]
                    obs, reward, done = step_env(env, action)
                    
                    total_reward += float(reward)
                    buf.append(obs.copy())
                    
                    curr_y = obs[0]
                    
                    # Logica Progresso Massimo
                    if curr_y > max_y_reached:
                        max_y_reached = curr_y
                    
                    # Logica Collisione (se la Y scende improvvisamente)
                    if curr_y < prev_y - 0.05: # Soglia di sicurezza
                        collisions += 1
                    
                    prev_y = curr_y
                    step_count += 1

                    if done:
                        break

                # --- CALCOLO FITNESS FINALE ---
                # Premia i punti, premia la salita, punisce i botti e il tempo perso
                fitness = (total_reward * SCORE_BONUS) + \
                          (max_y_reached * PROGRESS_WEIGHT) - \
                          (collisions * COLLISION_PENALTY) - \
                          (step_count * TIME_PENALTY_RATE)
                
                fitnesses.append(max(0.0, fitness)) # Evitiamo fitness negative estreme

            genome.fitness = float(np.mean(fitnesses))

        env.close()

    return eval_genomes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(FREEWAY_DIR / "config" / "neat_freeway_config.txt"))
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--stack-frames", type=int, default=4)
    args = parser.parse_args()

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, args.config)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    
    eval_genomes = eval_genomes_factory(
        env_id="ALE/Freeway-v5",
        max_steps=1500,
        episodes_per_genome=1,
        seed_base=42,
        stack_frames=args.stack_frames
    )

    winner = p.run(eval_genomes, n=args.generations)
    
    # Salvataggio
    outdir = FREEWAY_DIR / "results" / "neat_freeway_prevframes_timepenalty"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "winner_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

if __name__ == "__main__":
    main()