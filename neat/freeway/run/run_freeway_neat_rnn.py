# script to run freeway with NEAT RNN genomes
# IMPORTANT: requires modifying the config file to set num_inputs=11 and feed_forward = False

import argparse
import pickle
import sys
from pathlib import Path
from collections import deque

import numpy as np
import neat

# Gymnasium + ALE registration
import gymnasium as gym
try:
    import ale_py
    try:
        gym.register_envs(ale_py)
    except Exception:
        pass
except Exception:
    pass


# -------- Paths (your folder layout) --------
FREEWAY_DIR = Path(__file__).resolve().parents[1]  # .../neat/freeway
if str(FREEWAY_DIR) not in sys.path:
    sys.path.insert(0, str(FREEWAY_DIR))

DEFAULT_MAX_STEPS = 1000

# Fitness shaping (tune later if needed)
COLLISION_PENALTY_ALPHA = 0.2
PROGRESS_REWARD_BETA = 0.0
NO_PROGRESS_PATIENCE = 950      # steps without improving before ending episode
MIN_PROGRESS_DELTA = 0.002      # wrapper y is normalized (0..1); this is a small improvement threshold
ACTION_REPEAT = 2

def reset_env(env, seed=None):
    obs, info = env.reset(seed=seed)
    return obs


def step_env(env, action):
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    return obs, reward, done


def obs_to_net_input(obs):
    """Il wrapper restituisce già float32 normalizzati."""
    return np.asarray(obs, dtype=np.float32)


def make_env(env_id: str, seed: int | None, use_wrapper: bool):
    env = gym.make(env_id, obs_type="ram")
    
    if use_wrapper:
        # Cambia qui il nome della classe con quella a 22 input
        from wrapper.freeway_wrapper import FreewaySpeedWrapper
        env = FreewaySpeedWrapper(env, normalize=True, mirror_last_5=True)

    # Nota: il reset va fatto DOPO aver applicato il wrapper 
    # per inizializzare prev_cars_x correttamente
    obs, info = env.reset(seed=seed)
    return env


def get_action_mapping(env):
    """
    Your Freeway env is often Discrete(3). If not, map by meanings.
    Returns env action indices for [NOOP, UP, DOWN].
    """
    n = int(env.action_space.n)
    if n == 3:
        return [0, 1, 2]

    meanings = env.unwrapped.get_action_meanings()

    def idx(name):
        if name not in meanings:
            raise RuntimeError(f"Action '{name}' not in meanings: {meanings}")
        return meanings.index(name)

    return [idx("NOOP"), idx("UP"), idx("DOWN")]


def compute_collision_and_progress(prev_y: float, curr_y: float, use_wrapper: bool):
    """
    We use:
    - progress: upward movement (y decreases) => max(0, prev_y - curr_y)
    - collision proxy: sudden downward jump (y increases a lot)
      * wrapper y is normalized -> small threshold
      * raw RAM y -> bigger threshold
    """
    dy_up = prev_y - curr_y
    progress = dy_up if dy_up > 0 else 0.0

    if use_wrapper:
        collided = (curr_y - prev_y) > 0.03
    else:
        collided = (curr_y - prev_y) > 8.0

    return int(collided), float(progress)


def eval_genomes_factory(
    env_id: str,
    max_steps: int,
    episodes_per_genome: int,
    seed_base: int,
    use_wrapper: bool,
):
    # Parametri bilanciati per valori di fitness "umani"
    COLLISION_PENALTY = 0.5    # Sottrae mezzo punto per ogni incidente
    PROGRESS_WEIGHT = 10.0      # L'intera salita (da 0 a 1) vale 10 punti
    
    def eval_genomes(genomes, config):
        env = make_env(env_id=env_id, seed=seed_base, use_wrapper=use_wrapper)
        action_map = get_action_mapping(env)

        for genome_id, genome in genomes:
            fitnesses = []

            for ep in range(episodes_per_genome):
                net = neat.nn.RecurrentNetwork.create(genome, config)
                obs = reset_env(env, seed=seed_base + int(genome_id) + 1000 * ep)
                
                x = obs_to_net_input(obs)
                
                prev_y = float(x[0])
                was_colliding_prev = False
                
                total_game_reward = 0.0
                collision_count = 0
                total_progress = 0.0
                
                best_y = prev_y
                no_progress_steps = 0
                chosen_a3 = 0

                for t in range(max_steps):
                    out = net.activate(x)

                    if (t % ACTION_REPEAT) == 0:
                        chosen_a3 = int(np.argmax(out))

                    action = action_map[chosen_a3]
                    obs, reward, done = step_env(env, action)
                    
                    # Reward del gioco (+1 ogni volta che attraversa)
                    total_game_reward += float(reward)

                    x = obs_to_net_input(obs)
                    curr_y = float(x[0])  # Y normalizzata (0 basso, 1 alto)
                    
                    # --- LOGICA COLLISIONE (Edge Trigger) ---
                    is_colliding_now = x[1] > 0
                    if is_colliding_now and not was_colliding_prev:
                        collision_count += 1
                    was_colliding_prev = is_colliding_now

                    # --- LOGICA PROGRESSO ---
                    # In FreewayWrapper, y=0 è l'inizio, y=1 è l'arrivo.
                    # Quindi progresso = curr_y - prev_y
                    dy = curr_y - prev_y
                    if dy > 0:
                        total_progress += dy
                    
                    prev_y = curr_y

                    # Early stop se il pollo è bloccato
                    if curr_y > best_y + 0.002: # Se è salito
                        best_y = curr_y
                        no_progress_steps = 0
                    else:
                        no_progress_steps += 1

                    if done or (no_progress_steps >= NO_PROGRESS_PATIENCE):
                        break

                # Calcolo Fitness finale:
                # Esempio: 0 (punti) - 0.5 (1 colpo) + 5.0 (metà strada fatta) = 4.5
                fitness = (total_game_reward * 20.0) + \
                          (total_progress * PROGRESS_WEIGHT) - \
                          (collision_count * COLLISION_PENALTY)
                
                fitnesses.append(fitness)

            genome.fitness = float(np.mean(fitnesses))

        env.close()

    return eval_genomes

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=str(FREEWAY_DIR / "config" / "neat_freeway_config.txt"),
        help="NEAT config for RNN run (should have feed_forward=False and num_inputs=11).",
    )
    parser.add_argument("--env-id", type=str, default="ALE/Freeway-v5")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--episodes-per-genome", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)

    # Keep wrapper ON by default for RNN experiment
    parser.add_argument("--use-wrapper", action="store_true", default=True)

    parser.add_argument(
        "--outdir",
        type=str,
        default=str(FREEWAY_DIR / "results" / "neat_freeway_rnn"),
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )

    eval_genomes = eval_genomes_factory(
        env_id=args.env_id,
        max_steps=args.max_steps,
        episodes_per_genome=args.episodes_per_genome,
        seed_base=args.seed,
        use_wrapper=args.use_wrapper,
    )

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # checkpoint every 5 generations
    p.add_reporter(
        neat.Checkpointer(
            generation_interval=5,
            filename_prefix=str(outdir / "checkpoint-gen-"),
        )
    )

    winner = p.run(eval_genomes, n=args.generations)

    with open(outdir / "winner_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    with open(outdir / "stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    print("\nDone.")
    print(f"Winner saved: {outdir / 'winner_genome.pkl'}")
    print(f"Stats saved:  {outdir / 'stats.pkl'}")
    print(f"Used alpha={COLLISION_PENALTY_ALPHA}, beta={PROGRESS_REWARD_BETA}")


if __name__ == "__main__":
    main()

