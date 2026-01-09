## Script to run NEAT on Freeway with previous frame stacking WITH the wrapper.
# IMPORTANT: requires modifying the config file to set num_inputs=44 (11*4).

import argparse
import pickle
from pathlib import Path

import numpy as np
import neat

from pathlib import Path
import sys
from collections import deque

FREEWAY_DIR = Path(__file__).resolve().parents[1]  # .../neat/freeway
# Optional: allow imports from .../neat/freeway/*
sys.path.insert(0, str(FREEWAY_DIR))
DEFAULT_MAX_STEPS = 1000
COLLISION_PENALTY_ALPHA = 0.3
PROGRESS_REWARD_BETA = 0.02

# gymnasium preferred, fallback to gym
try:
    import gymnasium as gym
    IS_GYMNASIUM = True

    import ale_py  # <-- ADD THIS (it registers ALE environments)
    try:
        gym.register_envs(ale_py)  # optional but harmless
    except Exception:
        pass

except ImportError:
    import gym
    IS_GYMNASIUM = False


def reset_env(env, seed=None):
    if IS_GYMNASIUM:
        obs, info = env.reset(seed=seed)
        return obs
    else:
        if seed is not None:
            try:
                env.seed(seed)
            except Exception:
                pass
        return env.reset()


def step_env(env, action):
    if IS_GYMNASIUM:
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        return obs, reward, done
    else:
        obs, reward, done, info = env.step(action)
        return obs, reward, done


def get_noop_up_down_mapping(env):
    """
    Returns the actual env action indices for [NOOP, UP, DOWN]
    using ALE action meanings (works even if env has 18 actions).
    """
    try:
        meanings = env.unwrapped.get_action_meanings()
    except Exception as e:
        raise RuntimeError(
            "Could not read action meanings from env. "
            "If this is not an ALE Atari env, define the mapping manually."
        ) from e

    def idx(name):
        if name not in meanings:
            raise RuntimeError(f"Action '{name}' not in meanings: {meanings}")
        return meanings.index(name)

    return [idx("NOOP"), idx("UP"), idx("DOWN")]


def make_env(env_id: str, seed: int | None, use_wrapper: bool):
    # Always request RAM observations directly
    env = gym.make(env_id, obs_type="ram")

    obs = reset_env(env, seed=seed)
    if np.asarray(obs).shape != (128,):
        env.close()
        raise RuntimeError(
            f"Expected RAM obs shape (128,), got {np.asarray(obs).shape}. "
            "Check that obs_type='ram' is supported."
        )

    if use_wrapper:
        from wrapper.freeway_wrapper import FreewayRAM11Wrapper
        env = FreewayRAM11Wrapper(env, normalize=True, mirror_last_5=True)

    return env



def obs_to_net_input(obs):
    """
    Convert observation to float32 for NEAT.
    - If obs is RAM uint8 [0..255], normalize to [0..1]
    - If wrapper is used, it already outputs float32 [0..1]
    """
    x = np.asarray(obs, dtype=np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    return x


def eval_genomes_factory(env_id: str, max_steps: int, episodes_per_genome: int,
                         seed_base: int, use_wrapper: bool, stack_frames: int):
    def eval_genomes(genomes, config):
        env = make_env(env_id=env_id, seed=seed_base, use_wrapper=use_wrapper)
        mapping = get_noop_up_down_mapping(env)  # [NOOP, UP, DOWN] real indices

        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            returns = []
            for ep in range(episodes_per_genome):
                obs = reset_env(env, seed=seed_base + int(genome_id) + 1000 * ep)

                # --- frame stack buffer ---
                k = int(stack_frames)
                if k < 1:
                    k = 1

                obs0 = obs_to_net_input(obs)
                buf = deque([obs0.copy() for _ in range(k)], maxlen=k)

                total_reward = 0.0
                collisions = 0
                progress = 0.0

                # chicken y init
                # NOTE: with wrapper obs0[0] is normalized chicken y in [0,1]
                # without wrapper obs[14] is raw y in [0,255-ish]
                if use_wrapper:
                    prev_y = float(obs0[0])
                else:
                    prev_y = float(obs[14])

                for _ in range(max_steps):
                    stacked_inp = np.concatenate(list(buf), axis=0)
                    out = net.activate(stacked_inp)

                    a3 = int(np.argmax(out))
                    env_action = mapping[a3]

                    obs, reward, done = step_env(env, env_action)
                    total_reward += float(reward)

                    obs1 = obs_to_net_input(obs)
                    buf.append(obs1)

                    # read current y
                    if use_wrapper:
                        curr_y = float(obs1[0])
                        # going UP usually makes y smaller -> progress = prev_y - curr_y (only if positive)
                        dy = prev_y - curr_y
                    else:
                        curr_y = float(obs[14])
                        dy = prev_y - curr_y

                    # progress: accumulate only upward movement
                    if dy > 0:
                        progress += dy

                    # collision proxy: big sudden downward jump (y increases)
                    # Your previous heuristic was "curr_y < prev_y". That can also happen with upward movement.
                    # For collisions, what we really want is "sudden move down": curr_y > prev_y by a lot.
                    # We'll use a threshold.
                    #
                    # Wrapper y is normalized, so threshold should be small. Raw RAM y threshold should be larger.
                    if use_wrapper:
                        if (curr_y - prev_y) > 0.05:   # tune if needed
                            collisions += 1
                    else:
                        if (curr_y - prev_y) > 8.0:    # tune if needed
                            collisions += 1

                    prev_y = curr_y

                    if done:
                        break

                fitness = total_reward - COLLISION_PENALTY_ALPHA * collisions + PROGRESS_REWARD_BETA * progress
                returns.append(fitness)




            genome.fitness = float(np.mean(returns))

        env.close()

    return eval_genomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--config",
    type=str,
    default=str(FREEWAY_DIR / "config" / "neat_freeway_congif.txt"),
)
    parser.add_argument("--env-id", type=str, default="ALE/Freeway-v5")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--episodes-per-genome", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stack-frames", type=int, default=4,
                    help="Number of past observations to stack (default: 4).")

    

    # OFF by default. Later you can enable it and also change num_inputs=11 in the config.
    parser.add_argument("--use-wrapper", action="store_true", default=True,
                        help="Use external RAM->11 wrapper from wrappers/freeway_ram11_wrapper.py")

    parser.add_argument(
    "--outdir",
    type=str,
    default=str(FREEWAY_DIR / "results" / "neat_freeway_prevframes4"),
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
        stack_frames=args.stack_frames
    )

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # checkpoint every 5 generations
    p.add_reporter(neat.Checkpointer(generation_interval=5,
                                    filename_prefix=str(outdir / "checkpoint-gen-")))

    winner = p.run(eval_genomes, n=args.generations)

    with open(outdir / "winner_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    with open(outdir / "stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    print("\nDone.")
    print(f"Winner saved: {outdir / 'winner_genome.pkl'}")
    print(f"Stats saved:  {outdir / 'stats.pkl'}")


if __name__ == "__main__":
    main()
