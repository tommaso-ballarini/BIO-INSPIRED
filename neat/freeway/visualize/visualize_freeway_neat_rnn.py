import sys
import time
import pickle
from pathlib import Path
from collections import deque

import numpy as np
import neat
import gymnasium as gym

# IMPORTANT: register ALE envs
try:
    import ale_py
    try:
        gym.register_envs(ale_py)
    except Exception:
        pass
except Exception:
    pass


# -------- Paths (your folder layout) --------
SCRIPT_DIR = Path(__file__).resolve().parent              # .../neat/freeway/visualize
FREEWAY_DIR = SCRIPT_DIR.parents[0]                       # .../neat/freeway
RESULTS_DIR = FREEWAY_DIR / "results"
DEFAULT_CONFIG = FREEWAY_DIR / "config" / "neat_freeway_config.txt"

if str(FREEWAY_DIR) not in sys.path:
    sys.path.insert(0, str(FREEWAY_DIR))


def list_runs(results_dir: Path):
    if not results_dir.exists():
        return []
    runs = [p for p in results_dir.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.name)
    return runs


def choose_run():
    runs = list_runs(RESULTS_DIR)
    if not runs:
        print(f"No run folders found in: {RESULTS_DIR}")
        sys.exit(1)

    print("\n--- AVAILABLE RUNS ---")
    for i, r in enumerate(runs):
        print(f"[{i}] {r.name}")

    choice = input("\nChoose run id (or press Enter for 0): ").strip()
    idx = 0
    if choice:
        if choice.isdigit() and 0 <= int(choice) < len(runs):
            idx = int(choice)
        else:
            by_name = [r for r in runs if r.name == choice]
            if by_name:
                return by_name[0]
            print("Invalid choice, using 0.")
    return runs[idx]


def load_stats_and_winner(run_dir: Path):
    stats_path = run_dir / "stats.pkl"
    winner_path = run_dir / "winner_genome.pkl"

    if not stats_path.exists():
        raise FileNotFoundError(f"Missing: {stats_path}")
    if not winner_path.exists():
        raise FileNotFoundError(f"Missing: {winner_path}")

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    with open(winner_path, "rb") as f:
        winner = pickle.load(f)

    return stats, winner


def show_top10_and_pick_genome(stats, winner):
    mg = getattr(stats, "most_fit_genomes", None)
    if not mg:
        print("No most_fit_genomes found in stats.pkl. Using winner_genome.pkl only.")
        return winner

    rows = []
    for gen_idx, g in enumerate(mg):
        fit = getattr(g, "fitness", None)
        if fit is None:
            continue
        rows.append((gen_idx, float(fit), g))

    if not rows:
        print("No fitness values found in most_fit_genomes. Using winner_genome.pkl only.")
        return winner

    rows.sort(key=lambda x: x[1], reverse=True)
    top10 = rows[:10]
    best_gen, best_fit, best_genome = top10[0]

    print("\n--- TOP 10 (best genome per generation) ---")
    for i, (gen, fit, _) in enumerate(top10):
        print(f"[{i}] Fitness: {fit:.3f} | Generation: {gen}")

    print(f"\nPress ENTER to select best: generation {best_gen} (fitness {best_fit:.3f})")
    choice = input("Choose ID (0-9) or type a generation number: ").strip()

    if not choice:
        return best_genome

    if choice.isdigit():
        c = int(choice)
        if 0 <= c < len(top10):
            return top10[c][2]

        gen_num = int(choice)
        if 0 <= gen_num < len(mg) and getattr(mg[gen_num], "fitness", None) is not None:
            return mg[gen_num]

    print("Invalid choice. Using best.")
    return best_genome


def get_action_mapping(env):
    n = int(env.action_space.n)
    if n == 3:
        return [0, 1, 2]

    meanings = env.unwrapped.get_action_meanings()

    def idx(name):
        if name not in meanings:
            raise RuntimeError(f"Action '{name}' not in meanings: {meanings}")
        return meanings.index(name)

    return [idx("NOOP"), idx("UP"), idx("DOWN")]


def obs_to_net_input(obs):
    x = np.asarray(obs, dtype=np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    return x


def make_env(env_id: str, use_wrapper: bool, render_mode: str):
    env = gym.make(env_id, obs_type="ram", render_mode=render_mode)

    if use_wrapper:
        from wrapper.freeway_wrapper import FreewaySpeedWrapper
        env = FreewaySpeedWrapper(env, normalize=True, mirror_last_5=True)

    return env


def run_live_rnn(env_id: str, config_path: Path, genome, fps: int, max_steps: int, use_wrapper: bool):
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )

    # Recurrent net (RNN)
    net = neat.nn.RecurrentNetwork.create(genome, config)

    env = make_env(env_id, use_wrapper=use_wrapper, render_mode="human")
    action_map = get_action_mapping(env)

    obs, info = env.reset()
    x = obs_to_net_input(obs)

    expected = config.genome_config.num_inputs
    if len(x) != expected:
        raise RuntimeError(
            f"Config expects {expected} inputs, got {len(x)}. "
            f"(use_wrapper={use_wrapper}). Use the matching config + flags."
        )

    total_reward = 0.0
    steps = 0

    print("\nSTART! (Ctrl+C to exit)")
    print(f"Env: {env_id} | FPS: {fps} | Max steps/episode: {max_steps} | Wrapper: {use_wrapper} | RNN: True")

    try:
        while True:
            out = net.activate(x)
            a3 = int(np.argmax(out))
            action = action_map[a3]

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1

            x = obs_to_net_input(obs)

            time.sleep(1.0 / max(1, fps))

            if terminated or truncated or steps >= max_steps:
                print(f"Episode end | Return: {total_reward:.3f} | Steps: {steps}")

                # Reset env AND reset RNN state by recreating the network
                obs, info = env.reset()
                net = neat.nn.RecurrentNetwork.create(genome, config)
                x = obs_to_net_input(obs)

                total_reward = 0.0
                steps = 0

    except KeyboardInterrupt:
        env.close()
        print("\nClosed.")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="ALE/Freeway-v5")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=5000)

    # wrapper ON by default (since your RNN config is for 11 inputs)
    parser.add_argument("--use-wrapper", action="store_true", default=True)

    # choose a run folder under results/
    parser.add_argument("--run-dir", type=str, default="", help="If set, skip menu and use this run directory")

    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else choose_run()
    print(f"\nUsing run: {run_dir}")

    stats, winner = load_stats_and_winner(run_dir)
    genome_to_play = show_top10_and_pick_genome(stats, winner)

    run_live_rnn(
        env_id=args.env_id,
        config_path=Path(args.config),
        genome=genome_to_play,
        fps=args.fps,
        max_steps=args.max_steps,
        use_wrapper=args.use_wrapper,
    )


if __name__ == "__main__":
    main()
