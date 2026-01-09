import os
import sys
import time
import pickle
from pathlib import Path

import numpy as np
import neat

import gymnasium as gym
from collections import deque



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
            # allow typing a folder name
            by_name = [r for r in runs if r.name == choice]
            if by_name:
                return by_name[0]
            print("Invalid choice, using 0.")
    return runs[idx]


def load_stats_and_winner(run_dir: Path):
    stats_path = run_dir / "stats.pkl"
    winner_path = run_dir / "winner_genome.pkl"

    # Se manca il vincitore, non possiamo fare nulla
    if not winner_path.exists():
        raise FileNotFoundError(f"Mancante file del genoma: {winner_path}")

    # Carichiamo il vincitore
    with open(winner_path, "rb") as f:
        winner = pickle.load(f)

    # Carichiamo le statistiche solo se esistono
    stats = None
    if stats_path.exists():
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
    else:
        print(f"⚠️ Nota: {stats_path.name} non trovato. Vedrai solo l'ultimo vincitore salvato.")

    return stats, winner

def show_top10_and_pick_genome(stats, winner):
    # stats.most_fit_genomes: list of best genome per generation
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

    # choose by list id
    if choice.isdigit():
        c = int(choice)
        if 0 <= c < len(top10):
            return top10[c][2]

        # choose by generation number
        gen_num = int(choice)
        if 0 <= gen_num < len(mg) and getattr(mg[gen_num], "fitness", None) is not None:
            return mg[gen_num]

    print("Invalid choice. Using best.")
    return best_genome


def get_action_mapping(env):
    """
    If env.action_space.n == 3, we can directly use 0/1/2.
    Otherwise map [NOOP, UP, DOWN] using action meanings.
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


def obs_to_net_input(obs):
    x = np.asarray(obs, dtype=np.float32)
    # RAM uint8 -> normalize
    if x.max() > 1.5:
        x = x / 255.0
    return x


def make_env(env_id: str, fps: int, use_wrapper: bool):
    # human rendering (live)
    env = gym.make(env_id, obs_type="ram", render_mode="human")

    # optional wrapper (ONLY if you trained with wrapper + num_inputs=11)
    if use_wrapper:
        # Change this import if your wrapper filename differs
        from wrapper.freeway_wrapper_prevframes import FreewayRAM11Wrapper
        env = FreewayRAM11Wrapper(env, normalize=True, mirror_last_5=True)

    return env


def run_live(env_id: str, config_path: Path, genome, fps: int, max_steps: int,
             use_wrapper: bool, stack_frames: int):
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    env = make_env(env_id, fps=fps, use_wrapper=use_wrapper)
    action_map = get_action_mapping(env)

    obs, info = env.reset()
    total_reward = 0.0
    steps = 0

    k = int(stack_frames)
    if k < 1:
        k = 1

    obs0 = obs_to_net_input(obs)
    buf = deque([obs0.copy() for _ in range(k)], maxlen=k)



    # collision heuristic (same spirit as your old script):
    # count when chicken Y decreases suddenly. Only valid for RAW RAM obs (not wrapper).
    collisions = 0
    prev_y = None
    if not use_wrapper:
        prev_y = int(obs[14])

    print("\nSTART! (Ctrl+C to exit)")
    print(f"Env: {env_id} | FPS: {fps} | Max steps/episode: {max_steps} | Wrapper: {use_wrapper}")

    try:
        while True:
            stacked_inp = np.concatenate(list(buf), axis=0)

            # Safety check: avoids wasting time with wrong config/flags
            expected = config.genome_config.num_inputs
            if len(stacked_inp) != expected:
                raise RuntimeError(
                    f"Config expects {expected} inputs, got {len(stacked_inp)} "
                    f"(wrapper={use_wrapper}, stack_frames={k}). "
                    "Use the matching config and flags."
                )

            out = net.activate(stacked_inp)

            a3 = int(np.argmax(out))           # 0/1/2 (our policy)
            action = action_map[a3]            # actual env action

            obs, reward, terminated, truncated, info = env.step(action)
            buf.append(obs_to_net_input(obs))
            total_reward += float(reward)
            steps += 1

            # live collisions
            if not use_wrapper:
                curr_y = int(obs[14])
                if prev_y is not None and curr_y < prev_y:
                    collisions += 1
                    print(f"💥 Collision! (total: {collisions})")
                prev_y = curr_y

            time.sleep(1.0 / max(1, fps))

            if terminated or truncated or steps >= max_steps:
                print(f"Episode end | Return: {total_reward:.3f} | Collisions: {collisions} | Steps: {steps}")
                obs, info = env.reset()
                total_reward = 0.0
                steps = 0
                collisions = 0
                if not use_wrapper:
                    prev_y = int(obs[14])

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
    parser.add_argument("--stack-frames", type=int, default=4,
                    help="Number of past observations to stack (1=current only).")
    parser.add_argument("--use-wrapper", action="store_true", default=True)

    # choose a run folder under results/
    parser.add_argument("--run-dir", type=str, default="", help="If set, skip menu and use this run directory")

    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else choose_run()
    print(f"\nUsing run: {run_dir}")

    stats, winner = load_stats_and_winner(run_dir)
    
    # Se non abbiamo le statistiche, non possiamo mostrare la Top 10
    if stats is not None:
        genome_to_play = show_top10_and_pick_genome(stats, winner)
    else:
        genome_to_play = winner

    run_live(
        env_id=args.env_id,
        config_path=Path(args.config),
        genome=genome_to_play,
        fps=args.fps,
        max_steps=args.max_steps,
        use_wrapper=args.use_wrapper,
        stack_frames=args.stack_frames
    )


if __name__ == "__main__":
    main()

