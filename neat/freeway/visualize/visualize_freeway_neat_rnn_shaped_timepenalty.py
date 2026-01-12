import pickle
import sys
import time
import numpy as np
import neat
import gymnasium as gym
from pathlib import Path

# --- MODULE PATH FIX ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import ale_py
    gym.register_envs(ale_py)
except Exception: pass

RESULTS_DIR = PROJECT_ROOT / "results"

def choose_run():
    folders = sorted([f for f in RESULTS_DIR.iterdir() if f.is_dir()], reverse=True)
    print("\n--- SELECT RNN SHAPED RUN ---")
    for i, f in enumerate(folders): print(f"[{i}] {f.name}")
    idx = input("\nIndex (default 0): ").strip()
    return folders[int(idx) if idx.isdigit() else 0]

def run_visualize(run_path):
    from wrapper.freeway_wrapper import FreewaySpeedWrapper
    
    run_dir = Path(run_path)
    winner_file = run_dir / "winner.pkl"
    if not winner_file.exists():
        print(f"Error: {winner_file} not found.")
        return

    with open(winner_file, "rb") as f: winner = pickle.load(f)
    
    config_path = str(PROJECT_ROOT / "config" / "neat_freeway_config.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    env = gym.make("ALE/Freeway-v5", obs_type="ram", render_mode="human")
    env = FreewaySpeedWrapper(env, normalize=True, mirror_last_5=True)
    
    # RecurrentNetwork instead of FeedForwardNetwork
    net = neat.nn.RecurrentNetwork.create(winner, config)
    meanings = env.unwrapped.get_action_meanings()
    action_map = [meanings.index("NOOP"), meanings.index("UP"), meanings.index("DOWN")]

    print(f"\n--- VISUALIZING RNN WINNER: {run_dir.name} ---")
    obs, _ = env.reset(seed=42)
    done = False
    
    while not done:
        action = action_map[np.argmax(net.activate(obs))]
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        time.sleep(1/60.0)

    env.close()

if __name__ == "__main__":
    selected = choose_run()
    run_visualize(selected)