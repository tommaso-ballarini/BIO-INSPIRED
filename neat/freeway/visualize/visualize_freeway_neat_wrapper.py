import pickle
import sys
import time
import numpy as np
import neat
import gymnasium as gym
from pathlib import Path

# --- ALE FIX ---
try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass

# --- ROBUST PATH CONFIGURATION ---
# This gets the absolute path of the directory containing THIS script (visualize)
SCRIPT_DIR = Path(__file__).resolve().parent 
# This gets the project root (neat/freeway)
PROJECT_ROOT = SCRIPT_DIR.parent

# Add PROJECT_ROOT to sys.path so 'import wrapper' works correctly
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now we can import the wrapper safely
try:
    from wrapper.freeway_wrapper import FreewaySpeedWrapper
    print("✅ Wrapper successfully imported.")
except ImportError as e:
    print(f"❌ ERROR: Could not find 'wrapper/freeway_wrapper.py' in {PROJECT_ROOT}")
    print(f"Debug info: sys.path is {sys.path}")
    sys.exit(1)
FREEWAY_DIR = SCRIPT_DIR.parent
RESULTS_DIR = FREEWAY_DIR / "results"

def choose_run_folder():
    """ Provides a menu to select the training run from the results directory. """
    if not RESULTS_DIR.exists():
        print(f"ERROR: Results directory {RESULTS_DIR} not found.")
        sys.exit(1)

    folders = sorted([f for f in RESULTS_DIR.iterdir() if f.is_dir()], reverse=True)
    if not folders:
        print("No result folders found.")
        sys.exit(1)

    print("\n--- SELECT SPEED WRAPPER RUN ---")
    for i, folder in enumerate(folders):
        print(f"[{i}] {folder.name}")

    idx = input("\nIndex (default 0): ").strip()
    idx = int(idx) if idx.isdigit() else 0
    return folders[min(idx, len(folders)-1)]

def run_visualize(run_path):
    """ Plays the game using the Speed Wrapper and the winner genome. """
    from wrapper.freeway_wrapper import FreewaySpeedWrapper
    
    run_dir = Path(run_path)
    with open(run_dir / "winner.pkl", "rb") as f:
        winner = pickle.load(f)
    
    config_path = str(FREEWAY_DIR / "config" / "neat_freeway_config.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Human rendering for live visualization
    env = gym.make("ALE/Freeway-v5", obs_type="ram", render_mode="human")
    # Wrap the environment exactly as in training
    env = FreewaySpeedWrapper(env, normalize=True, mirror_last_5=True)
    
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    meanings = env.unwrapped.get_action_meanings()
    action_map = [meanings.index("NOOP"), meanings.index("UP"), meanings.index("DOWN")]

    print(f"\n--- VISUALIZING WINNER: {run_dir.name} ---")
    obs, _ = env.reset(seed=42) # Fixed seed for testing
    done = False
    total_score = 0
    
    try:
        while not done:
            action_idx = np.argmax(net.activate(obs))
            action = action_map[action_idx]
            
            obs, reward, terminated, truncated, _ = env.step(action)
            total_score += reward
            done = terminated or truncated
            time.sleep(1/60.0) # Approx 60 FPS
    except KeyboardInterrupt:
        pass

    print(f"Final Score: {total_score}")
    env.close()

if __name__ == "__main__":
    selected_run = choose_run_folder()
    run_visualize(selected_run)