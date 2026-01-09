import pickle
import sys
import time
import numpy as np
import neat
import gymnasium as gym
from pathlib import Path

# --- FIX PER NAMESPACE ALE ---
try:
    import ale_py
    # Gymnasium >= 0.29 richiede questa registrazione esplicita
    gym.register_envs(ale_py) 
except ImportError:
    print("ERRORE: ale-py non trovato. Esegui: pip install ale-py")
except Exception:
    pass
# -----------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
FREEWAY_DIR = SCRIPT_DIR.parent
RESULTS_DIR = FREEWAY_DIR / "results"

# ... resto del codice (choose_run_folder, run_visualize) ...

def choose_run_folder():
    """
    Scans the results directory and lets the user select a folder via index.
    """
    if not RESULTS_DIR.exists():
        print(f"ERROR: Results directory not found at {RESULTS_DIR}")
        sys.exit(1)

    # Filter only directories (runs)
    folders = sorted([f for f in RESULTS_DIR.iterdir() if f.is_dir()], reverse=True)
    
    if not folders:
        print(f"No result folders found in {RESULTS_DIR}")
        sys.exit(1)

    print("\n--- AVAILABLE TRAINING RUNS ---")
    for i, folder in enumerate(folders):
        print(f"[{i}] {folder.name}")

    while True:
        try:
            choice = input("\nSelect the run index (e.g., 0): ").strip()
            # Default to index 0 if Enter is pressed
            if choice == "":
                idx = 0
            else:
                idx = int(choice)
            
            if 0 <= idx < len(folders):
                return folders[idx]
            else:
                print(f"Invalid index. Please choose between 0 and {len(folders)-1}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def run_visualize(run_path):
    """
    Loads the winner genome and plays a match in 'human' rendering mode.
    """
    run_dir = Path(run_path)
    winner_file = run_dir / "winner.pkl"
    
    if not winner_file.exists():
        print(f"Error: Winner file 'winner.pkl' not found in {run_dir}")
        return

    # Load the best genome (winner) from training
    with open(winner_file, "rb") as f:
        winner = pickle.load(f)
    
    # Path to baseline config (must have num_inputs = 128)
    config_path = str(FREEWAY_DIR / "config" / "neat_freeway_config.txt")
    
    if not Path(config_path).exists():
        print(f"ERROR: Config file not found at {config_path}")
        return

    # Load NEAT configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Initialize Atari environment with rendering enabled
    print(f"Initializing Environment: ALE/Freeway-v5")
    env = gym.make("ALE/Freeway-v5", obs_type="ram", render_mode="human")
    
    # Create the neural network from the winner genome
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    # Map Atari actions: 0=NOOP, 1=UP, 2=DOWN
    meanings = env.unwrapped.get_action_meanings()
    action_map = [meanings.index("NOOP"), meanings.index("UP"), meanings.index("DOWN")]

    print(f"\n--- VISUALIZING RUN: {run_dir.name} ---")
    print("Test Condition: Fixed Seed 42 (Stochasticity: ON)")
    
    obs, _ = env.reset(seed=42)
    done = False
    total_score = 0
    
    try:
        while not done:
            # Process raw RAM: Normalize [0, 255] to [0.0, 1.0]
            obs_norm = obs.astype(np.float32) / 255.0
            
            # Activate Neural Network
            outputs = net.activate(obs_norm)
            action = action_map[np.argmax(outputs)]
            
            # Step in environment
            obs, reward, terminated, truncated, _ = env.step(action)
            total_score += reward
            done = terminated or truncated
            
            # Sleep to make the visualization watchable (approx 60 FPS)
            time.sleep(1/60.0)
            
    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")

    print(f"Final Atari Score: {total_score}")
    env.close()

if __name__ == "__main__":
    # Automatically ask for the folder via menu
    selected_run = choose_run_folder()
    run_visualize(selected_run)