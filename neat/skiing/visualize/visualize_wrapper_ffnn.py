import os
import sys
import neat
import pickle
import numpy as np
import gymnasium as gym

# --- 1. PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from wrapper.wrapper_ffnn import BioSkiingOCAtariWrapper
    print("BioSkiingOCAtariWrapper imported successfully.")
except ImportError:
    print("CRITICAL: wrapper/wrapper_ffnn.py not found.")
    print("   Ensure the OCAtari wrapper is saved in the wrapper folder.")
    sys.exit(1)

# --- CONFIGURATION ---

RESULTS_DIR = os.path.join(project_root, "evolution_results", "wrapper_ffnn_run")
CONFIG_PATH = os.path.join(project_root, "config", "config_wrapper_ffnn.txt")

def get_latest_winner():
    """Finds the latest .pkl file in the results directory."""
    from glob import glob
    search_pattern = os.path.join(RESULTS_DIR, "*.pkl")
    list_of_files = glob(search_pattern)
    
    if not list_of_files:
        print(f"No file found in: {RESULTS_DIR}")
        sys.exit(1)
        
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Loading file: {os.path.basename(latest_file)}")
    return latest_file

def visualize():
    # 1. Load Configuration
    if not os.path.exists(CONFIG_PATH):
        print(f"Config not found: {CONFIG_PATH}")
        return

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)
    
    if config.genome_config.num_inputs != 9:
        print(f"WARNING: Config specifies {config.genome_config.num_inputs} inputs.")
        print("   OCAtari wrapper requires 9.")
        print("   Ensure correct config usage.")

    # 2. Load Genome
    winner_path = get_latest_winner()
    with open(winner_path, "rb") as f:
        winner = pickle.load(f)

    if isinstance(winner, tuple):
        winner = winner[0]

    print(f"Registered fitness: {winner.fitness}")

    # 3. Create Neural Network
    net = neat.nn.RecurrentNetwork.create(winner, config)

    # 4. Start Environment (render_mode="human")
    print("\nStarting OCAtari (Skiing-v5)...")
    try:
        env = BioSkiingOCAtariWrapper(render_mode="human")
    except Exception as e:
        print(f"Error starting OCAtari: {e}")
        print("   Ensure installation: pip install ocatari[all]")
        return

    observation, info = env.reset(seed=14)
    done = False
    true_game_score = 0.0
    total_reward = 0.0
    steps = 0
    
    print("\n--- STARTING RUN (Ctrl+C to stop) ---")
    print(f"   Network Inputs: {len(observation)} (Expected: 9)")

    try:
        while not done:
            inputs = observation
            
            # Activate Network
            output = net.activate(inputs)
            action = np.argmax(output) 
            
            # Environment Step
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            done = terminated or truncated
            steps += 1

            if 'native_reward' in info:
                true_game_score += info['native_reward']
            else:
                pass
            
            # Debug info
            if steps % 60 == 0:
                target_status = "SEARCHING..."
                if inputs[5] > 0.5:
                    target_status = f"TARGET LOCKED (Dist: {inputs[4]:.2f})"
                print(f"Step {steps} | Reward: {total_reward:.1f} | {target_status}")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()
        print(f"\nGame finished.")
        print(f"   Total Score (Fitness): {total_reward:.2f}")
        print(f"🕹️  True Game Score (Atari):     {true_game_score:.2f}")

if __name__ == "__main__":
    visualize()