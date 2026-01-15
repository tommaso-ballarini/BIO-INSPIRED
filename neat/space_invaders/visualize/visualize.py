import sys
import os
import time
import pickle
import pathlib
import numpy as np
import neat
from tqdm import tqdm

# --- 1. USER CONFIGURATION (EDIT THIS) ---
# Paste the path to your .pkl file here (Winner or Top3 list)
AGENT_PKL_PATH = r"neat\space_invaders\results\winner_rnn_rank_1.pkl" 

# Paste the path to your original NEAT config file 
NEAT_CONFIG_PATH = r"neat\space_invaders\config\config_si_ego.txt"

VISUALIZATION_SEED = 42      # Seed for initial preview
TEST_SEEDS_RANGE = range(0, 100) # 100 benchmark seeds
MAX_STEPS = 5000             # Safety limit

# --- 2. IMPORT & WRAPPER HANDLING ---
# Resolve paths to import custom wrapper
current_file = pathlib.Path(__file__).resolve()
project_root = current_file.parent.parent # Move up levels (adjust if script is in root)

# Add root to sys.path to find 'wrapper' and 'ocatari'
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
sys.path.append(str(current_file.parent))

try:
    from ocatari.core import OCAtari
    from wrapper.wrapper_si_ego import SpaceInvadersEgocentricWrapper
except ImportError:
    print("❌ Import Error: Wrapper or OCAtari not found.")
    print(f"Ensure the script can see the 'wrapper' folder.")
    print(f"Current Path: {sys.path}")
    sys.exit(1)

# --- 3. NEAT LOADING FUNCTIONS ---
def load_neat_agent(pkl_path, config_path):
    """Loads genome and creates the Recurrent Neural Network."""
    
    # File Checks
    if not os.path.exists(pkl_path):
        print(f"❌ Error: PKL file not found: {pkl_path}")
        sys.exit(1)
    if not os.path.exists(config_path):
        print(f"❌ Error: NEAT Config not found: {config_path}")
        sys.exit(1)

    # 1. Load Config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # 2. Load Genome (Handles single object or Top3 list)
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    genome = None
    if isinstance(data, list):
        genome = data[0] # If list (e.g., top3), take the first (best)
    else:
        genome = data    # Otherwise use the single genome

    print(f"🧬 Genome ID: {genome.key} | Original Fitness: {genome.fitness}")

    # 3. Create Recurrent Network
    # IMPORTANT: Must match training type (RecurrentNetwork)
    net = neat.nn.RecurrentNetwork.create(genome, config)
    return net

def run_simulation(net, seed, render=False):
    """Runs a game using the NEAT network."""
    render_mode = "human" if render else None
    
    try:
        # Setup Environment
        try:
            env = OCAtari("ALE/SpaceInvaders-v5", mode="ram", hud=False, render_mode=render_mode)
        except:
            env = OCAtari("SpaceInvaders-v4", mode="ram", hud=False, render_mode=render_mode)
            
        env = SpaceInvadersEgocentricWrapper(env, skip=4)
        
        obs, info = env.reset(seed=seed)
        
        total_score = 0.0
        steps = 0
        terminated = False
        truncated = False
        
        # Reset RNN internal state for new episode
        net.reset()
        
        while not (terminated or truncated) and steps < MAX_STEPS:
            # --- NEAT LOGIC ---
            # Network receives observation and outputs values
            outputs = net.activate(obs)
            # Choose action with highest value (Argmax)
            action = np.argmax(outputs)
            
            # Physics Step
            obs, reward, terminated, truncated, info = env.step(action)
            total_score += reward
            steps += 1
            
            if render:
                time.sleep(0.01) # Playback speed
        
        env.close()
        return total_score

    except Exception as e:
        print(f"Critical simulation error (Seed {seed}): {e}")
        return -9999

# --- 4. MAIN LOOP ---
def main():
    print(f"\n🧠 NEAT BENCHMARK: {os.path.basename(AGENT_PKL_PATH)} 🧠")
    print("="*60)
    
    # 1. Load Agent
    net = load_neat_agent(AGENT_PKL_PATH, NEAT_CONFIG_PATH)
    print("✅ Neural Network reconstructed successfully.")
    
    # 2. Initial Preview
    print(f"🎥  PREVIEW (Seed {VISUALIZATION_SEED})...")
    run_simulation(net, seed=VISUALIZATION_SEED, render=True)
    print("-" * 60)
    
    # 3. Benchmark on 100 Seeds
    print(f"\n📊  BENCHMARK ON {len(TEST_SEEDS_RANGE)} SEEDS (Generalization)...")
    
    results = []
    
    # Use tqdm for green progress bar
    for seed in tqdm(TEST_SEEDS_RANGE, desc="Simulating", unit="game", colour="green"):
        score = run_simulation(net, seed=seed, render=False)
        results.append({'seed': seed, 'score': score})
        
    # 4. Statistical Analysis
    scores = [r['score'] for r in results]
    avg_score = np.mean(scores)
    std_dev = np.std(scores)
    
    best_run = max(results, key=lambda x: x['score'])
    worst_run = min(results, key=lambda x: x['score'])

    print("\n" + "="*60)
    print("📈  FINAL RESULTS")
    print("="*60)
    print(f"OVERALL AVERAGE:  {avg_score:.2f} (± {std_dev:.2f})")
    print("-" * 30)
    print(f"🏆 BEST RUN:  Seed {best_run['seed']} -> Score {best_run['score']:.0f}")
    print(f"💩 WORST RUN: Seed {worst_run['seed']} -> Score {worst_run['score']:.0f}")
    print("="*60)
    
    # 5. Replay Best
    choice = input(f"\nWatch the BEST run (Seed {best_run['seed']})? [y/n]: ")
    if choice.lower() == 'y':
        print(f"Replaying Seed {best_run['seed']}...")
        run_simulation(net, seed=best_run['seed'], render=True)

if __name__ == "__main__":
    main()