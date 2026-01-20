import os
import sys
import importlib.util
import time
import pathlib
import numpy as np
from tqdm import tqdm
from ocatari.core import OCAtari

# --- CONFIGURATION ---
AGENT_PATH = r"OPEN_EVOLVE\space_invaders\results\run_si_20260109_153440_best\interesting_agents\agent_1599_pts_1767819509889.py" # PASTE HERE YOUR EXACT AGENT PATH   


VISUALIZATION_SEED = 42
TEST_SEEDS_RANGE = range(0, 100) # Test on 100 different games
MAX_STEPS = 5000 

# --- PATH SETUP ---
BASE_DIR = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(BASE_DIR))

try:
    from wrapper.si_wrapper import SpaceInvadersEgocentricWrapper
except ImportError:
    print("❌ Error: Cannot import 'wrapper.si_wrapper'.")
    sys.exit(1)

def load_agent(path_str):
    path = pathlib.Path(path_str)
    if not path.exists():
        print(f"❌ Error: Agent file not found at: {path}")
        print("Check if AGENT_PATH is correct.")
        sys.exit(1)
    
    spec = importlib.util.spec_from_file_location("best_agent", str(path))
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    
    if not hasattr(agent_module, 'get_action'):
        print("❌ Agent is missing 'get_action(observation)' function!")
        sys.exit(1)

    return agent_module.get_action

def run_simulation(action_func, seed, render=False):
    """Runs a Space Invaders game and returns the score."""
    render_mode = "human" if render else None
    
    try:
        # Space Invaders Env Setup
        env = OCAtari("ALE/SpaceInvaders-v5", mode="ram", hud=False, render_mode=render_mode)
        env = SpaceInvadersEgocentricWrapper(env, skip=4)
        
        obs, info = env.reset(seed=seed)
        
        total_score = 0.0
        steps = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and steps < MAX_STEPS:
            try:
                action = int(action_func(obs))
            except Exception:
                break
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_score += reward
            steps += 1
            
            if render: 
                time.sleep(0.01) # Playback speed
                
        env.close()
        return total_score
        
    except Exception as e:
        print(f"Critical error (Seed {seed}): {e}")
        return -9999

def main():
    print(f"\n👾 TESTING AGENT (Hardcoded): {os.path.basename(AGENT_PATH)} 👾")
    print("="*60)
    
    # 1. Load Agent
    get_action = load_agent(AGENT_PATH)
    print("✅ Agent loaded successfully.")
    
    # 2. Initial Preview (Visual)
    print(f"🎥  PREVIEW (Seed {VISUALIZATION_SEED})...")
    run_simulation(get_action, seed=VISUALIZATION_SEED, render=True)
    print("-" * 60)
    
    # 3. Benchmark on 100 Seeds (Fast, no graphics)
    print(f"\n📊  BENCHMARK ON {len(TEST_SEEDS_RANGE)} SEEDS (Generalization)...")
    
    results = []
    
    for seed in tqdm(TEST_SEEDS_RANGE, desc="Simulating", unit="game"):
        score = run_simulation(get_action, seed=seed, render=False)
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
        run_simulation(get_action, seed=best_run['seed'], render=True)

if __name__ == "__main__":
    main()