import os
import sys
import importlib.util
import time
import pathlib
import gymnasium as gym
import numpy as np
import ale_py
from tqdm import tqdm

# --- CONFIGURATION ---
AGENT_PATH = r"OPEN_EVOLVE\freeway\results\best_agent.py.py" # PASTE HERE YOUR EXACT AGENT PATH 

VISUALIZATION_SEED = 42
TEST_SEEDS_RANGE = range(0, 100) # Test on 100 different games
MAX_STEPS = 2100  # Freeway has a natural limit of ~2048 frames, keeping buffer

# --- PATH SETUP ---
current_dir = pathlib.Path(__file__).parent.resolve()
experiment_root = current_dir.parent 
wrapper_dir = experiment_root / 'wrapper'

sys.path.append(str(wrapper_dir))
sys.path.append(str(current_dir))

# Register Gym ALE
gym.register_envs(ale_py)

try:
    from freeway_wrapper import FreewaySpeedWrapper as FreewayOCAtariWrapper
except ImportError:
    # Fallback se la struttura delle cartelle è diversa
    try:
        sys.path.append(os.path.join(current_dir, 'wrapper'))
        from wrapper.freeway_wrapper import FreewaySpeedWrapper as FreewayOCAtariWrapper
    except ImportError as e:
        print("❌ Error: Cannot import 'FreewaySpeedWrapper'.")
        print(f"Searched in: {wrapper_dir}")
        print(f"Details: {e}")
        sys.exit(1)

def load_agent(path_str):
    path = pathlib.Path(path_str)
    if not path.exists():
        print(f"❌ Error: Agent file not found at: {path}")
        print("Check if AGENT_PATH is correct.")
        sys.exit(1)
    
    spec = importlib.util.spec_from_file_location("test_agent", str(path))
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    
    if not hasattr(agent_module, 'get_action'):
        print("❌ Agent is missing 'get_action(observation)' function!")
        sys.exit(1)

    return agent_module.get_action

def run_simulation(action_func, seed, render=False):
    """Runs a Freeway game and returns the score."""
    render_mode = "human" if render else None
    
    try:
        # Freeway Environment Setup
        try:
            env = gym.make('ALE/Freeway-v5', render_mode=render_mode, obs_type='ram')
        except:
            env = gym.make('Freeway-v4', render_mode=render_mode, obs_type='ram')
            
        env = FreewayOCAtariWrapper(env)
        
        obs, info = env.reset(seed=seed)
        
        total_score = 0.0
        steps = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and steps < MAX_STEPS:
            try:
                # Handle agent output (list, array, or int)
                action = action_func(obs)
                if isinstance(action, (list, tuple, np.ndarray)):
                    action = int(action[0])
                action = int(action)
            except Exception:
                action = 0 # No-op on error
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_score += reward
            steps += 1
            
            if render: 
                time.sleep(0.01) # Playback speed (~100fps, smooth)
                
        env.close()
        return total_score
        
    except Exception as e:
        print(f"Critical error (Seed {seed}): {e}")
        return -9999

def main():
    print(f"\n🐔 TESTING AGENT (Hardcoded): {os.path.basename(AGENT_PATH)} 🐔")
    print("="*60)
    
    # 1. Load Agent
    get_action = load_agent(AGENT_PATH)
    print("✅ Agent loaded.")
    
    # 2. Initial Preview
    print(f"🎥  PREVIEW (Seed {VISUALIZATION_SEED})...")
    run_simulation(get_action, seed=VISUALIZATION_SEED, render=True)
    print("-" * 60)
    
    # 3. Benchmark on 100 Seeds
    print(f"\n📊  BENCHMARK ON {len(TEST_SEEDS_RANGE)} SEEDS (Variable Traffic)...")
    
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
    print(f"AVERAGE CROSSING:  {avg_score:.2f} (± {std_dev:.2f})")
    print(f"NOTE: In Freeway >21 is impossible, >18 is excellent.")
    print("-" * 30)
    print(f"🏆 BEST RUN:  Seed {best_run['seed']} -> Score {best_run['score']:.0f}")
    print(f"💩 WORST RUN: Seed {worst_run['seed']} -> Score {worst_run['score']:.0f}")
    print("="*60)
    
    # 5. Final Replay
    choice = input(f"\nWatch the BEST run (Seed {best_run['seed']})? [y/n]: ")
    if choice.lower() == 'y':
        print(f"Replaying Seed {best_run['seed']}...")
        run_simulation(get_action, seed=best_run['seed'], render=True)

if __name__ == "__main__":
    main()