import os
import sys
import importlib.util
import time
import pathlib
import re
import gymnasium as gym
import numpy as np

# --- PATH CONFIG ---
BASE_DIR = pathlib.Path(__file__).parent.resolve()
RESULTS_DIR = BASE_DIR / "results"

# Setup path for wrapper import
sys.path.append(str(BASE_DIR))

try:
    from wrapper.skiing_wrapper import SkiingOCAtariWrapper
except ImportError:
    print("❌ Error: Cannot import 'wrapper.skiing_wrapper'.")
    sys.exit(1)

def find_latest_run_dir():
    """Finds the most recent run directory."""
    if not RESULTS_DIR.exists():
        print(f"❌ Error: Results folder does not exist: {RESULTS_DIR}")
        sys.exit(1)

    run_folders = [f for f in RESULTS_DIR.iterdir() if f.is_dir() and f.name.startswith("run_skiing_")]
    
    if not run_folders:
        print(f"❌ Error: No 'run_skiing_' folders found.")
        sys.exit(1)

    # Sort chronologically
    latest_run = sorted(run_folders, key=lambda x: x.name)[-1]
    return latest_run

def list_interesting_agents(run_dir):
    """
    Scans 'interesting_agents' folder and returns files sorted by score.
    """
    agents_dir = run_dir / "interesting_agents"
    
    if not agents_dir.exists():
        print(f"⚠️ Folder 'interesting_agents' does not exist in {run_dir.name}")
        return []
    
    agent_files = list(agents_dir.glob("agent_*.py"))
    
    if not agent_files:
        print(f"⚠️ Folder 'interesting_agents' is empty.")
        return []

    # Parse files to extract score
    # Expected format: agent_{SCORE}_pts_{TIMESTAMP}.py
    parsed_agents = []
    for f in agent_files:
        match = re.search(r"agent_(-?\d+)_pts", f.name)
        score = int(match.group(1)) if match else 0
        parsed_agents.append({
            'path': f,
            'filename': f.name,
            'score': score
        })
    
    # Sort descending (best first)
    parsed_agents.sort(key=lambda x: x['score'], reverse=True)
    return parsed_agents

def load_agent_from_file(filepath):
    """Dynamically loads the get_action function."""
    spec = importlib.util.spec_from_file_location("interesting_agent", str(filepath))
    agent_module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(agent_module)
    except Exception as e:
        print(f"❌ Error loading code: {e}")
        return None
    
    if not hasattr(agent_module, 'get_action'):
        print("❌ ERROR: Function 'get_action' missing.")
        return None
        
    return agent_module.get_action

def run_simulation(agent_path):
    """Runs a single visual simulation."""
    print(f"\n📂 Loading: {agent_path.name}")
    get_action = load_agent_from_file(agent_path)
    if not get_action: return

    print("🎮 Starting visual env...")
    try:
        env = SkiingOCAtariWrapper(render_mode="human")
        observation, info = env.reset(seed=42)
        
        done = False
        total_reward = 0.0
        steps = 0
        
        print("--- STARTING RUN (Ctrl+C to stop) ---")
        while not done:
            try:
                action = int(get_action(observation))
            except Exception as e:
                print(f"❌ Agent Error: {e}")
                break

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            time.sleep(0.02) # 50 FPS
            
            done = terminated or truncated
            
            if steps % 60 == 0:
                print(f"Step {steps:4d} | Reward: {total_reward:6.1f} | DeltaX: {observation[3]:.2f}")
        
        env.close()
        print(f"🏆 Final Score: {total_reward:.2f}")
        
    except KeyboardInterrupt:
        print("\n🛑 Run interrupted by user.")
        if 'env' in locals(): env.close()
    except Exception as e:
        print(f"❌ Env Error: {e}")
        if 'env' in locals(): env.close()

def main():
    print(f"--- VISUALIZE INTERESTING AGENTS ---")
    
    latest_run = find_latest_run_dir()
    print(f"🔍 Latest Run: {latest_run.name}")
    
    while True:
        agents = list_interesting_agents(latest_run)
        
        if not agents:
            print("No interesting agents found.")
            break
            
        print("\n--- Available Agents ---")
        for i, agent in enumerate(agents):
            print(f"[{i+1}] Score: {agent['score']:<6} | File: {agent['filename']}")
            
        print("\n[0] Exit")
        
        choice = input("\nWhich agent to view? (Enter number): ").strip()
        
        if choice == '0':
            print("👋 Exiting.")
            break
            
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(agents):
                selected_agent = agents[idx]
                run_simulation(selected_agent['path'])
                input("\nPress ENTER to return to menu...")
            else:
                print("⚠️ Invalid number.")
        except ValueError:
            print("⚠️ Please enter a valid number.")

if __name__ == "__main__":
    main()