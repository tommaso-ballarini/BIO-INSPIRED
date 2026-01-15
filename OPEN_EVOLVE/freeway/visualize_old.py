import sys
import os
import time
import importlib.util
import pandas as pd
import gymnasium as gym
import glob
import numpy as np

# --- 1. PATH SETUP ---
# Location: OPEN_EVOLVE/freeway/run/visualize.py
current_dir = os.path.dirname(os.path.abspath(__file__))
# Experiment Root: OPEN_EVOLVE/freeway/
experiment_root = os.path.abspath(os.path.join(current_dir, '..'))

wrapper_dir = os.path.join(experiment_root, 'wrapper')
history_dir = os.path.join(experiment_root, 'history')
history_csv = os.path.join(history_dir, 'fitness_history.csv')

# Add wrapper to path
sys.path.append(wrapper_dir)

# --- 2. IMPORT WRAPPER ---
try:
    from freeway_wrapper import FreewayOCAtariWrapper
except ImportError as e:
    print(f"ERROR: Cannot find wrapper in {wrapper_dir}")
    sys.exit(1)

import ale_py
gym.register_envs(ale_py)

ENV_NAME = 'Freeway-v4'
FPS = 30

def load_agent_module(agent_path):
    spec = importlib.util.spec_from_file_location("view_agent", agent_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def select_agent():
    """Displays a menu to select an agent or returns the best one by default."""
    if not os.path.exists(history_csv):
        print(f"No history file found in: {history_csv}")
        sys.exit(1)
        
    df = pd.read_csv(history_csv)
    # Filter errors (score -1000)
    valid_df = df[df['score'] > -900].sort_values(by='score', ascending=False)
    
    if valid_df.empty:
        print("No valid agent found in history.")
        sys.exit(1)

    best_agent = valid_df.iloc[0]['filename']
    
    print("\n--- TOP 10 EVOLVED AGENTS ---")
    top_10 = valid_df.head(10).reset_index()
    for idx, row in top_10.iterrows():
        col_info = ""
        if 'collisions' in row:
            col_info = f"| Col: {row['collisions']:.1f}"
        print(f"[{idx}] Score: {row['score']:.1f} {col_info} | File: {row['filename']}")
    
    print("\nEnter ID (0-9) or press ENTER for the best.")
    choice = input(">> ").strip()
    
    filename = best_agent
    
    if choice:
        if choice.isdigit() and int(choice) < len(top_10):
            filename = top_10.iloc[int(choice)]['filename']
        elif choice.endswith('.py'):
            filename = choice

    full_path = os.path.join(history_dir, filename)
    if not os.path.exists(full_path):
        print(f"Error: File {full_path} does not exist.")
        sys.exit(1)
        
    print(f"Loading: {filename}...\n")
    return full_path

def run_visualization():
    agent_path = select_agent()
    try:
        agent_module = load_agent_module(agent_path)
        agent = agent_module.get_action
    except Exception as e:
        print(f"Error loading agent: {e}")
        return

    # --- 3. ENVIRONMENT + WRAPPER SETUP ---
    try:
        env = gym.make(ENV_NAME, render_mode='human', obs_type='ram')
    except:
        env = gym.make('ALE/Freeway-v5', render_mode='human', obs_type='ram')

    env = FreewayOCAtariWrapper(env)

    obs, _ = env.reset()
    total_reward = 0
    
    # Visual collision calculation using wrapper observation
    # obs[0] is normalized Chicken Y (0.0 start, 1.0 goal)
    prev_y = obs[0] 
    collisions = 0
    
    print(f"START! Visualizing agent: {os.path.basename(agent_path)}")
    print("(Press Ctrl+C in terminal to exit)")
    
    try:
        while True:
            # Agent receives processed wrapper obs (11 floats), not RAM
            try:
                action = agent(obs)
                
                if isinstance(action, (list, tuple, np.ndarray)):
                    action = action[0]
                action = int(action)
                if action not in [0, 1, 2]: action = 1
            except:
                action = 1

            obs, reward, done, trunc, _ = env.step(action)
            
            curr_y = obs[0]
            if curr_y < (prev_y - 0.05):
                collisions += 1
                print(f"💥 Collision! (Tot: {collisions})")
            
            prev_y = curr_y
            total_reward += reward
            
            time.sleep(1.0/FPS)
            
            if done or trunc:
                print(f"Episode End. Score: {total_reward} | Collisions: {collisions}")
                time.sleep(1) 
                obs, _ = env.reset()
                total_reward = 0
                collisions = 0
                prev_y = obs[0]
                
    except KeyboardInterrupt:
        print("\nClosing visualization...")
        env.close()

if __name__ == "__main__":
    run_visualization()