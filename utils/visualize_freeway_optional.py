import sys
import os
import time
import importlib.util
import pandas as pd
import gymnasium as gym
import glob

# Fix per Atari
try:
    import ale_py
    gym.register_envs(ale_py)
except:
    pass

# Setup Percorsi
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
history_dir = os.path.join(project_root, 'evolution_history')
history_csv = os.path.join(history_dir, 'fitness_history.csv')

if project_root not in sys.path:
    sys.path.append(project_root)

ENV_NAME = 'Freeway-v4'
FPS = 30

def load_agent_module(agent_path):
    spec = importlib.util.spec_from_file_location("view_agent", agent_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def select_agent():
    """Mostra un menu per scegliere l'agente o restituisce il migliore di default."""
    if not os.path.exists(history_csv):
        print("Nessun file history trovato.")
        sys.exit(1)
        
    df = pd.read_csv(history_csv)
    # Filtra errori
    valid_df = df[df['score'] > -900].sort_values(by='score', ascending=False)
    
    if valid_df.empty:
        print("Nessun agente valido trovato.")
        sys.exit(1)

    best_agent = valid_df.iloc[0]['filename']
    best_score = valid_df.iloc[0]['score']

    print("\n--- AGENTI DISPONIBILI (Top 10) ---")
    top_10 = valid_df.head(10).reset_index()
    for idx, row in top_10.iterrows():
        col_info = ""
        if 'collisions' in row:
            col_info = f"| Collisions: {row['collisions']:.1f}"
        print(f"[{idx}] Score: {row['score']:.1f} {col_info} | File: {row['filename']}")
    
    print("\nInserisci l'ID (0-9) o il nome del file.")
    print(f"Premi INVIO per il migliore: [{best_agent}]")
    
    choice = input(">> ").strip()
    
    filename = best_agent
    
    if choice:
        if choice.isdigit() and int(choice) < len(top_10):
            filename = top_10.iloc[int(choice)]['filename']
        elif choice.endswith('.py'):
            filename = choice
        else:
            print("Scelta non valida, uso il migliore.")

    full_path = os.path.join(history_dir, filename)
    print(f"Caricamento: {filename}...\n")
    return full_path

def run_visualization():
    agent_path = select_agent()
    agent = load_agent_module(agent_path).get_action
    
    try:
        env = gym.make(ENV_NAME, render_mode='human', obs_type='ram')
    except:
        env = gym.make('ALE/Freeway-v5', render_mode='human', obs_type='ram')

    obs, _ = env.reset()
    total_reward = 0
    collisions = 0
    prev_y = obs[14]
    
    print("START! (Ctrl+C per uscire)")
    
    try:
        while True:
            action = agent(obs)
            obs, reward, done, trunc, _ = env.step(action)
            
            # Visualizza collisioni live
            curr_y = obs[14]
            if curr_y < prev_y:
                collisions += 1
                print(f"💥 Collisione! (Tot: {collisions})")
            prev_y = curr_y
            
            total_reward += reward
            time.sleep(1.0/FPS)
            
            if done or trunc:
                print(f"Fine Episodio. Score: {total_reward} | Collisioni: {collisions}")
                obs, _ = env.reset()
                total_reward = 0
                collisions = 0
                prev_y = obs[14]
                
    except KeyboardInterrupt:
        env.close()

if __name__ == "__main__":
    run_visualization()