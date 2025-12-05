import sys
import os
import time
import importlib.util
import pandas as pd
import gymnasium as gym
import glob
import numpy as np

# --- 1. SETUP PERCORSI ---
# Posizione: OPEN_EVOLVE/freeway/run/visualize.py
current_dir = os.path.dirname(os.path.abspath(__file__))
# Root esperimento: OPEN_EVOLVE/freeway/
experiment_root = os.path.abspath(os.path.join(current_dir, '..'))

wrapper_dir = os.path.join(experiment_root, 'wrapper')
history_dir = os.path.join(experiment_root, 'history')
history_csv = os.path.join(history_dir, 'fitness_history.csv')

# Aggiungiamo il wrapper al path
sys.path.append(wrapper_dir)

# --- 2. IMPORT WRAPPER ---
try:
    from freeway_wrapper import FreewayOCAtariWrapper
except ImportError as e:
    print(f"ERRORE: Impossibile trovare il wrapper in {wrapper_dir}")
    sys.exit(1)

# Fix per Atari (Import esplicito come in evaluator.py)
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
    """Mostra un menu per scegliere l'agente o restituisce il migliore di default."""
    if not os.path.exists(history_csv):
        print(f"Nessun file history trovato in: {history_csv}")
        sys.exit(1)
        
    df = pd.read_csv(history_csv)
    # Filtra errori (score -1000)
    valid_df = df[df['score'] > -900].sort_values(by='score', ascending=False)
    
    if valid_df.empty:
        print("Nessun agente valido trovato nella history.")
        sys.exit(1)

    best_agent = valid_df.iloc[0]['filename']
    
    print("\n--- TOP 10 AGENTI EVOLUTI ---")
    top_10 = valid_df.head(10).reset_index()
    for idx, row in top_10.iterrows():
        col_info = ""
        if 'collisions' in row:
            col_info = f"| Col: {row['collisions']:.1f}"
        print(f"[{idx}] Score: {row['score']:.1f} {col_info} | File: {row['filename']}")
    
    print("\nInserisci ID (0-9) o premi INVIO per il migliore.")
    choice = input(">> ").strip()
    
    filename = best_agent
    
    if choice:
        if choice.isdigit() and int(choice) < len(top_10):
            filename = top_10.iloc[int(choice)]['filename']
        elif choice.endswith('.py'):
            filename = choice

    full_path = os.path.join(history_dir, filename)
    if not os.path.exists(full_path):
        print(f"Errore: Il file {full_path} non esiste.")
        sys.exit(1)
        
    print(f"Caricamento: {filename}...\n")
    return full_path

def run_visualization():
    agent_path = select_agent()
    # Carichiamo la funzione get_action dall'agente
    try:
        agent_module = load_agent_module(agent_path)
        agent = agent_module.get_action
    except Exception as e:
        print(f"Errore nel caricamento dell'agente: {e}")
        return

    # --- 3. CREAZIONE ENVIRONMENT + WRAPPER ---
    try:
        # Render mode 'human' per vedere la finestra
        env = gym.make(ENV_NAME, render_mode='human', obs_type='ram')
    except:
        env = gym.make('ALE/Freeway-v5', render_mode='human', obs_type='ram')

    # APPLICHIAMO LO STESSO WRAPPER DELL'ADDESTRAMENTO
    env = FreewayOCAtariWrapper(env)

    obs, _ = env.reset()
    total_reward = 0
    
    # Per il calcolo collisioni visuale, usiamo l'osservazione del wrapper
    # obs[0] è la Y del pollo normalizzata (0.0 start, 1.0 goal)
    prev_y = obs[0] 
    collisions = 0
    
    print(f"START! Visualizzazione agente: {os.path.basename(agent_path)}")
    print("(Premi Ctrl+C nel terminale per uscire)")
    
    try:
        while True:
            # L'agente riceve l'obs processata dal wrapper (11 float), non la RAM
            try:
                action = agent(obs)
                
                # Pulizia output LLM
                if isinstance(action, (list, tuple, np.ndarray)):
                    action = action[0]
                action = int(action)
                if action not in [0, 1, 2]: action = 1
            except:
                action = 1

            # Step dell'ambiente wrappato
            obs, reward, done, trunc, _ = env.step(action)
            
            # Logica collisioni basata sul wrapper (uguale a evaluator.py)
            curr_y = obs[0]
            # Se la Y scende bruscamente, è una collisione
            if curr_y < (prev_y - 0.05):
                collisions += 1
                print(f"💥 Collisione! (Tot: {collisions})")
            
            prev_y = curr_y
            total_reward += reward
            
            # Rallentiamo per rendere visibile all'occhio umano
            time.sleep(1.0/FPS)
            
            if done or trunc:
                print(f"Fine Episodio. Score: {total_reward} | Collisioni: {collisions}")
                time.sleep(1) # Pausa breve tra episodi
                obs, _ = env.reset()
                total_reward = 0
                collisions = 0
                prev_y = obs[0]
                
    except KeyboardInterrupt:
        print("\nChiusura visualizzazione...")
        env.close()

if __name__ == "__main__":
    run_visualization()
