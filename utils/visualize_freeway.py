import sys
import os
import time
import importlib.util
import pandas as pd
import gymnasium as gym
import numpy as np

# --- 1. FIX IMPORTANTE: Registrazione Atari ---
# Gymnasium ha bisogno di questo import per "vedere" Freeway
try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    print("Attenzione: 'ale_py' non trovato. Potrebbe servire 'pip install ale-py gymnasium[atari]'")
except AttributeError:
    pass # Versioni vecchie gestiscono diversamente

# --- 2. Setup Percorsi (Adattato per la cartella utils/) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Se siamo in utils/, il project root è la cartella padre
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# Aggiungiamo il path per eventuali import locali
if project_root not in sys.path:
    sys.path.append(project_root)

history_dir = os.path.join(project_root, 'evolution_history')
history_csv = os.path.join(history_dir, 'fitness_history.csv')

# --- Configurazione ---
ENV_NAME = 'Freeway-v4' 
FPS = 30

def load_agent_module(agent_path):
    """Carica dinamicamente il modulo dell'agente dal file .py."""
    try:
        spec = importlib.util.spec_from_file_location("best_agent", agent_path)
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        return agent_module
    except Exception as e:
        print(f"Errore nel caricamento dell'agente {agent_path}: {e}")
        sys.exit(1)

def get_best_agent_path():
    """Legge il CSV e trova il file con lo score più alto."""
    if not os.path.exists(history_csv):
        print(f"Errore: Non trovo il file di storia in {history_csv}")
        print("Assicurati di aver eseguito l'esperimento 'run_openevolve_freeway.py'.")
        sys.exit(1)

    try:
        df = pd.read_csv(history_csv)
        if df.empty:
            print("Il file CSV è vuoto.")
            sys.exit(1)
        
        # Trova la riga con il punteggio massimo
        # Ordiniamo per score (desc) e poi per timestamp (desc)
        best_row = df.sort_values(by=['score', 'timestamp'], ascending=[False, False]).iloc[0]
        
        best_score = best_row['score']
        filename = best_row['filename']
        full_path = os.path.join(history_dir, filename)
        
        print(f"\n--- MIGLIOR AGENTE TROVATO ---")
        print(f"File: {filename}")
        print(f"Score: {best_score}")
        print(f"------------------------------\n")
        
        return full_path
    except Exception as e:
        print(f"Errore nella lettura del CSV: {e}")
        sys.exit(1)

def run_visualization():
    agent_path = get_best_agent_path()
    
    # Carica l'agente
    agent_module = load_agent_module(agent_path)
    if not hasattr(agent_module, 'get_action'):
        print("Errore: L'agente non ha la funzione 'get_action'.")
        sys.exit(1)
    get_action = agent_module.get_action

    print(f"Avvio visualizzazione {ENV_NAME}...")
    
    # Creazione Ambiente
    # Prova prima col nome standard, se fallisce prova il nome ALE specifico
    try:
        env = gym.make(ENV_NAME, render_mode='human', obs_type='ram')
    except gym.error.NameNotFound:
        print(f"Non trovo '{ENV_NAME}', provo con 'ALE/Freeway-v5'...")
        try:
            env = gym.make('ALE/Freeway-v5', render_mode='human', obs_type='ram')
        except Exception as e:
            print("\nERRORE CRITICO: Impossibile caricare l'ambiente.")
            print("Assicurati di aver installato i pacchetti: pip install gymnasium[atari] ale-py")
            raise e

    observation, _ = env.reset()
    
    total_reward = 0
    steps = 0
    done = False
    
    print("Premi Ctrl+C per interrompere.")
    
    try:
        while not done:
            action = get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            time.sleep(1.0 / FPS)
            
            if steps % 100 == 0:
                print(f"Step: {steps} | Reward: {total_reward:.1f}", end='\r')

    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print(f"\n\nPartita terminata. Score Finale: {total_reward}")

if __name__ == "__main__":
    run_visualization()