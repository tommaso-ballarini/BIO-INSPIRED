import os
import sys
import importlib.util
import time
import pathlib
import gymnasium as gym
import numpy as np
from ocatari.core import OCAtari

# --- CONFIGURAZIONE PERCORSI ---
BASE_DIR = pathlib.Path(__file__).parent.resolve()
RESULTS_DIR = BASE_DIR / "results"

# Setup percorsi per importare il wrapper
# Aggiungiamo sia la base che la cartella wrapper per sicurezza
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "wrapper"))

try:
    # Tentativo 1: Nome standard
    from wrapper.si_wrapper import SpaceInvadersEgocentricWrapper
except ImportError:
    try:
        # Tentativo 2: Nome NEAT originale (più probabile)
        from wrapper.wrapper_si_ego import SpaceInvadersEgocentricWrapper
    except ImportError:
        try:
            # Tentativo 3: Import diretto se wrapper è nel path
            from wrapper_si_ego import SpaceInvadersEgocentricWrapper
        except ImportError:
            print("❌ Errore: Impossibile importare il wrapper di Space Invaders.")
            print(f"Verifica che esista 'wrapper_si_ego.py' in: {BASE_DIR / 'wrapper'}")
            sys.exit(1)

def find_latest_best_agent():
    """
    Scansiona la cartella results e trova il file best_program.py 
    dell'ultima run eseguita.
    """
    if not RESULTS_DIR.exists():
        print(f"❌ Errore: La cartella risultati non esiste: {RESULTS_DIR}")
        sys.exit(1)

    # Trova tutte le cartelle che iniziano con 'run_si_'
    run_folders = [f for f in RESULTS_DIR.iterdir() if f.is_dir() and f.name.startswith("run_si_")]
    
    if not run_folders:
        print(f"❌ Errore: Nessuna cartella 'run_si_' trovata in {RESULTS_DIR}")
        sys.exit(1)

    # Ordina cronologicamente
    latest_run = sorted(run_folders, key=lambda x: x.name)[-1]
    
    agent_path = latest_run / "best" / "best_program.py"
    
    print(f"🔍 Run più recente trovata: {latest_run.name}")
    
    if not agent_path.exists():
        print(f"⚠️ Attenzione: Il file 'best_program.py' non esiste in questa run.")
        print("Forse la run non ha ancora prodotto un vincitore o è crashata?")
        sys.exit(1)
        
    return agent_path

def load_agent_from_file(filepath):
    """Carica dinamicamente la funzione get_action dal file python."""
    print(f"📂 Caricamento agente da: {filepath}")
    
    spec = importlib.util.spec_from_file_location("best_agent", str(filepath))
    agent_module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(agent_module)
    except Exception as e:
        print(f"❌ Errore durante il caricamento del codice agente: {e}")
        sys.exit(1)
    
    if not hasattr(agent_module, 'get_action'):
        print("❌ ERRORE: Il file non contiene la funzione 'get_action'.")
        sys.exit(1)
        
    return agent_module.get_action

def visualize():
    print(f"\n--- VISUALIZZAZIONE BEST AGENT (Space Invaders) ---")
    
    agent_path = find_latest_best_agent()
    get_action = load_agent_from_file(agent_path)
    
    print("🎮 Avvio ambiente grafico...")
    try:
        # UPDATED: Usa NoFrameskip-v4 come nel training
        env = OCAtari("SpaceInvadersNoFrameskip-v4", mode="ram", hud=False, render_mode="human")
        env = SpaceInvadersEgocentricWrapper(env, skip=4)
    except Exception as e:
        print(f"Errore avvio ambiente: {e}")
        return

    # Usiamo un seed fisso per vedere una partita standard
    observation, info = env.reset(seed=42) 
    
    done = False
    total_reward = 0.0
    steps = 0
    
    print("--- INIZIO PARTITA (Premi Ctrl+C per fermare) ---")
    try:
        while not done:
            try:
                # L'agente decide
                action = int(get_action(observation))
            except Exception as e:
                print(f"❌ Errore nell'agente: {e}")
                break

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Rallenta per rendere visibile a occhio umano (0.03s ~ 30fps)
            time.sleep(0.03)
            
            done = terminated or truncated
            
            if steps % 60 == 0:
                # obs[3] = Danger, obs[11] = Targeting
                danger = observation[3]
                target = observation[11]
                print(f"Step {steps:4d} | Reward: {total_reward:6.1f} | Danger: {danger:.2f} | Aim: {target:.2f}")
                
    except KeyboardInterrupt:
        print("\n🛑 Interrotto dall'utente.")
    except Exception as e:
        print(f"\n❌ Errore runtime: {e}")
        
    env.close()
    print(f"\n--- FINE PARTITA ---")
    print(f"🏆 Punteggio Finale: {total_reward}")

if __name__ == "__main__":
    visualize()