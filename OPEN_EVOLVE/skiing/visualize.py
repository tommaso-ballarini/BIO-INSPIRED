import os
import sys
import importlib.util
import time
import pathlib
import gymnasium as gym
import numpy as np

# --- CONFIGURAZIONE PERCORSI ---
# Determina la cartella base (dove si trova questo script)
BASE_DIR = pathlib.Path(__file__).parent.resolve()
RESULTS_DIR = BASE_DIR / "results"

# Setup percorsi per importare il wrapper
sys.path.append(str(BASE_DIR))

try:
    from wrapper.skiing_wrapper import SkiingOCAtariWrapper
except ImportError:
    print("❌ Errore: Impossibile importare 'wrapper.skiing_wrapper'.")
    print(f"Assicurati che questo script sia nella cartella 'skiing' e che 'wrapper' esista in: {BASE_DIR}")
    sys.exit(1)

def find_latest_best_agent():
    """
    Scansiona la cartella results e trova il file best_program.py 
    dell'ultima run eseguita (basandosi sul timestamp nel nome della cartella).
    """
    if not RESULTS_DIR.exists():
        print(f"❌ Errore: La cartella risultati non esiste: {RESULTS_DIR}")
        sys.exit(1)

    # Trova tutte le cartelle che iniziano con 'run_skiing_'
    run_folders = [f for f in RESULTS_DIR.iterdir() if f.is_dir() and f.name.startswith("run_skiing_")]
    
    if not run_folders:
        print(f"❌ Errore: Nessuna cartella 'run_skiing_' trovata in {RESULTS_DIR}")
        sys.exit(1)

    # Ordina per nome (che contiene il timestamp YYYYMMDD_HHMMSS, quindi è cronologico)
    # L'ultima della lista è la più recente
    latest_run = sorted(run_folders, key=lambda x: x.name)[-1]
    
    # Costruisce il percorso al file del best agent
    agent_path = latest_run / "best" / "best_program.py"
    
    print(f"🔍 Run più recente trovata: {latest_run.name}")
    
    if not agent_path.exists():
        print(f"⚠️ Attenzione: Il file 'best_program.py' non esiste in questa run.")
        print(f"Percorso cercato: {agent_path}")
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
    print(f"\n--- VISUALIZZAZIONE CAMPIONE OPEN-EVOLVE ---")
    
    # 1. Trova e Carica il cervello automaticamente
    agent_path = find_latest_best_agent()
    get_action = load_agent_from_file(agent_path)
    
    # 2. Crea l'ambiente con render 'human' (finestra video)
    print("🎮 Avvio ambiente grafico...")
    try:
        env = SkiingOCAtariWrapper(render_mode="human")
    except Exception as e:
        print(f"Errore avvio ambiente: {e}")
        return

    observation, info = env.reset(seed=42) # Seed 42 per replicabilità
    
    done = False
    total_reward = 0.0
    steps = 0
    
    print("--- INIZIO DISCESA (Premi Ctrl+C nella console per fermare) ---")
    try:
        while not done:
            # L'agente LLM decide l'azione
            try:
                action = int(get_action(observation))
            except Exception as e:
                print(f"❌ Errore nell'agente durante la scelta azione: {e}")
                break

            # Step fisico
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Rallenta un po' per goderti la scena (0.02s = 50fps circa)
            time.sleep(0.02)
            
            done = terminated or truncated
            
            if steps % 60 == 0:
                # obs[3] è il delta magnetico
                magnet_dist = observation[3]
                print(f"Step {steps:4d} | Reward: {total_reward:6.1f} | Magnet Dist: {magnet_dist:.2f}")
                
    except KeyboardInterrupt:
        print("\n🛑 Interrotto dall'utente.")
    except Exception as e:
        print(f"\n❌ Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()
        
    env.close()
    print(f"\n--- FINE PARTITA ---")
    print(f"🏆 Punteggio Finale: {total_reward}")

if __name__ == "__main__":
    visualize()