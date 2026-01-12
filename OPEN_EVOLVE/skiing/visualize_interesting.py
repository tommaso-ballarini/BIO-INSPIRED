import os
import sys
import importlib.util
import time
import pathlib
import re
import gymnasium as gym
import numpy as np

# --- CONFIGURAZIONE PERCORSI ---
BASE_DIR = pathlib.Path(__file__).parent.resolve()
RESULTS_DIR = BASE_DIR / "results"

# Setup percorsi per importare il wrapper
sys.path.append(str(BASE_DIR))

try:
    from wrapper.skiing_wrapper import SkiingOCAtariWrapper
except ImportError:
    print("❌ Errore: Impossibile importare 'wrapper.skiing_wrapper'.")
    sys.exit(1)

def find_latest_run_dir():
    """Trova la cartella della run più recente."""
    if not RESULTS_DIR.exists():
        print(f"❌ Errore: La cartella risultati non esiste: {RESULTS_DIR}")
        sys.exit(1)

    run_folders = [f for f in RESULTS_DIR.iterdir() if f.is_dir() and f.name.startswith("run_skiing_")]
    
    if not run_folders:
        print(f"❌ Errore: Nessuna cartella 'run_skiing_' trovata.")
        sys.exit(1)

    # Ordina cronologicamente
    latest_run = sorted(run_folders, key=lambda x: x.name)[-1]
    return latest_run

def list_interesting_agents(run_dir):
    """
    Scansiona la cartella 'interesting_agents' e restituisce una lista di file
    ordinati per score decrescente.
    """
    agents_dir = run_dir / "interesting_agents"
    
    if not agents_dir.exists():
        print(f"⚠️ La cartella 'interesting_agents' non esiste in {run_dir.name}")
        return []
    
    agent_files = list(agents_dir.glob("agent_*.py"))
    
    if not agent_files:
        print(f"⚠️ La cartella 'interesting_agents' è vuota.")
        return []

    # Parsing dei file per estrarre lo score
    # Formato atteso: agent_{SCORE}_pts_{TIMESTAMP}.py
    parsed_agents = []
    for f in agent_files:
        match = re.search(r"agent_(-?\d+)_pts", f.name)
        score = int(match.group(1)) if match else 0
        parsed_agents.append({
            'path': f,
            'filename': f.name,
            'score': score
        })
    
    # Ordina per score decrescente (i migliori in cima)
    parsed_agents.sort(key=lambda x: x['score'], reverse=True)
    return parsed_agents

def load_agent_from_file(filepath):
    """Carica dinamicamente la funzione get_action."""
    spec = importlib.util.spec_from_file_location("interesting_agent", str(filepath))
    agent_module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(agent_module)
    except Exception as e:
        print(f"❌ Errore caricamento codice: {e}")
        return None
    
    if not hasattr(agent_module, 'get_action'):
        print("❌ ERRORE: Funzione 'get_action' mancante.")
        return None
        
    return agent_module.get_action

def run_simulation(agent_path):
    """Esegue una singola simulazione visiva."""
    print(f"\n📂 Caricamento: {agent_path.name}")
    get_action = load_agent_from_file(agent_path)
    if not get_action: return

    print("🎮 Avvio ambiente grafico...")
    try:
        env = SkiingOCAtariWrapper(render_mode="human")
        observation, info = env.reset(seed=42)
        
        done = False
        total_reward = 0.0
        steps = 0
        
        print("--- INIZIO DISCESA (Premi Ctrl+C per interrompere la run attuale) ---")
        while not done:
            try:
                action = int(get_action(observation))
            except Exception as e:
                print(f"❌ Errore agente: {e}")
                break

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            time.sleep(0.02) # 50 FPS
            
            done = terminated or truncated
            
            if steps % 60 == 0:
                print(f"Step {steps:4d} | Reward: {total_reward:6.1f} | DeltaX: {observation[3]:.2f}")
        
        env.close()
        print(f"🏆 Punteggio Finale: {total_reward:.2f}")
        
    except KeyboardInterrupt:
        print("\n🛑 Run interrotta dall'utente.")
        if 'env' in locals(): env.close()
    except Exception as e:
        print(f"❌ Errore ambiente: {e}")
        if 'env' in locals(): env.close()

def main():
    print(f"--- VISUALIZZAZIONE INTERESTING AGENTS ---")
    
    latest_run = find_latest_run_dir()
    print(f"🔍 Run più recente: {latest_run.name}")
    
    while True:
        agents = list_interesting_agents(latest_run)
        
        if not agents:
            print("Nessun agente interessante trovato.")
            break
            
        print("\n--- Agenti Disponibili ---")
        for i, agent in enumerate(agents):
            print(f"[{i+1}] Score: {agent['score']:<6} | File: {agent['filename']}")
            
        print("\n[0] Esci")
        
        choice = input("\nQuale agente vuoi vedere? (Inserisci numero): ").strip()
        
        if choice == '0':
            print("👋 Uscita.")
            break
            
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(agents):
                selected_agent = agents[idx]
                run_simulation(selected_agent['path'])
                input("\nPremi INVIO per tornare al menu...")
            else:
                print("⚠️ Numero non valido.")
        except ValueError:
            print("⚠️ Inserisci un numero valido.")

if __name__ == "__main__":
    main()