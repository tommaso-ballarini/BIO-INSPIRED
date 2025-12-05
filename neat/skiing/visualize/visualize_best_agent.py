# File: visualize/visualize_best_agent.py

import os
import sys
import neat
import pickle
import numpy as np
import gymnasium as gym
import time
from glob import glob
# IMPORTANTE: Registra gli ambienti ALE
try:
    import ale_py
    # Questa riga registra tutti gli ambienti Atari disponibili
    gym.register_envs(ale_py)
except ImportError:
    print("❌ ale-py non installato. Installa con: pip install ale-py[atari]")
    sys.exit(1)
# --- CONFIGURAZIONE E SETUP PERCORSI ---
# Assumi che lo script sia in neat/skiing/visualize/
script_dir = os.path.dirname(os.path.abspath(__file__))

# La cartella 'skiing' è la radice del sotto-progetto
project_root = os.path.abspath(os.path.join(script_dir, '..')) 

# Aggiungi 'skiing' al path per importare il wrapper
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from wrapper.skiing_wrapper import SkiingCustomWrapper
except ImportError:
    print("❌ Errore: Impossibile importare SkiingCustomWrapper. Controlla il path e __init__.py.")
    sys.exit(1)

# Impostazioni di ambiente e file
ENV_ID = "ALE/Skiing-v5"
OUTPUT_DIR = os.path.join(project_root, "evolution_results", "skiing")
MAX_STEPS = 20000

# --- FUNZIONI DI SUPPORTO ---

def normalize_ram(ram_state):
    """Normalizza lo stato RAM (128 bytes) in [0, 1]"""
    return np.array(ram_state, dtype=np.float32) / 255.0

def load_winner():
    """Trova e carica il genoma vincente più recente."""
    
    # Cerca tutti i file winner.pkl nell'output directory
    search_path = os.path.join(OUTPUT_DIR, "winner_*.pkl")
    winner_files = glob(search_path)
    
    if not winner_files:
        print(f"❌ Errore: Nessun file 'winner_*.pkl' trovato in {OUTPUT_DIR}")
        print("Esegui prima la run di evoluzione con run_ski_neat.py.")
        sys.exit(1)
        
    # Ordina per nome (che include il timestamp) e prende l'ultimo
    latest_winner_file = max(winner_files, key=os.path.getctime)
    
    print(f"✅ Caricamento genoma da: {os.path.basename(latest_winner_file)}")
    
    with open(latest_winner_file, 'rb') as f:
        winner_genome, config = pickle.load(f)
        
    return winner_genome, config

def visualize_game(genome, config):
    """Esegue il gioco usando il genoma NEAT e lo visualizza."""
    
    # 1. Setup Ambiente (ora con render_mode="human")
    env = gym.make(
        ENV_ID,
        obs_type="ram",
        frameskip=1,
        repeat_action_probability=0.0,
        render_mode="human" # ESSENZIALE per la visualizzazione
    )
    
    # 2. Wrappa l'ambiente con gli stessi parametri della run
    # Manteniamo gli stessi parametri usati per l'allenamento per una valutazione equa
    wrapped_env = SkiingCustomWrapper(
            env, 
            enable_steering_cost=False,             # Abilita/Disabilita costo sterzo
            min_change_ratio=0.2,                 # Minima percentuale di azioni non-NOOP richieste
            steering_cost_per_step=1.0,             # Penalità per ogni passo di sterzo,           
            edge_penalty_multiplier=30.0, # AUMENTA QUESTO VALORE per punire i bordi
            edge_threshold=40
        )
    
    # 3. Setup Rete NEAT
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # 4. Inizializzazione Partita
    observation, info = wrapped_env.reset()
    done = False
    steps = 0
    total_reward = 0.0
    
    print("\n--- AVVIO SIMULAZIONE ---")
    
    # 5. Loop di Gioco
    while not done and steps < MAX_STEPS:
        try:
            # Calcola l'azione
            norm_obs = normalize_ram(observation)
            output = net.activate(norm_obs)
            action = int(np.argmax(output)) 
            
            # Esegui lo step (il wrapper si occupa della mappatura)
            observation, reward, terminated, truncated, info = wrapped_env.step(action)
            
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            # Piccolo ritardo per vedere l'azione (opzionale)
            # time.sleep(0.01) 
            
            # Renderizza l'immagine
            wrapped_env.render()
            
        except Exception as e:
            print(f"⚠️ Errore durante lo step: {e}")
            break
            
    # 6. Fine Partita
    env.close()
    
    cost_steering = wrapped_env.steering_cost
    cost_stability = wrapped_env.get_stability_penalty()
    cost_edge = wrapped_env.edge_cost # NUOVO
    
    fitness_min = abs(total_reward) + cost_steering + cost_stability + cost_edge
    
    print("\n--- RISULTATI FINALI ---")
    print(f"Total Steps: {steps}")
    print(f"Total Reward (negativo): {total_reward:.2f}")
    print(f"Costo Base (Tempo): {abs(total_reward):.2f}")
    print(f"Costi aggiunti: Sterzo={cost_steering:.2f}, Stabilità={cost_stability:.2f}, Bordi={cost_edge:.2f}")
    print(f"Costo Totale (F_min): {fitness_min:.2f}")
    print(f"Vincitore Fitness (Massimizzata): {150000.0 - fitness_min:.2f}")
    print("------------------------")


if __name__ == "__main__":
    
    # 1. Carica il genoma vincente e la configurazione
    winner_genome, config = load_winner()
    
    # 2. Visualizza la partita
    visualize_game(winner_genome, config)