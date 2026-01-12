import sys
import os
import time
import pickle
import numpy as np
import neat
import gymnasium as gym

# --- 1. GESTIONE PERCORSI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from ocatari.core import OCAtari
    from wrapper.wrapper_si_ego import SpaceInvadersEgocentricWrapper
except ImportError as e:
    print(f"❌ Errore importazione: {e}")
    sys.exit(1)

# --- CONFIGURAZIONI ---
CONFIG_PATH = os.path.join(project_root, 'config', 'config_si_ego.txt')
RESULTS_DIR = os.path.join(project_root, 'results')

# Cerca prima la lista top 3, altrimenti il vincitore singolo
TOP3_PATH = os.path.join(RESULTS_DIR, 'top3_list.pkl')
WINNER_PATH = os.path.join(RESULTS_DIR, 'winner_ego.pkl')

GAME_NAME = "ALE/SpaceInvaders-v5"

def load_champion():
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Config non trovato: {CONFIG_PATH}")
        return None, None

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)
    
    genome = None
    if os.path.exists(TOP3_PATH):
        print(f"📂 Carico il CAMPIONE ASSOLUTO dalla Top 3: {TOP3_PATH}")
        with open(TOP3_PATH, 'rb') as f:
            top_genomes = pickle.load(f)
            genome = top_genomes[0] # Prende il primo (il migliore)
    elif os.path.exists(WINNER_PATH):
        print(f"📂 Carico il vincitore singolo: {WINNER_PATH}")
        with open(WINNER_PATH, 'rb') as f:
            genome = pickle.load(f)
    else:
        print("❌ Nessun genoma trovato!")
        return None, None

    return genome, config

def main():
    # 1. Carica Rete
    genome, config = load_champion()
    if not genome: return

    print(f"🏆 Genoma Caricato - ID: {genome.key} | Fitness: {genome.fitness}")
    
    # Crea la RNN
    net = neat.nn.RecurrentNetwork.create(genome, config)

    # 2. Setup Ambiente (Render HUMAN per vedere la finestra di gioco)
    try:
        print("🎮 Avvio finestra di gioco...")
        env = OCAtari(GAME_NAME, mode="ram", hud=False, render_mode="human")
    except Exception as e:
        print(f"⚠️ Impossibile aprire finestra grafica: {e}")
        print("   Eseguo in modalità solo testo.")
        env = OCAtari(GAME_NAME, mode="ram", hud=False, render_mode=None)

    env = SpaceInvadersEgocentricWrapper(env, skip=4)
    
    # Seed fisso per vedere sempre la stessa partita (opzionale, metti None per random)
    obs, info = env.reset(seed=42) 
    
    print("🚀 Inizio simulazione...")
    time.sleep(2) # Pausa per prepararsi

    total_reward = 0.0
    steps = 0
    
    try:
        while True:
            # --- LOGICA NEAT ---
            outputs = net.activate(obs)
            action = np.argmax(outputs)
            
            # --- STEP AMBIENTE ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # --- VISUALIZZAZIONE DASHBOARD (Ogni frame o se c'è pericolo) ---
            max_sensor_val = np.max(obs[1:6])
            
            # Pulisci console e ristampa
            # Nota: Riduce il flickering stampando tutto in una volta
            os.system('cls' if os.name == 'nt' else 'clear') 
            
            print(f"🤖 AI PLAYING | Step: {steps} | Score: {total_reward}")
            print(f"🕹️  AZIONE SCELTA: {['🛑 NOOP', '🔥 FIRE', '➡️ RIGHT', '⬅️ LEFT'][action]}")
            
            # --- GRAFICA SENSORI ---
            # Player X
            p_pos = int(obs[0] * 20) # Scala 0-20 per la barra
            p_bar = "." * p_pos + "A" + "." * (20 - p_pos)
            print(f"\nPOSIZIONE: [{p_bar}] (Val: {obs[0]:.2f})")

            # Radar Proiettili
            bars = [" " if v == 0 else "█" * int(v * 10) for v in obs[1:6]]
            deltas = [f"{v:+.2f}" for v in obs[6:11]]
            
            print("\n📡 RADAR MINACCE (Altezza proiettili)")
            print(f"   SX2  [{bars[0]:<10}] {obs[1]:.2f}")
            print(f"   SX1  [{bars[1]:<10}] {obs[2]:.2f}")
            print(f"   CTR  [{bars[2]:<10}] {obs[3]:.2f}  <-- SOPRA PLAYER")
            print(f"   DX1  [{bars[3]:<10}] {obs[4]:.2f}")
            print(f"   DX2  [{bars[4]:<10}] {obs[5]:.2f}")

            # Target & UFO
            tgt_arrow = "⬆️" if abs(obs[11]) < 0.1 else ("⬅️" if obs[11] < 0 else "➡️")
            print(f"\n🎯 MIRA: {tgt_arrow} (Err: {obs[11]:+.2f})")
            
            if obs[18] > 0.5:
                print(f"🛸 UFO AVVISTATO! Pos: {obs[17]:+.2f}")

            # Rallenta per rendere leggibile la dashboard (e guardare il gioco)
            # 0.05s è circa 20fps, buono per seguire l'azione
            time.sleep(0.05)

            if terminated or truncated:
                print(f"\n💀 GAME OVER - Punteggio Finale: {total_reward}")
                break

    except KeyboardInterrupt:
        print("\n🛑 Interrotto dall'utente.")
    
    env.close()

if __name__ == "__main__":
    main()