import sys
import os
# Rimosso import time che non usiamo più nel loop
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
        print(f"📂 Carico TOP 1: {TOP3_PATH}")
        with open(TOP3_PATH, 'rb') as f:
            top_genomes = pickle.load(f)
            genome = top_genomes[0] 
    elif os.path.exists(WINNER_PATH):
        print(f"📂 Carico Winner: {WINNER_PATH}")
        with open(WINNER_PATH, 'rb') as f:
            genome = pickle.load(f)
    else:
        print("❌ Nessun file trovato!")
        return None, None

    return genome, config

def main():
    genome, config = load_champion()
    if not genome: return

    # --- CREAZIONE RETE (RNN) ---
    try:
        net = neat.nn.RecurrentNetwork.create(genome, config)
    except Exception as e:
        print(f"❌ Errore rete: {e}")
        return

    # --- SETUP AMBIENTE ---
    # render_mode="human" è limitato dal V-Sync del monitor (solitamente 60fps).
    # Non si può andare molto più veloci di così se si vuole vedere la grafica,
    # a meno che non si disabiliti il rendering (render_mode=None).
    try:
        env = OCAtari(GAME_NAME, mode="ram", hud=False, render_mode="human")
    except Exception as e:
        env = OCAtari(GAME_NAME, mode="ram", hud=False, render_mode=None)

    env = SpaceInvadersEgocentricWrapper(env, skip=4)
    obs, info = env.reset(seed=42) 
    
    net.reset() # Reset memoria RNN
    
    print("🚀 TURBO MODE ATTIVA (Premi CTRL+C per uscire)")
    
    total_reward = 0.0
    steps = 0
    
    try:
        while True:
            # 1. Decisione Rete (Ultra veloce)
            outputs = net.activate(obs)
            action = np.argmax(outputs)
            
            # 2. Step Fisico
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # 3. Output Console (Raramente, per non rallentare)
            if steps % 60 == 0:
                print(f"\r⚡ Step: {steps} | Score: {total_reward:.1f}", end="")
            
            # RIMOSSO time.sleep()
            # Il loop ora gira alla massima velocità permessa dal rendering grafico.

            if terminated or truncated:
                print(f"\n💀 FINE. Score: {total_reward}")
                break

    except KeyboardInterrupt:
        print("\n🛑 Stop.")
    
    env.close()

if __name__ == "__main__":
    main()