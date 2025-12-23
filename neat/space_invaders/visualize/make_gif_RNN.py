import sys
import os
import pickle
import numpy as np
import neat
import gymnasium as gym
from ocatari.core import OCAtari
import imageio

# --- SETUP PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORT WRAPPER ---
try:
    from wrapper.wrapper_si_grid import SpaceInvadersGridWrapper
except ImportError as e:
    print(f"❌ Errore Importazione: {e}")
    sys.exit(1)

# --- CONFIGURAZIONE SPECIFICA PER QUESTA RUN (RNN) ---
GAME_NAME = "SpaceInvadersNoFrameskip-v4" 

# 1. CARTELLA RISULTATI (Aggiornata alla run RNN)
RESULTS_FOLDER_NAME = "results_rnn_grid16x16" 

CONFIG_PATH = os.path.join(project_root, 'config', 'config_si_wrapper.txt')
BEST_GENOME_PATH = os.path.join(project_root, RESULTS_FOLDER_NAME, 'best_ever_champion.pkl')
OUTPUT_DIR = os.path.join(project_root, RESULTS_FOLDER_NAME)
OUTPUT_GIF = os.path.join(OUTPUT_DIR, "best_rnn_agent.gif")

def make_gif():
    # Verifica file
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Config non trovato: {CONFIG_PATH}")
        return
    if not os.path.exists(BEST_GENOME_PATH):
        print(f"❌ Genoma non trovato: {BEST_GENOME_PATH}")
        print(f"   (Verifica di aver salvato il best_ever_champion.pkl nella cartella corretta)")
        return

    # Carica Config e Genoma
    print(f"📂 Caricamento config...")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)
    
    print(f"🏆 Caricamento CAMPIONE RNN da: {BEST_GENOME_PATH}")
    with open(BEST_GENOME_PATH, 'rb') as f:
        genome = pickle.load(f)
    
    # === MODIFICA CRITICA PER RNN ===
    # Usa RecurrentNetwork invece di FeedForwardNetwork
    net = neat.nn.RecurrentNetwork.create(genome, config)
    # ================================
    
    # Setup Ambiente
    print(f"🎮 Inizializzazione {GAME_NAME} (Grid 16x16, Skip 4)...")
    env = OCAtari(GAME_NAME, mode="ram", hud=False, render_mode="rgb_array")
    
    # Parametri allineati alla run di training
    env = SpaceInvadersGridWrapper(env, grid_shape=(16, 16), skip=4)
    
    observation, _ = env.reset()
    
    frames = []
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    
    print("🎥 Registrazione partita in corso...")
    frames.append(env.render()) # Primo frame

    while not (terminated or truncated):
        # La RNN gestisce la memoria (stato nascosto) automaticamente dentro .activate()
        inputs = observation
        outputs = net.activate(inputs)
        action = np.argmax(outputs)
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Cattura frame per la GIF
        frame = env.render()
        frames.append(frame)
        
        total_reward += reward
        steps += 1
        
        if steps % 10 == 0:
            print(f"\rStep: {steps} | Reward: {total_reward}", end="")
            
    env.close()
    print(f"\n✅ Partita finita! Score Finale: {total_reward}")
    
    # Salvataggio
    print(f"💾 Salvataggio GIF in: {OUTPUT_GIF}")
    try:
        imageio.mimsave(OUTPUT_GIF, frames, fps=15) 
        print(f"🎉 Fatto! GIF salvata.")
    except Exception as e:
        print(f"❌ Errore GIF: {e}")

if __name__ == '__main__':
    make_gif()