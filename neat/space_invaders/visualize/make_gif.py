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

# Aggiungiamo la root al system path
if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORT WRAPPER ---
try:
    from wrapper.wrapper_si_grid import SpaceInvadersGridWrapper
except ImportError as e:
    print(f"❌ Errore Importazione: {e}")
    sys.exit(1)

# --- CONFIGURAZIONE PERCORSI ---
# IMPORTANTE: Deve essere lo stesso usato nel training per coerenza fisica
GAME_NAME = "SpaceInvadersNoFrameskip-v4" 

# Configurazione
CONFIG_PATH = os.path.join(project_root, 'config', 'config_si_wrapper.txt')

# ⚠️ CARICHIAMO IL BEST EVER (Il record assoluto salvato in results)
BEST_GENOME_PATH = os.path.join(project_root, 'results', 'best_ever_champion.pkl')

# Output Directory
OUTPUT_DIR = os.path.join(project_root, 'results')
OUTPUT_GIF = os.path.join(OUTPUT_DIR, "best_agent_gameplay.gif")

def make_gif():
    # 1. Verifica esistenza file
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Config non trovato: {CONFIG_PATH}")
        return
    if not os.path.exists(BEST_GENOME_PATH):
        print(f"❌ Genoma non trovato: {BEST_GENOME_PATH}")
        print(f"   (Hai controllato nella cartella results?)")
        return

    # 2. Carica Config e Genoma
    print(f"📂 Caricamento config...")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)
    
    print(f"🏆 Caricamento CAMPIONE ASSOLUTO da: {BEST_GENOME_PATH}")
    with open(BEST_GENOME_PATH, 'rb') as f:
        genome = pickle.load(f)
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # 3. Setup Ambiente
    print(f"🎮 Inizializzazione {GAME_NAME}...")
    
    # Render_mode='rgb_array' serve per il video
    env = OCAtari(GAME_NAME, mode="ram", hud=False, render_mode="rgb_array")
    
    # IMPORTANTE: skip=4 deve corrispondere al training!
    env = SpaceInvadersGridWrapper(env, grid_shape=(16, 16), skip=4)
    
    observation, _ = env.reset()
    
    frames = []
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    
    print("🎥 Registrazione partita in corso...")
    
    # Aggiungi il primo frame
    frames.append(env.render())

    while not (terminated or truncated):
        # Logica Rete
        inputs = observation
        outputs = net.activate(inputs)
        action = np.argmax(outputs)
        
        # Step (Il wrapper esegue 4 frame interni)
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Cattura il frame DOPO l'azione
        # Nota: Poiché il wrapper salta 4 frame, la GIF sarà un po' "scattosa" (time-lapse).
        # È normale con questo tipo di training veloce.
        frame = env.render()
        frames.append(frame)
        
        total_reward += reward
        steps += 1
        
        if steps % 10 == 0: # Print meno frequente
            print(f"\rStep: {steps} | Reward: {total_reward}", end="")
            
    env.close()
    print(f"\n✅ Partita finita! Score Finale: {total_reward}")
    
    # 4. Salvataggio GIF
    print(f"💾 Salvataggio GIF in: {OUTPUT_GIF}")
    try:
        # fps=15 rende il video guardabile (visto che mancano i frame intermedi dello skip)
        imageio.mimsave(OUTPUT_GIF, frames, fps=15) 
        print(f"🎉 Fatto! GIF salvata.")
    except Exception as e:
        print(f"❌ Errore GIF: {e}")

if __name__ == '__main__':
    make_gif()