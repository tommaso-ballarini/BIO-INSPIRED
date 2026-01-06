import sys
import os
import pickle
import numpy as np
import neat
import imageio
import gymnasium as gym

# Import OCAtari
try:
    from ocatari.core import OCAtari
except ImportError:
    print("❌ ERRORE: Libreria OCAtari non installata.")
    sys.exit(1)

# --- GESTIONE PERCORSI ---
# Posizione attuale: .../neat/space_invaders/visualize/
current_dir = os.path.dirname(os.path.abspath(__file__))

# Risaliamo di 1 livello per arrivare alla root del progetto (es. space_invaders/)
# Questa è la cartella che contiene "wrapper", "config" e "results"
space_invaders_root = os.path.dirname(current_dir)

if space_invaders_root not in sys.path:
    sys.path.append(space_invaders_root)

print(f"📂 Root Space Invaders impostata a: {space_invaders_root}")

# --- IMPORT DEL WRAPPER ---
try:
    # Importiamo lo stesso wrapper usato nel training
    from wrapper.wrapper_si_columns_RNN import SpaceInvadersColumnWrapper
    print("✅ Wrapper importato correttamente.")
except ImportError as e:
    print(f"❌ ERRORE IMPORT: {e}")
    print(f"   Controlla che esista il file in: wrapper/wrapper_si_columns_RNN.py")
    sys.exit(1)

# --- CONFIGURAZIONI ---
CONFIG_PATH = os.path.join(space_invaders_root, 'config', 'config_si_columns_RNN.txt')
RESULTS_DIR = os.path.join(space_invaders_root, 'results')

# MODIFICA: Nome del file salvato nel training script (_fit)
BEST_GENOME_PATH = os.path.join(RESULTS_DIR, 'columns_winner_RNN_fit.pkl')
# MODIFICA: Nome video specifico per questa run
OUTPUT_VIDEO = os.path.join(RESULTS_DIR, 'columns_champion_rnn_fit.mp4')

GAME_NAME = "SpaceInvadersNoFrameskip-v4"
FIXED_SEED = 42 

def make_video():
    # 1. Verifiche Preliminari
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Config non trovato: {CONFIG_PATH}")
        return
    if not os.path.exists(BEST_GENOME_PATH):
        print(f"❌ Genoma non trovato: {BEST_GENOME_PATH}")
        print("   Hai completato il training 'run_columns_RNN_fit'?")
        return

    # 2. Carica Config e Genoma
    print(f"📂 Caricamento config...")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)
    
    print(f"🏆 Caricamento CAMPIONE da: {BEST_GENOME_PATH}")
    with open(BEST_GENOME_PATH, 'rb') as f:
        genome = pickle.load(f)
    
    # --- CREAZIONE RETE (CRUCIALE: RecurrentNetwork) ---
    print("🧠 Creazione Rete Ricorrente (RNN)...")
    net = neat.nn.RecurrentNetwork.create(genome, config)
    
    # 3. Setup Ambiente
    print(f"🎮 Inizializzazione {GAME_NAME} con Column Wrapper...")
    
    try:
        # Render mode "rgb_array" serve per catturare i frame del video
        env = OCAtari(GAME_NAME, mode="ram", hud=False, render_mode="rgb_array")
        
        # Applichiamo lo stesso Wrapper e Skip del training
        env = SpaceInvadersColumnWrapper(env, n_columns=10, skip=4)
        
    except Exception as e:
        print(f"❌ Errore ambiente: {e}")
        return
    
    # Reset con Seed Fisso
    observation, info = env.reset(seed=FIXED_SEED)
    print(f"🔒 Seed impostato a: {FIXED_SEED}")
    
    # Controllo Dimensioni Input
    if len(observation) != 32:
        print(f"⚠️ ERRORE CRITICO DIMENSIONI: Atteso 32, ricevuto {len(observation)}")
        print("   Probabile causa: Cache Python o Wrapper non aggiornato.")
        env.close()
        return

    frames = []
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    
    print("🎥 Registrazione partita in corso...")
    
    # Cattura frame iniziale
    try:
        first_frame = env.render()
        if first_frame is not None:
            frames.append(first_frame)
    except Exception:
        pass

    MAX_STEPS = 5000 

    # --- GAME LOOP ---
    while not (terminated or truncated) and steps < MAX_STEPS:
        inputs = observation 
        
        # Attivazione Rete
        outputs = net.activate(inputs)
        action = np.argmax(outputs)
        
        # Step Ambiente
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Cattura Frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        total_reward += reward
        steps += 1
        
        if steps % 100 == 0:
            print(f"\rStep: {steps} | Reward: {total_reward:.1f}", end="")
            
    env.close()
    print(f"\n✅ Partita finita! Score Finale: {total_reward}")
    
    # 4. Salvataggio VIDEO
    if len(frames) > 0:
        print(f"💾 Salvataggio VIDEO ({len(frames)} frames) in: {OUTPUT_VIDEO}")
        try:
            imageio.mimsave(OUTPUT_VIDEO, frames, fps=30, macro_block_size=1)
            print(f"🎉 Fatto! Video salvato correttamente.")
        except ImportError:
             print("❌ Errore: Manca ffmpeg. Prova: pip install imageio[ffmpeg]")
        except Exception as e:
            print(f"❌ Errore salvataggio VIDEO: {e}")
    else:
        print("❌ Nessun frame catturato.")

if __name__ == '__main__':
    make_video()