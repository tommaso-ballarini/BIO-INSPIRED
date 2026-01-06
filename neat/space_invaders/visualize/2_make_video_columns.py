import sys
import os
import pickle
import numpy as np
import neat
import imageio
import gymnasium as gym

# Import OCAtari e il Wrapper specifico
try:
    from ocatari.core import OCAtari
except ImportError:
    print("❌ ERRORE: Libreria OCAtari non installata.")
    sys.exit(1)

# --- SETUP PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../visualize
project_root = os.path.dirname(current_dir)               # .../space_invaders (root del progetto)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import del Wrapper (Deve essere quello modificato con stack=3)
try:
    from wrapper.wrapper_si_columns import SpaceInvadersColumnWrapper
except ImportError:
    print("❌ ERRORE: Non trovo 'wrapper_si_columns.py' nella cartella wrapper!")
    sys.exit(1)

# --- CONFIGURAZIONE ---
# Assicurati che questi puntino ai file corretti della tua run FFNN
CONFIG_PATH = os.path.join(project_root, 'config', 'config_si_columns.txt')
RESULTS_DIR = os.path.join(project_root, 'results')
BEST_GENOME_PATH = os.path.join(RESULTS_DIR, 'columns_winner_ffnn.pkl')
OUTPUT_VIDEO = os.path.join(RESULTS_DIR, "columns_champion_ffnn.mp4")

# --- PARAMETRI AMBIENTE ---
GAME_NAME = "SpaceInvadersNoFrameskip-v4"
FIXED_SEED = 42 

def make_video():
    # 1. Verifiche esistenza file
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Config non trovato: {CONFIG_PATH}")
        return
    if not os.path.exists(BEST_GENOME_PATH):
        print(f"❌ Genoma non trovato in: {BEST_GENOME_PATH}")
        return

    # 2. Carica Config e Genoma
    print(f"📂 Caricamento config da: {CONFIG_PATH}")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)
    
    print(f"🏆 Caricamento CAMPIONE da: {BEST_GENOME_PATH}")
    with open(BEST_GENOME_PATH, 'rb') as f:
        genome = pickle.load(f)
    
    # IMPORTANTE: Usiamo FeedForwardNetwork come da tua ultima richiesta
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # 3. Setup Ambiente (CRUCIALE: Identico al Training)
    print(f"🎮 Inizializzazione {GAME_NAME} con Column Wrapper...")
    
    try:
        # OCAtari in modalità RAM, render_mode="rgb_array" per registrare video
        env = OCAtari(GAME_NAME, mode="ram", hud=False, render_mode="rgb_array")
        
        # --- MODIFICA FONDAMENTALE ---
        # Parametri allineati al training: n_columns=10, skip=4
        # Il wrapper (modificato prima) gestirà internamente lo Stack=3
        env = SpaceInvadersColumnWrapper(env, n_columns=10, skip=4)
        
    except Exception as e:
        print(f"❌ Errore inizializzazione ambiente: {e}")
        return
    
    # SEED FISSO per replicare la partita
    observation, info = env.reset(seed=FIXED_SEED)
    print(f"🔒 Seed impostato a: {FIXED_SEED}")
    
    # CHECK DIMENSIONI INPUT
    input_len = len(observation)
    expected_len = config.genome_config.num_inputs # Dovrebbe leggere 96 dal file txt
    
    print(f"ℹ️ Input Rete Rilevati: {input_len} | Attesi da Config: {expected_len}")
    
    if input_len != expected_len:
        print(f"❌ ERRORE CRITICO DIMENSIONI:")
        print(f"   Il Config si aspetta {expected_len} input.")
        print(f"   Il Wrapper ne sta fornendo {input_len}.")
        print(f"   -> Verifica di aver modificato 'config.txt' (num_inputs=96) e 'wrapper_si_columns.py' (maxlen=3).")
        return

    frames = []
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    
    print("🎥 Registrazione partita in corso...")
    
    # Cattura primo frame
    try:
        first_frame = env.render()
        if first_frame is not None:
            frames.append(first_frame)
    except Exception as e:
        print(f"❌ Errore render iniziale: {e}")

    MAX_STEPS = 5000 # Limite per evitare video infiniti se l'agente si blocca

    while not (terminated or truncated) and steps < MAX_STEPS:
        # --- PRE-PROCESSING ---
        inputs = observation 
        
        if isinstance(inputs, np.ndarray):
            inputs = inputs.flatten()

        # Attivazione Rete
        outputs = net.activate(inputs)
        action = np.argmax(outputs)
        
        # Step ambiente
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Cattura frame grafico
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        total_reward += reward
        steps += 1
        
        if steps % 100 == 0:
            print(f"\rStep: {steps} | Reward: {total_reward:.1f}", end="")
            
    env.close()
    print(f"\n✅ Partita finita! Score Finale: {total_reward}")
    
    # 4. Salvataggio VIDEO MP4
    if len(frames) > 0:
        print(f"💾 Salvataggio VIDEO ({len(frames)} frames) in: {OUTPUT_VIDEO}")
        try:
            imageio.mimsave(OUTPUT_VIDEO, frames, fps=30, macro_block_size=1)
            print(f"🎉 Fatto! Video salvato correttamente.")
        except ImportError:
             print("❌ Errore: Manca imageio-ffmpeg. Esegui: pip install imageio[ffmpeg]")
        except Exception as e:
            print(f"❌ Errore salvataggio VIDEO: {e}")
    else:
        print("❌ Nessun frame catturato.")

if __name__ == '__main__':
    make_video()