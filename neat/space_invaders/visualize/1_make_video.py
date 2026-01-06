import sys
import os
import pickle
import numpy as np
import neat
import imageio
import gymnasium as gym
import ale_py 

# --- SETUP PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# --- CONFIGURAZIONE ---
CONFIG_PATH = os.path.join(project_root, 'config', 'config_baseline.txt')
BEST_GENOME_PATH = os.path.join(project_root, 'results', 'baseline_winner.pkl')
OUTPUT_DIR = os.path.join(project_root, 'results')

# --- MODIFICA ESTENSIONE FILE ---
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "baseline.mp4")

# --- PARAMETRI AMBIENTE ---
GAME_NAME = "SpaceInvadersNoFrameskip-v4"
FIXED_SEED = 42 

def make_video():
    # 1. Verifiche
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Config non trovato: {CONFIG_PATH}")
        return
    if not os.path.exists(BEST_GENOME_PATH):
        print(f"❌ Genoma non trovato in: {BEST_GENOME_PATH}")
        return

    # 2. Carica Config e Genoma
    print(f"📂 Caricamento config...")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)
    
    print(f"🏆 Caricamento CAMPIONE da: {BEST_GENOME_PATH}")
    with open(BEST_GENOME_PATH, 'rb') as f:
        genome = pickle.load(f)
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # 3. Setup Ambiente
    print(f"🎮 Inizializzazione {GAME_NAME} (RAM Mode)...")
    
    try:
        # render_mode="rgb_array" è essenziale per catturare i frame
        env = gym.make(GAME_NAME, obs_type="ram", render_mode="rgb_array")
    except Exception as e:
        print(f"❌ Errore ambiente: {e}")
        return
    
    # SEED FISSO
    observation, info = env.reset(seed=FIXED_SEED)
    print(f"🔒 Seed impostato a: {FIXED_SEED}")
    
    # CHECK DIMENSIONI INPUT
    input_len = len(observation)
    expected_len = config.genome_config.num_inputs
    if input_len != expected_len:
        print(f"❌ ERRORE CRITICO: La rete vuole {expected_len} input ma ne riceve {input_len}!")
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

    MAX_STEPS = 10000 

    while not (terminated or truncated) and steps < MAX_STEPS:
        # --- PRE-PROCESSING ---
        inputs = observation / 255.0
        
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
    
    if total_reward < 470:
        print("⚠️ ATTENZIONE: Lo score è diverso dal training.")

    # 4. Salvataggio VIDEO MP4
    if len(frames) > 0:
        print(f"💾 Salvataggio VIDEO ({len(frames)} frames) in: {OUTPUT_VIDEO}")
        try:
            # macro_block_size=1 evita errori se le dimensioni del video non sono divisibili per 16
            imageio.mimsave(OUTPUT_VIDEO, frames, fps=30, macro_block_size=1)
            print(f"🎉 Fatto! Video MP4 salvato.")
        except ImportError:
             print("❌ Errore: Manca ffmpeg. Esegui: pip install imageio[ffmpeg]")
        except Exception as e:
            print(f"❌ Errore salvataggio VIDEO: {e}")
    else:
        print("❌ Nessun frame catturato. Impossibile creare il video.")

if __name__ == '__main__':
    make_video()