import sys
import os
import pickle
import numpy as np
import neat
import imageio
from ocatari.core import OCAtari

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.append(project_root)

from wrapper.wrapper_si_ego import SpaceInvadersEgocentricWrapper

CONFIG_PATH = os.path.join(project_root, 'config', 'config_si_ego.txt')
RESULTS_DIR = os.path.join(project_root, 'results')

# Percorsi possibili
TOP3_PATH = os.path.join(RESULTS_DIR, 'top3_list.pkl')
WINNER_PATH = os.path.join(RESULTS_DIR, 'winner_ego.pkl')

GAME_NAME = "SpaceInvadersNoFrameskip-v4"
TEST_SEED = 42 

def make_videos():
    # 1. Carica Configurazione
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Config non trovato: {CONFIG_PATH}")
        return

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)
    
    # 2. LOGICA DI CARICAMENTO INTELLIGENTE
    top_genomes = []
    
    if os.path.exists(TOP3_PATH):
        print(f"📂 Trovata lista Top 3: {TOP3_PATH}")
        with open(TOP3_PATH, 'rb') as f:
            top_genomes = pickle.load(f)
    elif os.path.exists(WINNER_PATH):
        print(f"⚠️ Top 3 non trovata. Carico solo il Vincitore singolo: {WINNER_PATH}")
        with open(WINNER_PATH, 'rb') as f:
            winner = pickle.load(f)
            top_genomes = [winner] # Lo mettiamo in una lista per uniformare il codice
    else:
        print(f"❌ NESSUN FILE TROVATO! Manca sia {TOP3_PATH} che {WINNER_PATH}")
        return

    # 3. Generazione Video
    for rank, genome in enumerate(top_genomes):
        rank_idx = rank + 1
        print(f"\n🎬 Generazione video per RANK #{rank_idx} (Fitness: {genome.fitness})")
        
        # IMPORTANTE: Se hai usato run_training.py vecchio (FFNN), qui devi mettere FeedForwardNetwork.
        # Se hai usato il nuovo run_training_rnn.py, devi mettere RecurrentNetwork.
        # Metto RecurrentNetwork come default per il nuovo obiettivo.
        try:
            net = neat.nn.RecurrentNetwork.create(genome, config)
        except Exception:
            print("⚠️ Errore creazione RNN, provo con FeedForward...")
            net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        try:
            env = OCAtari(GAME_NAME, mode="ram", hud=False, render_mode="rgb_array")
            env = SpaceInvadersEgocentricWrapper(env, skip=4)
        except Exception as e:
            print(e); return

        # Seed fisso per il replay
        obs, _ = env.reset(seed=TEST_SEED)
        
        frames = []
        total_reward = 0.0
        terminated = False
        truncated = False
        steps = 0
        
        frames.append(env.render())
        
        while not (terminated or truncated) and steps < 6000:
            outputs = net.activate(obs)
            action = np.argmax(outputs)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            frames.append(env.render())

        env.close()
        print(f"   -> Score partita video: {total_reward}")
        
        video_name = os.path.join(RESULTS_DIR, f'video_rank_{rank_idx}_score_{int(total_reward)}.mp4')
        imageio.mimsave(video_name, frames, fps=30, macro_block_size=1)
        print(f"   ✅ Video salvato: {video_name}")

if __name__ == '__main__':
    make_videos()