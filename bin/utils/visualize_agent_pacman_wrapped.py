import os
import sys
import pickle
import numpy as np
import neat
import gymnasium as gym
import cv2
import imageio # Assicurati di averlo: pip install imageio

# --- Imposta i percorsi base ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import imageio
from itertools import chain # Importa chain, se non è già importato

# --- MONKEY PATCH OCATARI (CRITICO) ---
# Fix per bug noto di OCAtari con oggetti None
import ocatari.core

def patched_ns_state(self):
    """Filtra gli oggetti None prima di tentare l'accesso a _nsrepr."""
    valid_objects = [o for o in self.objects if o is not None and hasattr(o, '_nsrepr')]
    return list(chain.from_iterable([o._nsrepr for o in valid_objects]))

ocatari.core.OCAtari.ns_state = patched_ns_state
from ocatari.core import OCAtari
# 💡 MODIFICA 1: Importa il wrapper corretto (era HybridPacmanWrapper)
from core.wrappers_pacman import PacmanFeatureWrapper 

def get_latest_file(folder, prefix):
     candidates = [f for f in os.listdir(folder) if f.startswith(prefix)]
     if not candidates:
         return None
     candidates.sort(
         key=lambda f: os.path.getmtime(os.path.join(folder, f)), reverse=True
     )
     return os.path.join(folder, candidates[0])

def save_gif(frames, path, fps=30):
     """Salva una lista di frame come GIF."""
     print(f"💾 Salvataggio replay in corso: {path} ...")
     imageio.mimsave(path, frames, fps=fps, loop=0)
     print("✅ Replay salvato!")

def visualize_neat(env_name, result_dir, config_path, max_steps=3000, num_episodes=3):
     # 💡 MODIFICA 2: Cerca il file "winner_" (come salvato in run_pacman_neat.py)
     best_genome_file = get_latest_file(result_dir, "winner_")
     if best_genome_file is None:
         print(f"❌ Nessun genoma 'winner_*.pkl' trovato in {result_dir}")
         return

     print(f"✅ Caricamento genoma: {os.path.basename(best_genome_file)}")
     # Il file winner può contenere (winner, config), carichiamo solo il genoma
     with open(best_genome_file, "rb") as f:
         data = pickle.load(f)
         # Tenta di estrarre il genoma e la config in base al formato salvato
         if isinstance(data, tuple) and len(data) == 2:
            winner, _ = data 
         else:
            winner = data

     neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                          config_path)

     net = neat.nn.FeedForwardNetwork.create(winner, neat_config)

     # SETUP AMBIENTE
     env = OCAtari(env_name, mode="ram", obs_mode="obj", render_mode="rgb_array")
     if hasattr(env.unwrapped, 'ale'):
         env.unwrapped.ale.setFloat('repeat_action_probability', 0.0)
    
     # WRAPPER
     # 💡 MODIFICA 3: Usa PacmanFeatureWrapper con le dimensioni usate nel training
     env = PacmanFeatureWrapper(env, grid_rows=10, grid_cols=10)

     print("=" * 70)
     print(f"🎬 AVVIO REGISTRAZIONE ({num_episodes} episodi)")
     print("=" * 70)

     scores = []

     for episode in range(num_episodes):
         # L'oggetto env ora è il wrapper, chiama reset su di esso
         observation, info = env.reset()
         done = False
         total_reward = 0
         steps = 0
         frames = [] 

         print(f"\nRec Episodio {episode + 1}...")

         while not done and steps < max_steps:
            # 1. Cattura Frame per il video (dall'ambiente base non wrappato)
            frame = env.unwrapped.render() 
            if frame is not None:
              frames.append(frame)

            # 2. Decisione Rete
            output = net.activate(observation)
            action = np.argmax(output)

            # 3. Step
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

         scores.append(total_reward)
         print(f"🏁 Finito. Score: {total_reward}. Frames catturati: {len(frames)}")
    
         # Salva la GIF
         if total_reward > 0: # Salva solo se c'è un punteggio significativo
            gif_path = os.path.join(result_dir, f"pacman_replay_ep{episode+1}_score{int(total_reward)}.gif")
            save_gif(frames, gif_path)

     env.close()
     print("\n" + "=" * 70)
     print(f"📊 Media Punteggio: {np.mean(scores):.2f}")
     print("=" * 70)

if __name__ == "__main__":
     # 💡 MODIFICA 4: Nome ambiente OCAtari
     ENV_NAME = "Pacman" 
     # 💡 MODIFICA 5: Cartella risultati usata da run_pacman_neat.py
     RESULT_DIR = os.path.join(PROJECT_ROOT, "evolution_results", "pacman") 
     CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "neat_pacman_config.txt")
    
     # Il max_steps per la visualizzazione dovrebbe corrispondere al max_steps di training (3000)
     visualize_neat(ENV_NAME, RESULT_DIR, CONFIG_PATH, max_steps=3000, num_episodes=2)