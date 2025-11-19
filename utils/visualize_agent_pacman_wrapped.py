import os
import sys
import pickle
import numpy as np
import neat
import gymnasium as gym
import cv2
import imageio  # Assicurati di averlo: pip install imageio

# --- Imposta i percorsi base ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ocatari.core import OCAtari
from core.wrappers import HybridPacmanWrapper

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

def visualize_neat(env_name, result_dir, config_path, max_steps=1000, num_episodes=3):
    best_genome_file = get_latest_file(result_dir, "best_genome_neat_")
    if best_genome_file is None:
        print(f"❌ Nessun genoma trovato in {result_dir}")
        return

    print(f"✅ Caricamento genoma: {os.path.basename(best_genome_file)}")
    with open(best_genome_file, "rb") as f:
        winner = pickle.load(f)

    neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                              neat.DefaultSpeciesSet, neat.DefaultStagnation,
                              config_path)

    net = neat.nn.FeedForwardNetwork.create(winner, neat_config)

    # SETUP AMBIENTE
    # render_mode="rgb_array" è essenziale per registrare il video
    env = OCAtari(env_name, mode="ram", obs_mode="obj", render_mode="rgb_array")
    if hasattr(env.unwrapped, 'ale'):
        env.unwrapped.ale.setFloat('repeat_action_probability', 0.0)
    
    # WRAPPER
    env = HybridPacmanWrapper(env, grid_rows=8, grid_cols=8)

    print("=" * 70)
    print(f"🎬 AVVIO REGISTRAZIONE ({num_episodes} episodi)")
    print("=" * 70)

    scores = []

    for episode in range(num_episodes):
        observation, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        frames = [] # Lista per i frame del video

        print(f"\nRec Episodio {episode + 1}...")

        while not done and steps < max_steps:
            # 1. Cattura Frame per il video (dall'ambiente base)
            # OCAtari render() restituisce l'immagine processata o originale a seconda della config
            # Usiamo env.unwrapped.render() per sicurezza
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
        
        # Salva la GIF solo del primo (o del migliore) episodio per risparmiare spazio
        if episode == 0 or total_reward > np.mean(scores):
            gif_path = os.path.join(result_dir, f"pacman_replay_ep{episode+1}_score{int(total_reward)}.gif")
            save_gif(frames, gif_path)

    env.close()
    print("\n" + "=" * 70)
    print(f"📊 Media Punteggio: {np.mean(scores):.2f}")
    print("=" * 70)

if __name__ == "__main__":
    # Installa imageio se manca: pip install imageio
    ENV_NAME = "MsPacman-v4"
    RESULT_DIR = os.path.join(PROJECT_ROOT, "evolution_results")
    CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "neat_pacman_config.txt")
    
    visualize_neat(ENV_NAME, RESULT_DIR, CONFIG_PATH, num_episodes=3)