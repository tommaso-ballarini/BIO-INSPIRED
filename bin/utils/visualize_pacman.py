import os
import sys
import pickle
import numpy as np
import neat
import gymnasium as gym
import imageio
from itertools import chain

# --- Imposta i percorsi base ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- MONKEY PATCH OCAtari (FONDAMENTALE PER PACMAN) ---
import ocatari.core
def patched_ns_state(self):
    valid_objects = [o for o in self.objects if o is not None and hasattr(o, '_nsrepr')]
    return list(chain.from_iterable([o._nsrepr for o in valid_objects]))
ocatari.core.OCAtari.ns_state = patched_ns_state
# ------------------------------------------------------

from ocatari.core import OCAtari
from core.wrappers import PacmanHybridWrapper # Assicurati che il nome coincida con la tua classe nel file

def get_latest_file(folder, prefix):
    candidates = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith('.pkl')]
    if not candidates:
        return None
    candidates.sort(
        key=lambda f: os.path.getmtime(os.path.join(folder, f)), reverse=True
    )
    return os.path.join(folder, candidates[0])

def save_gif(frames, path, fps=15): # 15 o 30 fps va bene
    print(f"💾 Salvataggio replay: {path} ...")
    try:
        # Ottimizzazione: salviamo solo ogni 2 frame se sono troppi, per ridurre dimensione file
        if len(frames) > 1000:
            frames = frames[::2]
        imageio.mimsave(path, frames, fps=fps, loop=0)
        print("✅ GIF salvata!")
    except Exception as e:
        print(f"❌ Errore salvataggio GIF: {e}")

def visualize_neat(env_name, result_dir, config_path, max_steps=2000, num_episodes=1):
    # CORREZIONE 1: Cerca il file "pacman_winner"
    best_genome_file = get_latest_file(result_dir, "pacman_winner")
    
    if best_genome_file is None:
        # Fallback: prova con "best_genome_neat" se non trova l'altro
        best_genome_file = get_latest_file(result_dir, "best_genome_neat")
        
    if best_genome_file is None:
        print(f"❌ Nessun genoma trovato in {result_dir}")
        return

    print(f"✅ Caricamento genoma: {os.path.basename(best_genome_file)}")
    with open(best_genome_file, "rb") as f:
        winner = pickle.load(f)

    # Carica Config
    neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                              neat.DefaultSpeciesSet, neat.DefaultStagnation,
                              config_path)

    # Crea Rete
    net = neat.nn.FeedForwardNetwork.create(winner, neat_config)

    # SETUP AMBIENTE
    print(f"🎮 Creazione ambiente: {env_name}")
    env = OCAtari(env_name, mode="ram", obs_mode="obj", render_mode="rgb_array")
    
    # WRAPPER (Assicurati che n_vector_features nel wrapper corrisponda al config caricato!)
    # Se hai usato la versione con velocità (86 input), il wrapper deve essere quello aggiornato.
    env = PacmanHybridWrapper(env) 

    print("=" * 70)
    print(f"🎬 AVVIO EPISODIO CAMPIONE")
    print("=" * 70)

    scores = []

    for episode in range(num_episodes):
        observation, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        frames = [] 

        print(f"Giocando episodio {episode + 1}...")

        while not done and steps < max_steps:
            # 1. Cattura Frame
            # OCAtari .render() da l'immagine pulita. 
            frame = env.render() 
            if frame is not None:
                # Se OCAtari ritorna una lista (succede in alcune versioni), prendi il primo elemento
                if isinstance(frame, list):
                    frame = frame[0]
                frames.append(frame)

            # 2. Azione Rete
            output = net.activate(observation)
            action = np.argmax(output)

            # 3. Step fisico
            try:
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
            except Exception as e:
                print(f"⚠️ Errore durante step: {e}")
                break

        scores.append(total_reward)
        print(f"🏁 Finito. Score: {total_reward}. Frames: {len(frames)}")
        
        # Salva GIF
        gif_name = f"champion_replay_{os.path.basename(best_genome_file)[:15]}_score{int(total_reward)}.gif"
        gif_path = os.path.join(result_dir, gif_name)
        save_gif(frames, gif_path)

    env.close()

if __name__ == "__main__":
    # CORREZIONE 2: Nome ambiente corretto
    ENV_NAME = "Pacman" 
    
    RESULT_DIR = os.path.join(PROJECT_ROOT, "evolution_results")
    CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "neat_pacman_config.txt")
    
    # Assicurati che imageio sia installato
    try:
        import imageio
    except ImportError:
        print("Installazione imageio...")
        os.system("pip install imageio")
        import imageio

    visualize_neat(ENV_NAME, RESULT_DIR, CONFIG_PATH)