# FILE: utils/visualize_agent_freeway.py

import os
import sys
import pickle
import numpy as np
import neat
import gymnasium as gym
from PIL import Image
import datetime

# --- Imposta i percorsi base ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.evaluator import run_game_simulation

# --- Utility per trovare il file più recente ---
def get_latest_file(folder, prefix):
    """Ritorna il file più recente in 'folder' che inizia con 'prefix'."""
    candidates = [f for f in os.listdir(folder) if f.startswith(prefix)]
    if not candidates:
        return None
    candidates.sort(
        key=lambda f: os.path.getmtime(os.path.join(folder, f)), reverse=True
    )
    return os.path.join(folder, candidates[0])


def save_frames_as_gif(frames, path, duration=50):
    """
    Salva una lista di frame come GIF animata.
    
    Args:
        frames: lista di numpy arrays (frames RGB)
        path: percorso dove salvare la GIF
        duration: durata di ogni frame in millisecondi
    """
    # Converti i frame numpy in PIL Images
    pil_frames = [Image.fromarray(frame) for frame in frames]
    
    # Salva come GIF
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0  # loop infinito
    )
    print(f"💾 GIF salvata in: {path}")


def run_game_with_recording(agent_func, env_name, max_steps=1500):
    """
    Esegue il gioco registrando i frame per creare una GIF.
    
    Args:
        agent_func: funzione che prende game_state e ritorna un'azione
        env_name: nome dell'ambiente Gymnasium
        max_steps: numero massimo di step
    
    Returns:
        fitness, metrics, frames
    """
    # Crea ambiente con render_mode="rgb_array" per catturare i frame
    env = gym.make(env_name, render_mode="rgb_array")
    
    frames = []
    total_reward = 0
    
    obs, info = env.reset()
    
    for step in range(max_steps):
        # Cattura il frame corrente
        frame = env.render()
        frames.append(frame)
        
        # Ottieni RAM per l'agente
        ram = env.unwrapped.ale.getRAM()
        
        # Decidi l'azione
        action = agent_func(ram)
        
        # Esegui l'azione
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    env.close()
    
    metrics = {
        "total_reward": total_reward,
        "steps": step + 1
    }
    
    return total_reward, metrics, frames


def visualize_neat(env_name, result_dir, config_path, max_steps=1500, save_gif=True):
    """Carica e visualizza il miglior agente NEAT in simulazione."""
    best_genome_file = get_latest_file(result_dir, "best_genome_neat_")
    if best_genome_file is None:
        print(f"❌ Nessun file 'best_genome_neat_*.pkl' trovato in {result_dir}")
        return

    print(f"✅ Genoma NEAT trovato: {best_genome_file}")

    # Carica il genoma vincitore
    with open(best_genome_file, "rb") as f:
        winner = pickle.load(f)

    # Carica il file di configurazione NEAT
    if not os.path.isfile(config_path):
        print(f"❌ Config NEAT non trovata: {config_path}")
        return

    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Ricostruisci la rete vincitrice
    net = neat.nn.FeedForwardNetwork.create(winner, neat_config)

    # Definisci la funzione agente
    def agent_decision_function(game_state):
        features = game_state / 255.0
        output = net.activate(features)
        return int(np.argmax(output))

    print("🎮 Avvio simulazione con il vincitore NEAT...")

    # Esegui con registrazione
    fitness, metrics, frames = run_game_with_recording(
        agent_func=agent_decision_function,
        env_name=env_name,
        max_steps=max_steps
    )

    print(f"🎯 Fitness episodio: {fitness}")
    if metrics is not None:
        print(f"📊 Metrics: {metrics}")

    # Salva la GIF se richiesto
    if save_gif and len(frames) > 0:
        timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        gif_path = os.path.join(result_dir, f"neat_gameplay_{timestamp_str}.gif")
        
        # Campiona i frame se sono troppi (per ridurre dimensione file)
        # Prendi un frame ogni 2 per mantenere la GIF fluida ma non troppo grande
        sampled_frames = frames[::2]
        
        save_frames_as_gif(sampled_frames, gif_path, duration=50)
        print(f"🎬 Numero di frame: {len(frames)} (campionati: {len(sampled_frames)})")


if __name__ == "__main__":
    ENV_NAME = "ALE/Freeway-v5"
    RESULT_DIR = os.path.join(PROJECT_ROOT, "evolution_results")
    CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "neat_freeway_config.txt")

    if not os.path.isdir(RESULT_DIR):
        print(f"❌ Cartella {RESULT_DIR} non trovata. Esegui prima l'evoluzione NEAT.")
        sys.exit(1)

    visualize_neat(
        env_name=ENV_NAME,
        result_dir=RESULT_DIR,
        config_path=CONFIG_PATH,
        max_steps=1500,
        save_gif=True  # Imposta a False per disabilitare il salvataggio della GIF
    )