import os
import sys
import pickle
import importlib.util
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# --- Setup Percorsi ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Assicurati di avere il wrapper disponibile
try:
    from core.wrappers import FreewayOCAtariWrapper
except ImportError:
    print("❌ Errore: Impossibile importare FreewayOCAtariWrapper da core.wrappers")
    sys.exit(1)

# --- Configurazione ---
HISTORY_DIR = os.path.join(PROJECT_ROOT, "evolution_history")
BEST_AGENT_FILE = os.path.join(HISTORY_DIR, "best_gp_agent.py")
LOGBOOK_FILE = os.path.join(HISTORY_DIR, "gp_logbook.pkl")
VIDEO_DIR = os.path.join(HISTORY_DIR, "videos")

def load_agent_module(agent_path):
    """Carica dinamicamente lo script python dell'agente."""
    if not os.path.exists(agent_path):
        raise FileNotFoundError(f"Agente non trovato: {agent_path}")
    
    spec = importlib.util.spec_from_file_location("evolved_agent", agent_path)
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    return agent_module.get_action

def plot_evolution_stats(logbook_path):
    """Genera i grafici dell'evoluzione basati sul Logbook DEAP."""
    if not os.path.exists(logbook_path):
        print(f"⚠️ File di log '{logbook_path}' non trovato. Salta grafici.")
        return

    with open(logbook_path, "rb") as f:
        logbook = pickle.load(f)

    gen = logbook.select("gen")
    fit_mins = logbook.select("min")
    fit_maxs = logbook.select("max")
    fit_avgs = logbook.select("avg")

    plt.figure(figsize=(10, 6))
    
    # Plot Fitness
    plt.plot(gen, fit_maxs, "g-", label="Best Fitness", linewidth=2)
    plt.plot(gen, fit_avgs, "b--", label="Avg Fitness")
    plt.fill_between(gen, fit_mins, fit_maxs, alpha=0.1, color='gray')
    
    plt.title("Evoluzione Fitness Genetic Programming")
    plt.xlabel("Generazione")
    plt.ylabel("Fitness Score")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    
    output_plot = os.path.join(HISTORY_DIR, "gp_fitness_trend.png")
    plt.savefig(output_plot)
    print(f"📈 Grafico salvato in: {output_plot}")
    plt.show()
    plt.close()

def record_best_agent_gameplay(agent_path, env_name="Freeway-v4"):
    """Registra una GIF/Video del miglior agente."""
    print(f"🎥 Registrazione gameplay per: {agent_path}...")
    
    try:
        # Carica la logica dell'agente
        agent_action_fn = load_agent_module(agent_path)
        
        # Setup Ambiente con Video Recorder
        # Usiamo render_mode='rgb_array' per il video recorder
        try:
            env = gym.make(env_name, render_mode='rgb_array', obs_type='ram')
        except:
            env = gym.make('ALE/Freeway-v5', render_mode='rgb_array', obs_type='ram')
            
        # Wrapper Video (Salva solo l'episodio 0)
        env = RecordVideo(
            env, 
            video_folder=VIDEO_DIR, 
            episode_trigger=lambda e: e == 0,
            name_prefix="best_gp_agent"
        )
        
        # Wrapper Logica (RAM -> 11 Input)
        # Nota: Il wrapper OCAtari deve stare "sopra" l'env base ma "sotto" il recorder se vogliamo registrare i pixel originali
        # Tuttavia, per semplicità qui usiamo il translator separatamente
        translator = FreewayOCAtariWrapper(env)
        
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # 1. Traduci osservazione per l'agente
            agent_view = translator.observation(obs)
            
            # 2. Chiedi azione all'agente evoluto
            action = agent_action_fn(agent_view)
            
            # 3. Step ambiente
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
        env.close()
        print(f"✅ Video salvato in: {VIDEO_DIR}")
        print(f"🏆 Punteggio finale nel video: {total_reward}")

    except Exception as e:
        print(f"❌ Errore durante la registrazione: {e}")

if __name__ == "__main__":
    # 1. Genera Grafici
    plot_evolution_stats(LOGBOOK_FILE)
    
    # 2. Genera Video
    if os.path.exists(BEST_AGENT_FILE):
        record_best_agent_gameplay(BEST_AGENT_FILE)
    else:
        print(f"❌ File agente '{BEST_AGENT_FILE}' non trovato.")