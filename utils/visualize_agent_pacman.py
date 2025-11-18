# FILE: utils/visualize_agent_pacman.py

import os
import sys
import pickle
import numpy as np
import neat

# --- Imposta i percorsi base ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.evaluator import run_game_simulation  # import interno

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


def visualize_neat(env_name, result_dir, config_path, max_steps=6000):
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

    print("=" * 70)
    print("🎮 AVVIO VISUALIZZAZIONE MS. PAC-MAN")
    print("=" * 70)
    print(f"🧬 Genoma: {os.path.basename(best_genome_file)}")
    print(f"🏆 Fitness allenamento: {winner.fitness:.2f}")
    print(f"⏱️  Max steps: {max_steps}")
    print("=" * 70)

    try:
        fitness, metrics = run_game_simulation(
            agent_decision_function=agent_decision_function,
            env_name=env_name,
            max_steps=max_steps,
            obs_type="ram",
            frameskip=1,  # Stesso valore usato nel training
            repeat_action_probability=0.0,  # Stesso valore usato nel training
            render=True,  # Abilita visualizzazione
        )
    except TypeError:
        # fallback se run_game_simulation non accetta render=
        fitness, metrics = run_game_simulation(
            agent_decision_function=agent_decision_function,
            env_name=env_name,
            max_steps=max_steps,
            obs_type="ram",
            frameskip=1,
            repeat_action_probability=0.0,
        )

    print("\n" + "=" * 70)
    print("📊 RISULTATI EPISODIO")
    print("=" * 70)
    print(f"🎯 Fitness: {fitness:.2f}")
    if metrics is not None:
        print(f"📈 Metrics: {metrics}")
    print("=" * 70)


if __name__ == "__main__":
    ENV_NAME = "ALE/MsPacman-v5"
    RESULT_DIR = os.path.join(PROJECT_ROOT, "evolution_results")
    CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "neat_pacman_config.txt")

    if not os.path.isdir(RESULT_DIR):
        print(f"❌ Cartella {RESULT_DIR} non trovata. Esegui prima l'evoluzione NEAT.")
        sys.exit(1)

    visualize_neat(
        env_name=ENV_NAME,
        result_dir=RESULT_DIR,
        config_path=CONFIG_PATH,
        max_steps=6000,  # Stesso max_steps del training
    )