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


def visualize_neat(env_name, result_dir, config_path, max_steps=10000, num_episodes=10): # <--- AGGIUNTA: num_episodes
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
    net = neat.nn.RecurrentNetwork.create(winner, neat_config) 

    # Definisci la funzione agente
    def agent_decision_function(game_state):
        features = game_state / 255.0
        output = net.activate(features)
        
        action_idx = np.argmax(output)
        
        # 💡 MODIFICA 2: Mappatura Azioni per Pacman (0, 2, 3, 4, 5)
        # 0: NOOP, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN
        ACTIONS = [0, 2, 3, 4, 5] 
        return ACTIONS[action_idx]

    print("=" * 70)
    print("🎮 AVVIO VALUTAZIONE MULTI-EPISODIO MS. PAC-MAN")
    print("=" * 70)
    print(f"🧬 Genoma: {os.path.basename(best_genome_file)}")
    print(f"🏆 Fitness allenamento (MAX): {winner.fitness:.2f}")
    print(f"⏱️  Max steps: {max_steps}")
    print(f"🔄 Episodi di test: {num_episodes}") # <--- NUOVA RIGA
    print("=" * 70)

    fitness_scores = [] # <--- LISTA PER I RISULTATI

    for i in range(num_episodes): # <--- LOOP PER LA VALUTAZIONE
        # Renderizza solo il primo episodio per visualizzazione.
        current_render = True if i == 0 else False 
        
        # Se utilizzi una rete ricorrente (feed_forward=False), devi resettare lo stato interno:
        if hasattr(net, 'reset'):
            net.reset()
            
        print(f"Esecuzione Episodio {i+1}/{num_episodes} (Visualizzazione: {current_render})")

        try:
            fitness, metrics = run_game_simulation(
                agent_decision_function=agent_decision_function,
                env_name=env_name,
                max_steps=max_steps,
                obs_type="ram",
                frameskip=2,  # <--- CORREZIONE: Impostato a 4 (come nel tuo training)
                repeat_action_probability=0.0,
                render=current_render, # <--- Renderizza solo il primo episodio
            )
        except TypeError:
            # fallback se run_game_simulation non accetta render=
            fitness, metrics = run_game_simulation(
                agent_decision_function=agent_decision_function,
                env_name=env_name,
                max_steps=max_steps,
                obs_type="ram",
                frameskip=2, # <--- CORREZIONE: Impostato a 4
                repeat_action_probability=0.0,
            )
            
        fitness_scores.append(fitness) # <--- AGGIUNGI IL RISULTATO ALLA LISTA

    # Calcola statistiche aggregate
    avg_fitness = np.mean(fitness_scores)
    std_fitness = np.std(fitness_scores)
    
    print("\n" + "=" * 70)
    print("📊 RISULTATI AGGREGATI (SU PIÙ EPISODI)")
    print("=" * 70)
    print(f"🎯 Fitness Media: {avg_fitness:.2f} (Deviazione Standard: {std_fitness:.2f})")
    print(f"📈 Fitness Min/Max Test: {np.min(fitness_scores):.2f} / {np.max(fitness_scores):.2f}")
    if metrics is not None:
        # Mostra le metriche dell'ultimo episodio.
        print(f"📈 Metriche Ultimo Episodio: {metrics}")
    print("=" * 70)


if __name__ == "__main__":
    ENV_NAME = "ALE/Pacman-v5"
    RESULT_DIR = os.path.join(PROJECT_ROOT, "evolution_results")
    CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "temp_rnn_neat_pacman_config.txt")
    
    # Imposta i parametri di valutazione
    TEST_MAX_STEPS = 10000 # <--- MODIFICA: Allineato al training
    TEST_NUM_EPISODES = 10 # <--- NUOVO: Numero di episodi da eseguire (es. 10)

    if not os.path.isdir(RESULT_DIR):
        print(f"❌ Cartella {RESULT_DIR} non trovata. Esegui prima l'evoluzione NEAT.")
        sys.exit(1)

    visualize_neat(
        env_name=ENV_NAME,
        result_dir=RESULT_DIR,
        config_path=CONFIG_PATH,
        max_steps=TEST_MAX_STEPS, # Passa il nuovo valore di max_steps
        num_episodes=TEST_NUM_EPISODES, # Passa il numero di episodi
    )