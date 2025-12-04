# FILE: experiments/run_freeway_neat_parallel.py

import sys
import os
import neat
import numpy as np
import pickle
import datetime
import multiprocessing as mp

# --- SOLUZIONE PER L'IMPORT ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- FINE SOLUZIONE ---

from core.evaluator import run_game_simulation
from algorithms.neat_runner import run_neat
from utils.neat_plotting_utils import plot_stats, plot_species, draw_net

# --- 0. Parametri dell'Esperimento ---
ENV_NAME = "ALE/Freeway-v5"
CONFIG_FILE_NAME = "neat_freeway_config.txt"
NUM_GENERATIONS = 20
MAX_STEPS = 3000

# Calcola numero di worker (CPU disponibili - 2)
NUM_WORKERS = max(1, mp.cpu_count() - 2)

root_dir = project_root
OUTPUT_DIR = os.path.join(root_dir, "evolution_results")
CONFIG_PATH = os.path.join(root_dir, "configs", CONFIG_FILE_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def agent_decision_function_neat(game_state, net):
    """
    Funzione agente che usa il network NEAT.
    Ora riceve il network come parametro per essere thread-safe.
    """
    if net is None:
        return 0
    
    # Normalizza l'input (RAM 0-255 -> 0.0-1.0)
    features = game_state / 255.0
    
    # Attiva il network
    output = net.activate(features)
    
    # Restituisci l'azione migliore
    return np.argmax(output)


def eval_genome(genome_config_tuple):
    """
    Funzione per valutare un singolo genoma.
    Deve essere una funzione top-level per essere picklable da multiprocessing.
    
    Args:
        genome_config_tuple: tupla (genome_id, genome, config)
    
    Returns:
        tupla (genome_id, fitness)
    """
    genome_id, genome, config = genome_config_tuple
    
    # Crea il network per questo genoma
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Crea una funzione agente che usa questo network specifico
    def agent_func(game_state):
        return agent_decision_function_neat(game_state, net)
    
    # Esegui la simulazione
    fitness, metrics = run_game_simulation(
        agent_decision_function=agent_func,
        env_name=ENV_NAME,
        max_steps=MAX_STEPS,
        obs_type="ram"
    )
    
    return genome_id, fitness


def eval_genomes(genomes, config):
    """
    Funzione fitness per NEAT con parallelizzazione.
    """
    # Prepara i dati per il multiprocessing
    genome_tuples = [(gid, g, config) for gid, g in genomes]
    
    # Usa un pool di worker per valutare i genomi in parallelo
    with mp.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(eval_genome, genome_tuples)
    
    # Assegna le fitness ai genomi
    for genome_id, fitness in results:
        # Trova il genoma corrispondente
        for gid, genome in genomes:
            if gid == genome_id:
                genome.fitness = fitness
                break


if __name__ == "__main__":
    print("--- 🧬 Avvio Evoluzione NEAT Parallelizzata per Freeway ---")
    print(f"--- 🔧 Numero di worker: {NUM_WORKERS} ---")
    print(f"--- 📂 Config: {CONFIG_PATH} ---")
    print(f"--- 📈 Output in: {OUTPUT_DIR} ---")
    
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        winner, config, stats = run_neat(
            eval_function=eval_genomes,
            config_file_path=CONFIG_PATH,
            num_generations=NUM_GENERATIONS,
            output_dir=OUTPUT_DIR,
            timestamp_str=timestamp_str
        )
        
        # --- 4. Salvataggio Risultati ---
        winner_file = os.path.join(OUTPUT_DIR, f"best_genome_neat_{timestamp_str}.pkl")
        with open(winner_file, 'wb') as f:
            pickle.dump(winner, f)
        print(f"\n💾 Genoma vincitore salvato in: {winner_file}")
        
        # Salva i grafici
        plot_stats_file = os.path.join(OUTPUT_DIR, f"neat_fitness_{timestamp_str}.png")
        plot_species_file = os.path.join(OUTPUT_DIR, f"neat_speciation_{timestamp_str}.png")
        
        print(f"📈 Salvataggio grafici...")
        plot_stats(stats, ylog=False, filename=plot_stats_file)
        plot_species(stats, filename=plot_species_file)

        print(f"--- ✅ Fatto. ---")

    except Exception as e:
        print(f"\n❌ Si è verificato un errore durante l'evoluzione NEAT: {e}")
        import traceback
        traceback.print_exc()