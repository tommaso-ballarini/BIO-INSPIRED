# FILE: experiments/run_bankheist_neat.py

import sys
import os
import neat
import numpy as np
import pickle
import datetime

# --- SOLUZIONE PER L'IMPORT ---
# Aggiungi la root del progetto al sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- FINE SOLUZIONE ---

# Importa i nostri moduli custom
from core.evaluator import run_game_simulation
from algorithms.neat_runner import run_neat
from utils.neat_plotting_utils import plot_stats, plot_species, draw_net

# --- 0. Parametri dell'Esperimento ---
ENV_NAME = "ALE/Freeway-v5"
CONFIG_FILE_NAME = "neat_freeway_config.txt"
NUM_GENERATIONS = 5 # Basso per un test, per un run reale usa 100+
MAX_STEPS = 3000 #1500

# Trova la cartella di output e il file di config
root_dir = project_root
OUTPUT_DIR = os.path.join(root_dir, "evolution_results")
CONFIG_PATH = os.path.join(root_dir, "configs", CONFIG_FILE_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True) 

# Variabile globale per il network (non bellissimo, ma richiesto da eval_genomes)
current_net = None

def agent_decision_function_neat(game_state):
    """
    Funzione agente che usa il network NEAT.
    """
    if current_net is None:
        return 0 # Azione NOOP di default

    # 1. Normalizza l'input (RAM 0-255 -> 0.0-1.0)
    features = game_state / 255.0
    
    # 2. Attiva il network
    # NEAT si aspetta l'input come una lista/tuple
    output = current_net.activate(features)
    
    # 3. Restituisci l'azione migliore (indice dell'output più alto)
    return np.argmax(output)


def eval_genomes(genomes, config):
    """
    La funzione fitness richiesta da NEAT.
    Valuta ogni genoma nella popolazione.
    """
    global current_net # Riferimento al network
    
    # Loop su ogni genoma
    for genome_id, genome in genomes:
        
        # 1. Crea il network (il "cervello") da questo genoma
        current_net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # 2. Esegui la simulazione usando il nostro valutatore generico
        #    e la funzione agente specifica per NEAT
        fitness, metrics = run_game_simulation(
            agent_decision_function=agent_decision_function_neat,
            env_name=ENV_NAME,
            max_steps=MAX_STEPS,
            obs_type="ram"
        )
        
        # 3. Assegna la fitness al genoma
        genome.fitness = fitness


# --- 3. Esecuzione ---
if __name__ == "__main__":
    print("--- 🧬 Avvio Evoluzione NEAT per Freeway ---")
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
        
        # Salva il genoma vincitore (come oggetto binario)
        winner_file = os.path.join(OUTPUT_DIR, f"best_genome_neat_{timestamp_str}.pkl")
        with open(winner_file, 'wb') as f:
            pickle.dump(winner, f)
        print(f"\n💾 Genoma vincitore salvato in: {winner_file}")
        
        # Salva i grafici
        plot_stats_file = os.path.join(OUTPUT_DIR, f"neat_fitness_{timestamp_str}.png")
        plot_species_file = os.path.join(OUTPUT_DIR, f"neat_speciation_{timestamp_str}.png")
        plot_net_file = os.path.join(OUTPUT_DIR, f"neat_winner_net_{timestamp_str}") # .png aggiunto da draw_net
        
        print(f"📈 Salvataggio grafici...")
        plot_stats(stats, ylog=False, filename=plot_stats_file)
        plot_species(stats, filename=plot_species_file)
        
        # Disegna il network vincitore
        # node_names = {i: f"RAM_{i}" for i in range(-128, 0)}
        # node_names.update({i: f"ACTION_{i}" for i in range(18)})
        # draw_net(config, winner, view=False, filename=plot_net_file, node_names=node_names)

        print(f"--- ✅ Fatto. ---")

    except Exception as e:
        print(f"\n❌ Si è verificato un errore durante l'evoluzione NEAT: {e}")
        import traceback
        traceback.print_exc()