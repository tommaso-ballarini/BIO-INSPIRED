import sys
import os
import neat
import numpy as np
import pickle
import datetime
import multiprocessing 

# --- SETUP PATH ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------

from ocatari.core import OCAtari
from core.wrappers import HybridPacmanWrapper 
from algorithms.neat_runner import run_neat
from utils.neat_plotting_utils import plot_stats, plot_species

# --- PARAMETRI ---
ENV_ID = "MsPacman-v4" 
CONFIG_FILE_NAME = "neat_pacman_config.txt"
NUM_GENERATIONS = 50 # Aumentato un po' dato che ora andremo veloci
MAX_STEPS = 2000
NUM_WORKERS = multiprocessing.cpu_count() # Usa tutti i core disponibili (o metti un numero fisso es. 4)

root_dir = project_root
OUTPUT_DIR = os.path.join(root_dir, "evolution_results")
CONFIG_PATH = os.path.join(root_dir, "configs", CONFIG_FILE_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True) 

# --- FUNZIONE DI VALUTAZIONE SINGOLA (Per Worker) ---
def eval_single_genome(genome, config):
    """
    Questa funzione viene eseguita in parallelo da ogni worker.
    Crea il suo ambiente, valuta il genoma e chiude.
    """
    # Creiamo l'ambiente LOCALMENTE nel processo worker
    # render_mode=None è fondamentale per la velocità in parallelo
    env = OCAtari(ENV_ID, mode="ram", obs_mode="obj", render_mode="rgb_array")
    if hasattr(env.unwrapped, 'ale'):
        env.unwrapped.ale.setFloat('repeat_action_probability', 0.0) #per adesso no stocasticity
    env = HybridPacmanWrapper(env)
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    observation, info = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < MAX_STEPS:
        output = net.activate(observation)
        action = np.argmax(output)
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
        
    env.close()
    
    # Restituisce il fitness (NEAT se lo assegna da solo)
    return total_reward

# --- FUNZIONE OBSOLETA (Mantenuta vuota o rimossa) ---
# NEAT ParallelEvaluator non usa questa funzione classica, ma per compatibilità
# con il tuo runner attuale, definiremo il flusso nel main.

# --- ESECUZIONE ---
if __name__ == "__main__":
    # Fix per multiprocessing su alcuni OS (Linux solitamente usa fork, spawn è più sicuro ma più lento all'avvio)
    try:
        multiprocessing.set_start_method('fork') 
    except RuntimeError:
        pass

    print("=" * 70)
    print(f"🧬 AVVIO EVOLUZIONE NEAT PARALLELA ({NUM_WORKERS} Cores)")
    print("=" * 70)
    print(f"🎮 Environment: {ENV_ID} (OCAtari RAM)")
    
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ ERRORE Config: {CONFIG_PATH}")
        sys.exit(1)
    
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            CONFIG_PATH
        )
        
        # --- SETUP PARALLELIZZAZIONE ---
        # ParallelEvaluator gestisce il pool di processi
        pe = neat.ParallelEvaluator(NUM_WORKERS, eval_single_genome)
        
        print(f"🚀 Avvio con {NUM_WORKERS} worker in parallelo...")
        
        # Creiamo la popolazione
        p = neat.Population(config)
        
        # Aggiungiamo reporter standard
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        
        # --- ESEGUIAMO L'EVOLUZIONE ---
        # Nota: Qui usiamo pe.evaluate invece della funzione eval_genomes custom
        winner = p.run(pe.evaluate, NUM_GENERATIONS)
        
        # --- SALVATAGGIO ---
        print("\n💾 Salvataggio risultati...")
        winner_file = os.path.join(OUTPUT_DIR, f"best_genome_neat_{timestamp_str}.pkl")
        with open(winner_file, 'wb') as f:
            pickle.dump(winner, f)
            
        try:
            plot_stats(stats, ylog=False, filename=os.path.join(OUTPUT_DIR, f"neat_fitness_{timestamp_str}.png"))
            plot_species(stats, filename=os.path.join(OUTPUT_DIR, f"neat_speciation_{timestamp_str}.png"))
        except Exception as e:
            print(f"⚠️ Grafici non generati: {e}")

    except KeyboardInterrupt:
        print("\n⚠️ Interrotto dall'utente")
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()