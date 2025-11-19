import sys
import os
import neat
import numpy as np
import pickle
import datetime
import multiprocessing 
import math
from itertools import chain

# --- MONKEY PATCH OCAtari (Critica per evitare crash) ---
import ocatari.core
def patched_ns_state(self):
    valid_objects = [o for o in self.objects if o is not None and hasattr(o, '_nsrepr')]
    return list(chain.from_iterable([o._nsrepr for o in valid_objects]))
ocatari.core.OCAtari.ns_state = patched_ns_state
# --------------------------------------------------------

# --- SETUP PATH ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ocatari.core import OCAtari
from core.wrappers import PacmanHybridWrapper
from utils.neat_plotting_utils import plot_stats, plot_species

# --- PARAMETRI ---
ENV_ID = "Pacman" 
CONFIG_FILE_NAME = "neat_pacman_config.txt"
NUM_GENERATIONS = 10 
MAX_STEPS = 2000
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2) 

OUTPUT_DIR = os.path.join(project_root, "evolution_results")
CONFIG_PATH = os.path.join(project_root, "configs", CONFIG_FILE_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True) 

def get_distance_to_powerpill(player, objects):
    """
    Calcola la distanza dalla PowerPill più vicina.
    Dato che i pellet normali non sono rilevati, questo è l'unico 'faro' di navigazione.
    """
    targets = [o for o in objects if "PowerPill" in o.category]
    
    if not targets:
        return None # Tutte le PowerPill mangiate o non presenti
        
    # Distanza euclidea minima
    dists = [(t.x - player.x)**2 + (t.y - player.y)**2 for t in targets]
    return math.sqrt(min(dists))

def eval_single_genome(genome, config):
    """
    Fitness 3.0: Anti-Camping & Aggressive Feeding
    """
    try:
        # Usa sempre repeat_action_probability=0 per determinismo
        env = OCAtari(ENV_ID, mode="ram", obs_mode="obj", render_mode="rgb_array")
        if hasattr(env.unwrapped, 'ale'):
            env.unwrapped.ale.setFloat('repeat_action_probability', 0.0)
    except Exception as e:
        print(f"Env Error: {e}")
        return 0.0

    env = PacmanHybridWrapper(env)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    observation, info = env.reset()
    done = False
    steps = 0
    fitness = 0.0
    
    # Variabili per "Starvation" (Fame)
    steps_without_food = 0
    last_score = 0
    
    while not done and steps < MAX_STEPS:
        output = net.activate(observation)
        action = np.argmax(output)
        
        try:
            observation, reward, terminated, truncated, info = env.step(action)
        except Exception:
            break 
            
        # --- NUOVA LOGICA FITNESS ---
        
        # 1. SOPRAVVIVENZA (Molto ridotta)
        # Diamo pochissimo per il semplice esistere, per evitare loop infiniti
        fitness += 0.1 
        
        # 2. SCORE (Il vero obiettivo)
        # Se fa punti, resettiamo il contatore della fame e diamo un grosso premio
        if reward > 0:
            fitness += reward * 10.0  # 1 pallina (10pt) = +100 fitness
            steps_without_food = 0
        else:
            steps_without_food += 1
            
        # 3. STARVATION (Meccanismo anti-camping)
        # Se non mangia per 200 frame (circa 4 secondi), penalità e chiudiamo l'episodio
        if steps_without_food > 200:
            fitness -= 20.0 # Punizione per pigrizia
            done = True # Uccidiamo l'agente inutile
        
        steps += 1
        # Aggiorniamo done
        done = done or terminated or truncated
        
    env.close()
    
    return max(0.0, fitness)
    
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork') 
    except RuntimeError:
        pass

    print("=" * 70)
    print(f"🧬 AVVIO PACMAN NEAT (SURVIVAL + SCORE FOCUS)")
    print("=" * 70)
    
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        CONFIG_PATH
    )
    
    pe = neat.ParallelEvaluator(NUM_WORKERS, eval_single_genome)
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix=os.path.join(OUTPUT_DIR, "pacman_chk_")))

    try:
        winner = p.run(pe.evaluate, NUM_GENERATIONS)
        
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(os.path.join(OUTPUT_DIR, f"pacman_winner_{ts}.pkl"), 'wb') as f:
            pickle.dump(winner, f)
            
        try:
            plot_stats(stats, ylog=False, filename=os.path.join(OUTPUT_DIR, f"fitness_{ts}.png"))
            plot_species(stats, filename=os.path.join(OUTPUT_DIR, f"speciation_{ts}.png"))
        except:
            pass
            
    except KeyboardInterrupt:
        print("\nInterrotto.")
    except Exception as e:
        print(f"\nErrore: {e}")