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
NUM_GENERATIONS = 20 # Aumentato leggermente per dare tempo all'evoluzione
MAX_STEPS = 2000
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2) 

OUTPUT_DIR = os.path.join(project_root, "evolution_results")
CONFIG_PATH = os.path.join(project_root, "configs", CONFIG_FILE_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True) 

def get_distance_to_powerpill(player, objects):
    """
    Calcola la distanza dalla PowerPill più vicina.
    """
    targets = [o for o in objects if "PowerPill" in o.category]
    if not targets:
        return None 
    dists = [(t.x - player.x)**2 + (t.y - player.y)**2 for t in targets]
    return math.sqrt(min(dists))

def eval_single_genome(genome, config):
    """
    Fitness 4.0: Exploration + Anti-Camping + Aggressive Feeding
    """
    try:
        env = OCAtari(ENV_ID, mode="ram", obs_mode="obj", render_mode="rgb_array")
        if hasattr(env.unwrapped, 'ale'):
            env.unwrapped.ale.setFloat('repeat_action_probability', 0.0)
    except Exception as e:
        print(f"Env Error: {e}")
        return 0.0

    # Usa il wrapper aggiornato (che deve essere salvato in core/wrappers.py)
    env = PacmanHybridWrapper(env, grid_rows=10, grid_cols=10)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    observation, info = env.reset()
    done = False
    steps = 0
    fitness = 0.0
    
    # Variabili per "Starvation" (Fame)
    steps_without_significant_progress = 0
    
    # Logica Esplorazione
    visited_sectors = set()
    
    while not done and steps < MAX_STEPS:
        output = net.activate(observation)
        action = np.argmax(output)
        
        try:
            observation, reward, terminated, truncated, info = env.step(action)
        except Exception:
            break 
            
        # --- ESTRAZIONE POSIZIONE DAL VETTORE (Primi 2 valori) ---
        # Il wrapper mette p_x in [0] e p_y in [1] (normalizzati 0.0-1.0)
        p_x = observation[0]
        p_y = observation[1]
        
        # Calcolo settore (Griglia virtuale 20x20 per mappare l'esplorazione)
        # 20x20 è abbastanza fine da premiare il movimento tra stanze diverse
        sector_x = int(p_x * 20)
        sector_y = int(p_y * 20)
        current_sector = (sector_x, sector_y)
        
        # --- CALCOLO FITNESS ---
        
        # 1. SOPRAVVIVENZA BASICA
        fitness += 0.1 
        
        explored_new_area = False
        
        # 2. PREMIO ESPLORAZIONE (Fondamentale se non vede i pellet)
        if current_sector not in visited_sectors:
            visited_sectors.add(current_sector)
            fitness += 5.0 # Bonus consistente per aver scoperto una nuova zona
            explored_new_area = True
        
        # 3. SCORE (Il vero obiettivo)
        if reward > 0:
            # Aumentiamo il peso del cibo reale per distinguerlo dall'esplorazione
            fitness += reward * 20.0  
            steps_without_significant_progress = 0
        elif explored_new_area:
            # Se ha esplorato, resettiamo il timer della fame anche se non ha mangiato
            steps_without_significant_progress = 0
        else:
            # Se non mangia E non esplora, la fame sale
            steps_without_significant_progress += 1
            
        # 4. STARVATION / ANTI-CAMPING
        # Se non fa nulla di utile (mangiare o esplorare) per 200 frame, muore.
        if steps_without_significant_progress > 200:
            fitness -= 10.0 # Punizione
            done = True 
        
        steps += 1
        done = done or terminated or truncated
        
    env.close()
    
    return max(0.0, fitness)
    
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork') 
    except RuntimeError:
        pass

    print("=" * 70)
    print(f"🧬 AVVIO PACMAN NEAT (EXPLORATION v4 + WALL SENSORS)")
    print("=" * 70)
    
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        CONFIG_PATH
    )
    
    # Nota: Assicurati che CONFIG_PATH (neat_pacman_config.txt) abbia num_inputs aggiornato!
    # Con le modifiche al wrapper, num_inputs dovrebbe essere circa 126.
    
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