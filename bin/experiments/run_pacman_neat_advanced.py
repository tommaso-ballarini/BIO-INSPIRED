# experiments/run_pacman_neat_advanced.py
"""
Script di training NEAT per Pac-Man con Feature Engineering Avanzata.
Implementa tutte le raccomandazioni del documento.
"""

import sys
import os
import neat
import numpy as np
import pickle
import datetime
import multiprocessing
from itertools import chain

# --- SETUP PATH ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- MONKEY PATCH OCAtari ---
import ocatari.core
def patched_ns_state(self):
    valid_objects = [o for o in self.objects if o is not None and hasattr(o, '_nsrepr')]
    return list(chain.from_iterable([o._nsrepr for o in valid_objects]))
ocatari.core.OCAtari.ns_state = patched_ns_state

# --- IMPORTS ---
from ocatari.core import OCAtari
from core.wrappers_pacman_advanced import PacmanAdvancedWrapper
from core.fitness_pacman import PacmanFitnessCalculator
from utils.neat_plotting_utils import plot_stats, plot_species

# --- PARAMETRI ---
ENV_ID = "Pacman"
CONFIG_FILE_NAME = "neat_pacman_config.txt"
NUM_GENERATIONS = 50
MAX_STEPS = 3000
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)

OUTPUT_DIR = os.path.join(project_root, "evolution_results")
CONFIG_PATH = os.path.join(project_root, "configs", CONFIG_FILE_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def eval_single_genome(genome, config):
    """
    Funzione di valutazione per singolo genoma.
    Usa il wrapper avanzato e la fitness shaping.
    """
    try:
        # Crea ambiente con wrapper avanzato
        env = OCAtari(ENV_ID, mode="ram", obs_mode="obj", render_mode=None)
        if hasattr(env.unwrapped, 'ale'):
            env.unwrapped.ale.setFloat('repeat_action_probability', 0.0)
        
        # Applica wrapper avanzato
        env = PacmanAdvancedWrapper(env, num_rays=8, num_sectors=8)
        
    except Exception as e:
        print(f"❌ Errore creazione ambiente: {e}")
        return 0.0
    
    # Crea rete neurale
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Inizializza fitness calculator
    fitness_calc = PacmanFitnessCalculator(
        w_score=1.0,
        w_survival=0.01,
        w_exploration=3.0,      # Aumentato per incentivare esplorazione
        w_ghost_bonus=15.0,     # Aumentato per comportamento aggressivo
        w_potential=1.0,        # Potential-based shaping attivo
        penalty_death=100.0,
        penalty_camping=0.1,
        camping_threshold=150
    )
    
    observation, info = env.reset()
    done = False
    steps = 0
    
    while not done and steps < MAX_STEPS:
        # Decisione rete
        try:
            output = net.activate(observation)
            action = np.argmax(output)
        except Exception:
            action = 0  # NOOP di default
        
        # Step ambiente
        try:
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Aggiorna fitness (passa anche gli oggetti per potential-based)
            objects = []
            if hasattr(env.unwrapped, "objects"):
                objects = [o for o in env.unwrapped.objects if o is not None]
            
            fitness_calc.update(observation, reward, objects, terminated or truncated)
            
            done = terminated or truncated
            steps += 1
            
        except Exception as e:
            print(f"⚠️ Errore step: {e}")
            break
    
    env.close()
    
    # Fitness finale
    final_fitness = fitness_calc.get_final_fitness()
    
    return final_fitness

if __name__ == "__main__":
    if sys.platform == "win32":
        multiprocessing.set_start_method("spawn", force=True)
    else:
        multiprocessing.set_start_method("fork", force=True)

    
    print("=" * 70)
    print("🧬 AVVIO NEAT PAC-MAN AVANZATO")
    print("=" * 70)
    print(f"🎮 Ambiente: {ENV_ID}")
    print(f"🔬 Feature Engineering: Ray-Casting + Pie-Slice Radar")
    print(f"🎯 Fitness: Multi-Componente + Potential-Based Shaping")
    print(f"⚙️  Workers: {NUM_WORKERS}")
    print(f"📈 Generazioni: {NUM_GENERATIONS}")
    print("=" * 70)
    
    # Verifica config
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Config non trovata: {CONFIG_PATH}")
        sys.exit(1)
    
    # Carica config NEAT
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )
    
    # IMPORTANTE: Verifica che num_inputs nel config corrisponda al wrapper!
    # Il wrapper avanzato usa 42 input
    if config.genome_config.num_inputs != 42:
        print(f"⚠️  ATTENZIONE: Config ha {config.genome_config.num_inputs} input, ma il wrapper usa 42!")
        print("Aggiorna configs/neat_pacman_config.txt con: num_inputs = 42")
        sys.exit(1)
    
    # Setup parallelizzazione
    pe = neat.ParallelEvaluator(NUM_WORKERS, eval_single_genome)
    
    # Crea popolazione
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # Checkpoint ogni 5 generazioni
    checkpoint_prefix = os.path.join(OUTPUT_DIR, "pacman_adv_chk_")
    p.add_reporter(neat.Checkpointer(5, filename_prefix=checkpoint_prefix))
    
    # --- ESECUZIONE EVOLUZIONE ---
    try:
        print("\n🚀 Avvio evoluzione...\n")
        winner = p.run(pe.evaluate, NUM_GENERATIONS)
        
        # --- SALVATAGGIO RISULTATI ---
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Salva genoma vincitore
        winner_file = os.path.join(OUTPUT_DIR, f"pacman_advanced_winner_{timestamp}.pkl")
        with open(winner_file, 'wb') as f:
            pickle.dump(winner, f)
        
        print(f"\n✅ Evoluzione completata!")
        print(f"💾 Genoma salvato: {winner_file}")
        print(f"🏆 Fitness vincitore: {winner.fitness:.2f}")
        
        # Salva grafici
        try:
            plot_stats(stats, ylog=False, 
                      filename=os.path.join(OUTPUT_DIR, f"fitness_adv_{timestamp}.png"))
            plot_species(stats, 
                        filename=os.path.join(OUTPUT_DIR, f"species_adv_{timestamp}.png"))
            print("📊 Grafici salvati")
        except Exception as e:
            print(f"⚠️ Errore grafici: {e}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrotto dall'utente")
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()