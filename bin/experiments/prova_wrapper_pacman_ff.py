# FILE: experiments/run_pacman_neat.py
"""
Script di esecuzione NEAT per Pacman (ALE) con fitness SEMPLICE.

FITNESS = SCORE DEL GIOCO (nessun reward shaping)
"""

import sys
import os
import neat
import numpy as np
import pickle
import datetime
import multiprocessing

# --- MONKEY PATCH OCATARI (CRITICO) ---
import ocatari.core
from itertools import chain

def patched_ns_state(self):
    valid_objects = [o for o in self.objects if o is not None and hasattr(o, '_nsrepr')]
    return list(chain.from_iterable([o._nsrepr for o in valid_objects]))

ocatari.core.OCAtari.ns_state = patched_ns_state
# ---------------------------------------

# --- SETUP PATHS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ocatari.core import OCAtari
from core.wrappers_pacman import PacmanFeatureWrapper
from utils.neat_plotting_utils import plot_stats, plot_species

# --- CONFIGURAZIONE ESPERIMENTO ---
ENV_ID = "Pacman"
CONFIG_FILE_NAME = "neat_pacman_config.txt"
NUM_GENERATIONS = 50
MAX_STEPS = 3000
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

OUTPUT_DIR = os.path.join(project_root, "evolution_results", "pacman")
CONFIG_PATH = os.path.join(project_root, "configs", CONFIG_FILE_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_fitness(env, net, max_steps=MAX_STEPS):
    """
    🎯 FITNESS = SCORE DEL GIOCO
    
    Nessun reward shaping, nessuna penalità artificiale.
    Solo il punteggio reale di Pacman.
    """
    observation, info = env.reset()
    done = False
    steps = 0
    total_score = 0.0
    
    while not done and steps < max_steps:
        # Attivazione rete NEAT
        output = net.activate(observation)
        action = np.argmax(output)
        
        # Step ambiente
        try:
            observation, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            print(f"⚠️ Errore durante step: {e}")
            break
        
        # Accumula score
        total_score += reward
        
        steps += 1
        done = done or terminated or truncated
    
    return total_score


def eval_single_genome(genome, config):
    """
    Valuta un singolo genoma NEAT.
    """
    try:
        # Creazione ambiente
        env = OCAtari(ENV_ID, mode="ram", obs_mode="obj", render_mode="rgb_array", hud=False)
        
        # Disabilita sticky actions
        if hasattr(env.unwrapped, 'ale'):
            env.unwrapped.ale.setFloat('repeat_action_probability', 0.0)
        
        # Wrapper di feature extraction
        env = PacmanFeatureWrapper(env, grid_rows=10, grid_cols=10)
        
        # Creazione rete NEAT
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Valutazione
        fitness = calculate_fitness(env, net, max_steps=MAX_STEPS)
        
        env.close()
        return fitness
        
    except Exception as e:
        print(f"❌ Errore in eval_single_genome: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def run_evolution():
    """
    Funzione principale per eseguire l'evoluzione NEAT.
    """
    print("=" * 80)
    print("🧬 NEAT EVOLUTION - PACMAN (FITNESS = SCORE)")
    print("=" * 80)
    print(f"📁 Config: {CONFIG_PATH}")
    print(f"📊 Output: {OUTPUT_DIR}")
    print(f"⚙️ Workers: {NUM_WORKERS}")
    print(f"🔄 Generazioni: {NUM_GENERATIONS}")
    print(f"⏱ Max Steps: {MAX_STEPS}")
    print("=" * 80)
    
    # Caricamento config NEAT
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"❌ Config file non trovato: {CONFIG_PATH}")
    
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )
    
    # Parallel evaluator
    pe = neat.ParallelEvaluator(NUM_WORKERS, eval_single_genome)
    
    # Inizializzazione popolazione
    population = neat.Population(config)
    
    # Reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    # Checkpointer
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_prefix = os.path.join(OUTPUT_DIR, f"checkpoint_{timestamp}_")
    population.add_reporter(neat.Checkpointer(
        generation_interval=5,
        filename_prefix=checkpoint_prefix
    ))
    
    # Evoluzione
    print("\n🚀 Avvio evoluzione...\n")

    try:
        winner = population.run(pe.evaluate, NUM_GENERATIONS)
        
        # Salvataggio risultati
        winner_path = os.path.join(OUTPUT_DIR, f"winner_{timestamp}.pkl")
        with open(winner_path, 'wb') as f:
            pickle.dump((winner, config), f)
        print(f"\n✅ Winner salvato: {winner_path}")
        
        # Visualizzazione statistiche
        print("\n📈 Generazione grafici...")
        try:
            plot_stats(
                stats,
                ylog=False,
                filename=os.path.join(OUTPUT_DIR, f"fitness_evolution_{timestamp}.png")
            )
            plot_species(
                stats,
                filename=os.path.join(OUTPUT_DIR, f"speciation_{timestamp}.png")
            )
            print("✅ Grafici salvati")
        except Exception as e:
            print(f"⚠️ Errore generazione grafici: {e}")

        # Statistiche finali
        print("\n" + "=" * 80)
        print("🏆 EVOLUZIONE COMPLETATA")
        print("=" * 80)
        print(f"Best Score: {winner.fitness:.2f}")
        print(f"Best Genome ID: {winner.key}")
        print(f"Generazioni: {len(stats.generation_statistics)}")
        print(f"Specie finali: {len(stats.get_species_sizes())}")
        print("=" * 80)

        return winner, config, stats
    
    except KeyboardInterrupt:
        print("\n\n⏸ Evoluzione interrotta dall'utente")
        return None, config, None
    
    except Exception as e:
        print(f"\n\n❌ Errore fatale durante evoluzione: {e}")
        import traceback
        traceback.print_exc()
        return None, config, None


if __name__ == "__main__":
    # Metodo di avvio multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    # Esegui evoluzione
    winner, config, stats = run_evolution()
    
    if winner is not None:
        print("\n✨ Usa il genoma winner per testare l'agente!")
        print(f"   Carica con: pickle.load(open('winner_*.pkl', 'rb'))")