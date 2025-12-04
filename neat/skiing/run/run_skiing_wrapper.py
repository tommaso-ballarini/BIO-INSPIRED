"""
Script di esecuzione NEAT per Skiing con Feature Wrapper
Usa SkiingFeatureWrapper per feature interpretabili
"""

import sys
import os
import neat
import numpy as np
import pickle
import datetime
import multiprocessing

# --- SETUP PATHS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ocatari.core import OCAtari
from wrapper.wrappers_skiing import SkiingFeatureWrapper

# --- CONFIGURAZIONE ESPERIMENTO ---
ENV_ID = "Skiing"
CONFIG_FILE_NAME = "neat_skiing_config.txt"
NUM_GENERATIONS = 100
MAX_STEPS = 2000
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# Configurazione wrapper
USE_VISUAL_GRID = True
GRID_SIZE = 10

OUTPUT_DIR = os.path.join(project_root, "evolution_results", "skiing_wrapper")
CONFIG_PATH = os.path.join(project_root, "configs", CONFIG_FILE_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_fitness(env, net, max_steps=MAX_STEPS):
    """
    🎯 FITNESS = TEMPO DI DISCESA (da minimizzare)
    
    Con il wrapper, le feature sono già interpretabili e normalizzate.
    """
    observation, info = env.reset()
    done = False
    steps = 0
    total_reward = 0.0
    
    while not done and steps < max_steps:
        # Attivazione rete NEAT (FeedForward)
        # observation è già il vettore di feature dal wrapper
        output = net.activate(observation)
        action = np.argmax(output)
        
        # Mapping azioni per Skiing: 0->LEFT, 1->NOOP, 2->RIGHT
        # Il wrapper gestisce internamente la conversione ad azioni ALE
        action_map = {0: 4, 1: 0, 2: 3}  # LEFT, NOOP, RIGHT
        actual_action = action_map.get(action, 0)
        
        # Step ambiente
        try:
            observation, reward, terminated, truncated, info = env.step(actual_action)
            total_reward += reward
        except Exception as e:
            print(f"⚠️ Errore durante step: {e}")
            break
        
        steps += 1
        done = terminated or truncated
    
    # Fitness = valore assoluto del reward totale
    fitness = abs(total_reward)
    
    # Se non completa, aggiungi penalità
    if not terminated and steps >= max_steps:
        fitness += 5000.0
    
    # Se fitness è zero (errore), assegna penalità
    if fitness == 0:
        fitness = 20000.0
    
    return fitness


def eval_single_genome(genome, config):
    """
    Valuta un singolo genoma NEAT con il wrapper.
    """
    try:
        # Creazione ambiente OCAtari
        base_env = OCAtari(
            ENV_ID,
            mode="ram",
            render_mode=None,
            hud=False
        )
        
        # Disabilita sticky actions
        if hasattr(base_env.unwrapped, 'ale'):
            base_env.unwrapped.ale.setFloat('repeat_action_probability', 0.0)
        elif hasattr(base_env, 'ale'):
            base_env.ale.setFloat('repeat_action_probability', 0.0)
        
        # Applica wrapper feature extraction
        env = SkiingFeatureWrapper(
            base_env,
            use_visual_grid=USE_VISUAL_GRID,
            grid_size=GRID_SIZE
        )
        
        # Creazione rete NEAT (FeedForward)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Valutazione (media su 3 run per ridurre varianza)
        fitness_values = []
        for _ in range(3):
            fitness = calculate_fitness(env, net, max_steps=MAX_STEPS)
            fitness_values.append(fitness)
        
        avg_fitness = np.mean(fitness_values)
        
        env.close()
        return avg_fitness
        
    except Exception as e:
        print(f"❌ Errore in eval_single_genome: {e}")
        import traceback
        traceback.print_exc()
        return 100000.0


def run_evolution():
    """
    Funzione principale per eseguire l'evoluzione NEAT.
    """
    print("=" * 80)
    print("🧬 NEAT EVOLUTION - SKIING (FeedForward + Feature Wrapper)")
    print("=" * 80)
    print(f"📁 Config: {CONFIG_PATH}")
    print(f"📊 Output: {OUTPUT_DIR}")
    print(f"⚙️ Workers: {NUM_WORKERS}")
    print(f"🔄 Generazioni: {NUM_GENERATIONS}")
    print(f"⏱ Max Steps: {MAX_STEPS}")
    print(f"🎯 Obiettivo: MINIMIZZARE tempo di discesa")
    print(f"🎮 Ambiente: {ENV_ID} + SkiingFeatureWrapper")
    print(f"📐 Visual Grid: {USE_VISUAL_GRID} ({GRID_SIZE}x{GRID_SIZE})")
    
    # Calcola feature dimension
    base_features = 24
    grid_features = GRID_SIZE * GRID_SIZE if USE_VISUAL_GRID else 0
    total_features = base_features + grid_features
    print(f"📊 Total Features: {total_features}")
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
    
    # Verifica num_inputs nel config
    if config.genome_config.num_inputs != total_features:
        print(f"⚠️ WARNING: Config num_inputs={config.genome_config.num_inputs}, "
              f"ma wrapper usa {total_features} features!")
        print(f"   Aggiorna il config con num_inputs = {total_features}")
    
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
            from utils.neat_plotting_utils import plot_stats, plot_species
            
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
        except ImportError:
            print("⚠️ utils.neat_plotting_utils non disponibile, salto grafici")
        except Exception as e:
            print(f"⚠️ Errore generazione grafici: {e}")

        # Statistiche finali
        print("\n" + "=" * 80)
        print("🏆 EVOLUZIONE COMPLETATA")
        print("=" * 80)
        print(f"Best Fitness: {winner.fitness:.2f}")
        print(f"Best Genome ID: {winner.key}")
        print(f"Generazioni: {len(stats.generation_statistics)}")
        
        if hasattr(stats, 'get_species_sizes'):
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


def test_winner(winner_path, num_episodes=5, render=True):
    """
    Testa il genoma vincitore con visualizzazione feature.
    """
    print("\n" + "=" * 80)
    print("🎮 TEST WINNER GENOME + FEATURE ANALYSIS")
    print("=" * 80)
    
    # Carica winner e config
    with open(winner_path, 'rb') as f:
        winner, config = pickle.load(f)
    
    print(f"Genome ID: {winner.key}")
    print(f"Fitness: {winner.fitness:.2f}")
    print(f"Testing {num_episodes} episodes...")
    
    # Crea ambiente
    render_mode = "human" if render else None
    base_env = OCAtari(ENV_ID, mode="ram", render_mode=render_mode, hud=False)
    
    if hasattr(base_env.unwrapped, 'ale'):
        base_env.unwrapped.ale.setFloat('repeat_action_probability', 0.0)
    
    env = SkiingFeatureWrapper(
        base_env,
        use_visual_grid=USE_VISUAL_GRID,
        grid_size=GRID_SIZE
    )
    
    # Crea rete
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    scores = []
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        done = False
        steps = 0
        total_reward = 0.0
        
        print(f"\n--- Episode {episode + 1} ---")
        
        while not done and steps < MAX_STEPS:
            output = net.activate(observation)
            action = np.argmax(output)
            
            action_map = {0: 4, 1: 0, 2: 3}
            actual_action = action_map.get(action, 0)
            
            observation, reward, terminated, truncated, info = env.step(actual_action)
            total_reward += reward
            
            # Print feature ogni 50 step
            if steps % 50 == 0:
                print(f"  Step {steps}: X={observation[0]:.2f}, "
                      f"Speed={observation[2]:.2f}, Collision={observation[17]:.2f}")
            
            steps += 1
            done = terminated or truncated
        
        score = abs(total_reward)
        scores.append(score)
        print(f"  Final Score: {score:.2f}, Steps: {steps}")
    
    env.close()
    
    print("\n" + "=" * 80)
    print(f"📊 Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"📊 Best Score: {np.min(scores):.2f}")
    print(f"📊 Worst Score: {np.max(scores):.2f}")
    print("=" * 80)


if __name__ == "__main__":
    # Metodo di avvio multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    # Test rapido wrapper
    print("🧪 Test wrapper...")
    try:
        test_env = OCAtari(ENV_ID, mode="ram", render_mode=None, hud=False)
        wrapped = SkiingFeatureWrapper(test_env, use_visual_grid=USE_VISUAL_GRID, grid_size=GRID_SIZE)
        obs, info = wrapped.reset()
        print(f"✅ Wrapper OK - Observation shape: {obs.shape}")
        wrapped.close()
    except Exception as e:
        print(f"❌ Errore test wrapper: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Esegui evoluzione
    winner, config, stats = run_evolution()
    
    if winner is not None:
        print("\n✨ Testa il winner con:")
        print(f"   from experiments.run_skiing_neat_wrapper import test_winner")
        print(f"   test_winner('evolution_results/skiing_wrapper/winner_TIMESTAMP.pkl')")