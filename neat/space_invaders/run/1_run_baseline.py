import sys
import os
import pickle
import multiprocessing
import numpy as np
import neat
import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py 

# --- GESTIONE PERCORSI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# --- CONFIGURAZIONI GLOBALI ---
CONFIG_PATH = os.path.join(project_root, 'config', 'config_baseline.txt')
RESULTS_DIR = os.path.join(project_root, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# MODIFICA: Nome hardcodato per stabilità
GAME_NAME = "SpaceInvadersNoFrameskip-v4"
GENERATIONS = 30
FIXED_SEED = 42

print(f"✅ Configurazione Ambiente: {GAME_NAME}")
print(f"🔒 Seed Fissato a: {FIXED_SEED} (Determinismo attivato)")

def eval_genome(genome, config):
    """ Valuta un singolo genoma """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    try:
        env = gym.make(GAME_NAME, obs_type="ram", render_mode=None)
    except Exception:
        import ale_py
        env = gym.make(GAME_NAME, obs_type="ram", render_mode=None)

    # --- SEED FISSO PER DETERMINISMO ---
    observation, info = env.reset(seed=FIXED_SEED)

    if len(observation) != 128:
        env.close()
        return 0.0

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    max_steps = 10000 

    while not (terminated or truncated) and steps < max_steps:
        inputs = observation / 255.0 
        if isinstance(inputs, np.ndarray):
            inputs = inputs.flatten()

        outputs = net.activate(inputs)
        action = np.argmax(outputs)
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    env.close()
    return total_reward

def plot_stats(statistics):
    """ Grafico Fitness Media e Migliore """
    print("📊 Generazione grafico Fitness...")
    if not statistics.most_fit_genomes:
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())

    plt.figure(figsize=(10, 6))
    plt.plot(generation, best_fitness, 'r-', label="Best Fitness")
    plt.plot(generation, avg_fitness, 'b-', label="Avg Fitness")
    plt.title(f"Baseline Training - Raw RAM (Seed {FIXED_SEED})")
    plt.xlabel("Generazioni")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend()
    try:
        plt.savefig(os.path.join(RESULTS_DIR, "fitness_baseline.png"))
    except Exception as e:
        print(f"❌ Errore salvataggio fitness: {e}")
    plt.close()

def plot_species(statistics):
    """ Speciation Graph (Stacked Plot) - Correct Version """
    print("📊 Generating Speciation Plot...")
    
    # Official NEAT method to get the correct counts
    # This fixes the "dict object has no attribute members" error
    species_sizes = statistics.get_species_sizes()
    
    if not species_sizes:
        print("⚠️ No speciation data found.")
        return

    num_generations = len(species_sizes)
    
    # Transpose the matrix to fit stackplot (Rows=Species, Cols=Generations)
    curves = np.array(species_sizes).T

    plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    
    try:
        # Use stackplot for the filled area effect
        ax.stackplot(range(num_generations), *curves)
        
        plt.title("Evolution of Species (Speciation)")
        plt.ylabel("Number of Genomes per Species")
        plt.xlabel("Generations")
        plt.margins(0, 0) # Removes white side margins
        
        # Ensure RESULTS_DIR is defined in your global scope
        output_path = os.path.join(RESULTS_DIR, "speciation.png")
        plt.savefig(output_path)
        print(f"✅ Speciation plot saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error during plotting: {e}")
        # Debug info
        print(f"   Curves shape: {curves.shape if hasattr(curves, 'shape') else 'Unknown'}")
    
    plt.close()
def run_baseline():
    print(f"📂 Caricamento config Baseline: {CONFIG_PATH}")
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Config non trovato: {CONFIG_PATH}")
        return

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)

    p = neat.Population(config)
    
    # Reporter
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10, filename_prefix=os.path.join(RESULTS_DIR, "neat-checkpoint-")))

    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"🚀 Avvio Baseline su {num_workers} processi...")
    
    pe = neat.ParallelEvaluator(num_workers, eval_genome)
    
    try:
        winner = p.run(pe.evaluate, GENERATIONS)
        
        print(f"\n🏆 Fine Training.")
        print(f"💎 Best Ever Fitness: {winner.fitness}")
        
        with open(os.path.join(RESULTS_DIR, 'baseline_winner.pkl'), 'wb') as f:
            pickle.dump(winner, f)
        
        # Generazione Grafici
        plot_stats(stats)
        plot_species(stats)
        
        print("✅ Baseline completata con successo.")

    except Exception as e:
        print(f"\n❌ ERRORE CRITICO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    print(f"Gymnasium version: {gym.__version__}")
    run_baseline()