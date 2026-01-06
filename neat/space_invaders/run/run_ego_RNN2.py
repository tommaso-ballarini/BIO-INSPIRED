import sys
import os
import pickle
import multiprocessing
import numpy as np
import neat
import matplotlib.pyplot as plt
import gymnasium as gym
from ocatari.core import OCAtari

# --- PERCORSI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import Wrapper
try:
    from wrapper.wrapper_si_ego import SpaceInvadersEgocentricWrapper
except ImportError:
    sys.exit("❌ Errore Import Wrapper")

CONFIG_PATH = os.path.join(project_root, 'config', 'config_si_ego.txt')
RESULTS_DIR = os.path.join(project_root, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

GAME_NAME = "SpaceInvadersNoFrameskip-v4"
GENERATIONS = 30  # Aumentato un po' perché le RNN richiedono più tempo per evolvere

# --- EVALUATION (NO SEED) ---
def eval_genome(genome, config):
    # 1. CAMBIO: Usa RecurrentNetwork
    net = neat.nn.RecurrentNetwork.create(genome, config)
    
    try:
        env = OCAtari(GAME_NAME, mode="ram", hud=False, render_mode=None)
    except Exception:
        return 0.0

    env = SpaceInvadersEgocentricWrapper(env, skip=4)
    
    # 2. CAMBIO: Niente seed (o seed=None). Ogni partita è diversa.
    # Questo costringe l'IA a imparare regole generali, non a memorizzare lo schema.
    observation, info = env.reset(seed=None) 
    
    if len(observation) != 19:
        env.close()
        return 0.0
    
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    max_steps = 8000 

    while not (terminated or truncated) and steps < max_steps:
        # Per le RNN, l'activate gestisce la memoria interna automaticamente
        outputs = net.activate(observation)
        action = np.argmax(outputs)
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    env.close()
    
    # Fitness calculation
    fitness = total_reward
    if fitness == 0:
        fitness += (steps / 10000.0)

    return max(0.001, fitness)

# --- UTILS PLOT (Uguale a prima) ---
# --- PLOTTING (Invariato) ---
def plot_stats(statistics):
    if not statistics.most_fit_genomes: return
    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    plt.figure(figsize=(10, 6))
    plt.plot(generation, best_fitness, 'r-', label="Best Fitness")
    plt.plot(generation, avg_fitness, 'b-', label="Avg Fitness")
    plt.title(f"Egocentric RNN Training")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "fitness_ego_rnn.png"))
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
        output_path = os.path.join(RESULTS_DIR, "speciation_ego_rnn.png")
        plt.savefig(output_path)
        print(f"✅ Speciation plot saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error during plotting: {e}")
        # Debug info
        print(f"   Curves shape: {curves.shape if hasattr(curves, 'shape') else 'Unknown'}")
    
    plt.close()

def run_training():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10, filename_prefix=os.path.join(RESULTS_DIR, "neat-rnn-chk-")))

    num_workers = max(1, multiprocessing.cpu_count() - 2)
    pe = neat.ParallelEvaluator(num_workers, eval_genome)
    
    try:
        print(f"🚀 Avvio Training RNN (No Seed) su {num_workers} processi...")
        
        # Eseguiamo l'evoluzione
        winner = p.run(pe.evaluate, GENERATIONS)
        
        print("\n🏆 Training Completato!")

        # --- SALVATAGGIO TOP 3 ---
        all_genomes = list(p.population.values())
        all_genomes.sort(key=lambda g: g.fitness if g.fitness else 0.0, reverse=True)
        top_3 = all_genomes[:3]
        
        print(f"💾 Salvataggio Top 3 Genomi...")
        for i, g in enumerate(top_3):
            filename = os.path.join(RESULTS_DIR, f'winner_rnn_rank_{i+1}.pkl')
            with open(filename, 'wb') as f:
                pickle.dump(g, f)
            print(f"   -> {filename} (Fitness: {g.fitness})")

        with open(os.path.join(RESULTS_DIR, 'top3_list.pkl'), 'wb') as f:
            pickle.dump(top_3, f)

        print("\n📊 Generazione Grafici...")
        plot_stats(stats)
        plot_species(stats)
        print("✅ Grafici generati e salvati in 'results/'")

    except Exception as e:
        print(f"❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_training()