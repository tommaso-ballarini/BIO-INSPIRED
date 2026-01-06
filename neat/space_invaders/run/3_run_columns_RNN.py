import sys
import os
import pickle
import multiprocessing
import numpy as np
import neat
import matplotlib.pyplot as plt
import gymnasium as gym

# Import OCAtari per l'estrazione oggetti
try:
    from ocatari.core import OCAtari
except ImportError:
    print("❌ ERRORE: Libreria OCAtari non installata. Serve per il wrapper a oggetti.")
    sys.exit(1)

# --- GESTIONE PERCORSI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import del Wrapper specifico
try:
    from wrapper.wrapper_si_columns_RNN import SpaceInvadersColumnWrapper
except ImportError:
    print("❌ ERRORE: Non trovo 'wrapper_si_columns.py' nella cartella wrapper!")
    sys.exit(1)

# --- CONFIGURAZIONI ---
CONFIG_PATH = os.path.join(project_root, 'config', 'config_si_columns_RNN.txt')
RESULTS_DIR = os.path.join(project_root, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

GAME_NAME = "SpaceInvadersNoFrameskip-v4"
GENERATIONS = 30
FIXED_SEED = 42 

print(f"✅ Configurazione: {GAME_NAME} con Column Wrapper (RNN Mode)")
print(f"🔒 Seed Fissato a: {FIXED_SEED}")

# --- FUNZIONI DI PLOTTING (Adattate dalla Baseline) ---

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
    plt.title(f"Columns RNN Training (Seed {FIXED_SEED})")
    plt.xlabel("Generations")
    plt.ylabel("Fitness (Score)")
    plt.grid()
    plt.legend()
    # Nome file specifico per Columns
    plt.savefig(os.path.join(RESULTS_DIR, "fitness_columns_rnn.png"))
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


# --- LOGICA DI VALUTAZIONE ---

def eval_genome(genome, config):
    # 1. Creazione Rete (Ora può essere Recurrent grazie al config)
    # create() gestisce automaticamente FeedForward o Recurrent in base al config
    net = neat.nn.RecurrentNetwork.create(genome, config)
    
    # 2. Setup Ambiente (OCAtari + Wrapper)
    try:
        # Import locale per sicurezza nei sottoprocessi
        import ale_py
        env = OCAtari(GAME_NAME, mode="ram", hud=False, render_mode=None)
    except Exception as e:
        print(f"⚠️ Errore init OCAtari worker: {e}")
        return 0.0
    
    # Applichiamo il Wrapper (Skip=4, 10 Colonne)
    env = SpaceInvadersColumnWrapper(env, n_columns=10, skip=4)
    
    # 3. Reset Deterministico
    observation, info = env.reset(seed=FIXED_SEED)
    
    
    if len(observation) != 32:
        print(f"⚠️ ERRORE DIMENSIONI: Atteso 32, ricevuto {len(observation)}")
        env.close()
        return 0.0

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    max_steps = 10000 

    while not (terminated or truncated) and steps < max_steps:
        inputs = observation
        
        # Attivazione rete
        outputs = net.activate(inputs)
        action = np.argmax(outputs)
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    env.close()
    return total_reward

# --- MAIN ---

def run_columns():
    print(f"📂 Caricamento config: {CONFIG_PATH}")
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Config non trovato! Crea {CONFIG_PATH} con num_inputs=32 e feed_forward=False.")
        return

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)

    # Verifica veloce config
    if config.genome_config.num_inputs != 32:
        print(f"❌ ERRORE CONFIG: num_inputs è {config.genome_config.num_inputs}, deve essere 32!")
        return

    p = neat.Population(config)
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10, filename_prefix=os.path.join(RESULTS_DIR, "neat-col-checkpoint-")))

    # Parallelismo
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"🚀 Avvio Training Columns RNN su {num_workers} processi...")
    
    pe = neat.ParallelEvaluator(num_workers, eval_genome)
    
    try:
        winner = p.run(pe.evaluate, GENERATIONS)
        
        best_ever = stats.best_genome()
        print(f"\n🏆 Fine Training Columns.")
        print(f"💎 Best Ever Fitness: {best_ever.fitness}")
        
        # Salvataggio Genoma
        with open(os.path.join(RESULTS_DIR, 'columns_winner.pkl'), 'wb') as f:
            pickle.dump(best_ever, f)
        print(f"💾 Salvato in: columns_winner.pkl")

        # --- GENERAZIONE GRAFICI ---
        plot_stats(stats)
        plot_species(stats)
        
    except Exception as e:
        print(f"\n❌ ERRORE CRITICO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_columns()