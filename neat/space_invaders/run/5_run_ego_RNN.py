import sys
import os
import pickle
import multiprocessing
import numpy as np
import neat
import matplotlib.pyplot as plt
import gymnasium as gym
from ocatari.core import OCAtari
import random
import time

# --- PERCORSI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import Wrapper
try:
    from wrapper.wrapper_si_ego import SpaceInvadersEgocentricWrapper
except ImportError:
    print("❌ ERRORE: Non trovo 'wrapper_si_ego.py'!")
    sys.exit(1)


# --- CONFIGURAZIONI ---
CONFIG_PATH = os.path.join(project_root, 'config', 'config_si_ego.txt')
RESULTS_DIR = os.path.join(project_root, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

GAME_NAME = "SpaceInvadersNoFrameskip-v4"
GENERATIONS = 30
# IMPORTANTE: Niente seed fisso per il training RNN
print(f"✅ Configurazione: {GAME_NAME} (RNN Mode - No Fixed Seed)")

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
    plt.savefig(os.path.join(RESULTS_DIR, "fitness_rnn.png"))
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


# --- EVALUATION ---
def eval_genome(genome, config):
    # Usiamo RNN
    net = neat.nn.RecurrentNetwork.create(genome, config)
    
    # Setup Ambiente
    try:
        env = OCAtari(GAME_NAME, mode="ram", hud=False, render_mode=None)
    except Exception:
        return 0.0
    
    env = SpaceInvadersEgocentricWrapper(env, skip=4)
    
    # --- MODIFICA ROBUSTEZZA: MEDIA SU 3 PARTITE ---
    n_episodes = 3
    total_fitness_acc = 0.0
    
    # Per sicurezza sulla casualità nei processi paralleli
    random.seed(os.getpid() + time.time())
    
    for _ in range(n_episodes):
        # 1. Reset
        observation, info = env.reset(seed=None)
        
        # 2. FIX CRITICO: RANDOM NO-OPS
        # Senza questo, le 3 partite sarebbero identiche!
        # Facciamo passare da 0 a 30 frame a vuoto per desincronizzare gli alieni.
        random_delay = random.randint(0, 30)
        for _ in range(random_delay):
            observation, _, terminated, truncated, _ = env.step(0) # 0 = NOOP
            if terminated or truncated: break
        
        # Check sicurezza dopo il delay
        if len(observation) != 19:
            env.close()
            return 0.0
        
        episode_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        max_steps = 6000 
        
        # 3. Reset memoria RNN per ogni episodio! (Molto importante)
        net.reset()

        while not (terminated or truncated) and steps < max_steps:
            outputs = net.activate(observation)
            action = np.argmax(outputs)
            
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
        # Bonus sopravvivenza minimo
        if episode_reward == 0:
            episode_reward += (steps / 10000.0)
            
        total_fitness_acc += episode_reward

    env.close()

    # FITNESS FINALE = MEDIA DELLE 3 PARTITE
    avg_fitness = total_fitness_acc / n_episodes
    
    return max(0.001, avg_fitness)

# --- MAIN ---
def run_training():
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Config non trovato: {CONFIG_PATH}")
        return

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)

    # Assicurati che nel config ci sia feed_forward = False!
    if config.genome_config.num_inputs != 19:
        print(f"❌ ERRORE CONFIG: num_inputs deve essere 19!")
        return

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10, filename_prefix=os.path.join(RESULTS_DIR, "neat-rnn-chk-")))

    num_workers = max(1, multiprocessing.cpu_count() - 2)
    pe = neat.ParallelEvaluator(num_workers, eval_genome)
    
    try:
        print(f"🚀 Avvio Training RNN...")
        winner = p.run(pe.evaluate, GENERATIONS)
        
        # Salvataggio vincitore singolo
        with open(os.path.join(RESULTS_DIR, 'winner_ego.pkl'), 'wb') as f:
            pickle.dump(winner, f)
        
        # --- 3. MODIFICA: SALVATAGGIO TOP 3 ---
        # Recuperiamo tutti i genomi, li ordiniamo e salviamo i migliori 3
        all_genomes = list(p.population.values())
        all_genomes.sort(key=lambda g: g.fitness if g.fitness else 0.0, reverse=True)
        top_3 = all_genomes[:3]
        
        top3_path = os.path.join(RESULTS_DIR, 'top3_list.pkl')
        with open(top3_path, 'wb') as f:
            pickle.dump(top_3, f)
            
        print(f"💾 Salvato winner_ego.pkl")
        print(f"💾 Salvata lista Top 3 in: {top3_path}")

        plot_stats(stats)
        
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_training()