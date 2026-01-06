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
GENERATIONS = 400  # Run lunga
print(f"✅ Configurazione: {GAME_NAME} (RNN + Survival Logic + Fast Mode)")

# --- PLOTTING ---
def plot_stats(statistics):
    if not statistics.most_fit_genomes: return
    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    plt.figure(figsize=(10, 6))
    plt.plot(generation, best_fitness, 'r-', label="Best Fitness")
    plt.plot(generation, avg_fitness, 'b-', label="Avg Fitness")
    plt.title(f"Egocentric RNN Training (Survival)")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "fitness_ego_rnn_fit.png"))
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
        output_path = os.path.join(RESULTS_DIR, "speciation_ego_rnn_fit.png")
        plt.savefig(output_path)
        print(f"✅ Speciation plot saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error during plotting: {e}")
        # Debug info
        print(f"   Curves shape: {curves.shape if hasattr(curves, 'shape') else 'Unknown'}")
    
    plt.close()


# --- EVALUATION ---
def eval_genome(genome, config):
    # 1. Creazione Rete RNN
    net = neat.nn.RecurrentNetwork.create(genome, config)
    
    try:
        env = OCAtari(GAME_NAME, mode="ram", hud=False, render_mode=None)
    except Exception:
        return 0.0
    
    env = SpaceInvadersEgocentricWrapper(env, skip=4)
    
    # Per sicurezza sulla casualità
    random.seed(os.getpid() + time.time())
    
    # --- PARTITA SINGOLA (FAST MODE) ---
    # Usiamo 1 sola partita per velocità, ma manteniamo il random delay
    # per evitare l'overfitting su una singola configurazione.
    
    # A. Reset e Random No-Ops
    observation, info = env.reset(seed=None)
    
    random_delay = random.randint(0, 30)
    for _ in range(random_delay):
        observation, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated: break
    
    if len(observation) != 19:
        env.close()
        return 0.0
    
    # Variabili Episodio
    episode_fitness = 0.0  
    steps = 0
    terminated = False
    truncated = False
    max_steps = 6000 
    
    # Reset RNN
    net.reset()

    while not (terminated or truncated) and steps < max_steps:
        outputs = net.activate(observation)
        action = np.argmax(outputs)
        
        # --- LOGICA DI FITNESS COMPLESSA (Survival + Aiming) ---
        
        # 1. ANALISI PERICOLO (Survival Layer)
        # obs[3] è il sensore CENTRALE (sopra la testa)
        # Valore 0.0 = Libero, 1.0 = Proiettile addosso
        danger_level = observation[3]
        is_safe = danger_level < 0.25  # Soglia di sicurezza
        
        # 2. Penalità Sparo (Anti-Spam) - Sempre attiva
        if action == 1:
            episode_fitness -= 0.05 
        
        if is_safe:
            # --- ZONA SICURA: Premia la Mira ---
            # Se non c'è pericolo, l'IA deve pensare ad attaccare
            rel_x = observation[11]
            if abs(rel_x) < 0.15: 
                episode_fitness += 0.02 # Bonus Mira ATTIVO
        else:
            # --- ZONA PERICOLO: Punisci la staticità ---
            # Se c'è un proiettile sopra, NON dare bonus mira.
            # Penalizza per ogni frame passato sotto il proiettile.
            episode_fitness -= (danger_level * 0.2) 

        # Step Ambiente
        observation, reward, terminated, truncated, info = env.step(action)
        
        # 3. Reward Principale (Uccisione) - Priorità Assoluta
        if reward > 0:
            episode_fitness += reward         
            episode_fitness += (reward * 0.5) # Bonus kill extra

        steps += 1
        
    # Bonus sopravvivenza minimo
    if episode_fitness <= 0:
        episode_fitness = max(0.001, steps / 10000.0)

    env.close()
    
    return episode_fitness

# --- MAIN ---
def run_training():
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Config non trovato: {CONFIG_PATH}")
        return

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)

    if config.genome_config.num_inputs != 19:
        print(f"❌ ERRORE CONFIG: num_inputs deve essere 19!")
        return

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10, filename_prefix=os.path.join(RESULTS_DIR, "neat-rnn-chk-")))

    # Usa tutti i core disponibili meno 2
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    pe = neat.ParallelEvaluator(num_workers, eval_genome)
    
    try:
        print(f"🚀 Avvio Training RNN (Survival + Aiming)...")
        winner = p.run(pe.evaluate, GENERATIONS)
        
        with open(os.path.join(RESULTS_DIR, 'winner_ego.pkl'), 'wb') as f:
            pickle.dump(winner, f)
        
        # Salvataggio Top 3
        all_genomes = list(p.population.values())
        all_genomes.sort(key=lambda g: g.fitness if g.fitness else 0.0, reverse=True)
        top_3 = all_genomes[:3]
        
        top3_path = os.path.join(RESULTS_DIR, 'top3_list.pkl')
        with open(top3_path, 'wb') as f:
            pickle.dump(top_3, f)
            
        print(f"💾 Salvato winner_ego.pkl e top3_list.pkl")
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