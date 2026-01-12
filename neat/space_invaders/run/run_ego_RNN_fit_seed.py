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

GAME_NAME = "ALE/SpaceInvaders-v5"
GENERATIONS = 300
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# --- NUOVE CONFIGURAZIONI SEED ---
TRAINING_SEED_MIN = 100      # Seed < 100 riservati al test
TRAINING_SEED_MAX = 100000   # Range ampio per il training
EPISODES_PER_GENOME = 3      # Media su 3 partite per robustezza

print(f"✅ Configurazione: {GAME_NAME} (RNN + Survival Logic + Seed {TRAINING_SEED_MIN}+)")

# --- PLOTTING (Invariato) ---
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
    print("📊 Generating Speciation Plot...")
    species_sizes = statistics.get_species_sizes()
    if not species_sizes:
        print("⚠️ No speciation data found.")
        return

    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    try:
        ax.stackplot(range(num_generations), *curves)
        plt.title("Evolution of Species (Speciation)")
        plt.ylabel("Number of Genomes per Species")
        plt.xlabel("Generations")
        plt.margins(0, 0)
        output_path = os.path.join(RESULTS_DIR, "speciation_ego_rnn_fit.png")
        plt.savefig(output_path)
        print(f"✅ Speciation plot saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error during plotting: {e}")
    plt.close()


# --- EVALUATION ---
def eval_genome(genome, config):
    # 1. Creazione Rete RNN
    net = neat.nn.RecurrentNetwork.create(genome, config)
    
    # Creazione ambiente (una volta per genoma per efficienza)
    try:
        # Nota: render_mode=None per velocità massima
        env = OCAtari(GAME_NAME, mode="ram", hud=False, render_mode=None)
    except Exception:
        return 0.0
    
    env = SpaceInvadersEgocentricWrapper(env, skip=4)
    
    # Per sicurezza sulla casualità nei processi paralleli
    random.seed(os.getpid() + time.time())
    
    fitness_history = []

    # --- CICLO EPISODI (3 Partite diverse) ---
    for episode in range(EPISODES_PER_GENOME):
        
        # A. Pesca un seed casuale dal range "sicuro" (>100)
        current_seed = random.randint(TRAINING_SEED_MIN, TRAINING_SEED_MAX)
        
        # B. Reset con il seed specifico
        observation, info = env.reset(seed=current_seed)
        
        # Random Delay (utile anche con seed fissi per variare leggermente lo start)
        random_delay = random.randint(0, 30)
        for _ in range(random_delay):
            observation, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated: break
        
        if len(observation) != 19:
            break # Errore wrapper
        
        # Variabili Episodio
        episode_fitness = 0.0  
        steps = 0
        terminated = False
        truncated = False
        max_steps = 6000 
        
        # IMPORTANTE: Reset dello stato interno della RNN a inizio partita
        net.reset()

        while not (terminated or truncated) and steps < max_steps:
            outputs = net.activate(observation)
            action = np.argmax(outputs)
            
            # --- LOGICA DI FITNESS ---
            danger_level = observation[3]
            is_safe = danger_level < 0.25 
            
            # Penalità Sparo
            if action == 1:
                episode_fitness -= 0.05 
            
            if is_safe:
                # Bonus Mira
                rel_x = observation[11]
                if abs(rel_x) < 0.15: 
                    episode_fitness += 0.02 
            else:
                # Penalità Pericolo
                episode_fitness -= (danger_level * 0.2) 

            # Step Ambiente
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Reward Kill
            if reward > 0:
                episode_fitness += reward         
                episode_fitness += (reward * 0.5)

            steps += 1
            
        # Bonus sopravvivenza minimo per l'episodio
        if episode_fitness <= 0:
            episode_fitness = max(0.001, steps / 10000.0)
            
        fitness_history.append(episode_fitness)

    env.close()
    
    # Ritorna la MEDIA delle partite
    if not fitness_history:
        return 0.0
    return np.mean(fitness_history)

# --- MAIN (Invariato) ---
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

    # Usa i worker definiti sopra
    pe = neat.ParallelEvaluator(NUM_WORKERS, eval_genome)
    
    try:
        print(f"🚀 Avvio Training RNN (Survival + Aiming) su {EPISODES_PER_GENOME} seed per genoma...")
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