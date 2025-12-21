import sys
import os
import pickle
import numpy as np
import multiprocessing
import neat
import matplotlib.pyplot as plt
from ocatari.core import OCAtari

# --- GESTIONE PERCORSI E CARTELLE ---
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../space_invaders/run
parent_dir = os.path.dirname(current_dir)               # .../space_invaders

# Aggiunge il path per trovare il wrapper
sys.path.append(parent_dir)

# 1. Definizione Cartella Risultati
RESULTS_DIR = os.path.join(parent_dir, 'results_rnn_grid16x16')       # .../space_invaders/results
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')

# 2. Creazione Cartelle (se non esistono)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 3. Definizione File di Output
BEST_EVER_PATH = os.path.join(RESULTS_DIR, 'best_ever_champion.pkl')
FINAL_WINNER_PATH = os.path.join(RESULTS_DIR, 'final_winner.pkl')
GRAPH_PATH = os.path.join(RESULTS_DIR, 'fitness_graph.png')
CONFIG_PATH = os.path.join(parent_dir, 'config', 'config_si_wrapper_RNN.txt')

# Import Wrapper
try:
    from wrapper.wrapper_si_grid import SpaceInvadersGridWrapper
except ImportError:
    print("ERRORE: Non trovo il file wrapper_si_grid.py nella cartella wrapper!")
    sys.exit(1)

# --- CONFIGURAZIONE GIOCO ---
GAME_NAME = "SpaceInvadersNoFrameskip-v4" 

# --- CLASS SAVER (Salva il record assoluto in results/) ---
class SaveBestGenome(neat.reporting.BaseReporter): 
    def __init__(self):
        self.best_fitness = -1.0

    def post_evaluate(self, config, population, species, best_genome):
        if best_genome.fitness > self.best_fitness:
            print(f"\n🔥 NUOVO RECORD: {best_genome.fitness:.2f} (Salvato in results)")
            self.best_fitness = best_genome.fitness
            with open(BEST_EVER_PATH, 'wb') as f:
                pickle.dump(best_genome, f)

# --- VALUTAZIONE ---
def eval_genome(genome, config):
    # Setup Ambiente Veloce
    env = OCAtari(GAME_NAME, mode="ram", hud=False, render_mode=None)
    # Skip=4 per velocità e stabilità
    env = SpaceInvadersGridWrapper(env, grid_shape=(16, 16), skip=4)
    
    net = neat.nn.RecurrentNetwork.create(genome, config)
    
    observation, _ = env.reset()
    total_reward = 0.0
    steps = 0
    
    max_steps = 2000 
    steps_without_points = 0
    max_idle_steps = 200
    
    terminated = False
    truncated = False
    
    while not (terminated or truncated) and steps < max_steps:
        inputs = observation
        outputs = net.activate(inputs)
        action = np.argmax(outputs)
        
        # Wrapper gestisce lo skip di 4 frame
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        if reward > 0:
            steps_without_points = 0
        else:
            steps_without_points += 1
            
        if steps_without_points > max_idle_steps:
            break 
            
    env.close()
    
    # Fitness Function
    return total_reward + (steps * 0.001)

# --- GRAFICI (Salvati in results/) ---
def plot_stats(stats):
    generation = range(len(stats.get_fitness_mean()))
    best_fitness = [c.fitness for c in stats.most_fit_genomes]
    avg_fitness = stats.get_fitness_mean()

    plt.figure(figsize=(10, 5))
    plt.plot(generation, best_fitness, 'r-', label="Best Fitness")
    plt.plot(generation, avg_fitness, 'b-', label="Average Fitness")
    plt.title("Progresso Fitness Space Invaders")
    plt.xlabel("Generazioni")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend()
    
    print(f"Salvataggio grafico in: {GRAPH_PATH}")
    plt.savefig(GRAPH_PATH)
    plt.close()

def run():
    if not os.path.exists(CONFIG_PATH):
        print(f"ERRORE CRITICO: Config non trovato in {CONFIG_PATH}")
        return

    print(f"--- INIZIO TRAINING ---")
    print(f"Output salvati in: {RESULTS_DIR}")
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)

    p = neat.Population(config)

    # Reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # Checkpointer salvato nella sottocartella results/checkpoints
    p.add_reporter(neat.Checkpointer(generation_interval=10, 
                                     filename_prefix=os.path.join(CHECKPOINT_DIR, 'neat-checkpoint-')))
    
    # Custom Saver
    p.add_reporter(SaveBestGenome())

    # Esecuzione
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count() - 1, eval_genome)
    
    try:
        winner = p.run(pe.evaluate, n=20) # Imposta qui il numero di generazioni (es. 50 o 100)
        
        # Salva il vincitore dell'ultima generazione
        with open(FINAL_WINNER_PATH, 'wb') as f:
            pickle.dump(winner, f)
            
        plot_stats(stats)
        print(f"\n✅ Training Completato con Successo.")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrotto dall'utente. I dati parziali sono comunque salvi in results/")

if __name__ == '__main__':
    run()