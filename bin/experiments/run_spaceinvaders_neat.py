import sys
import os
import neat
import numpy as np
import pickle
import datetime
import multiprocessing
from neat.reporting import BaseReporter 

# --- SETUP PATH ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import moduli custom
from core.env_factory import make_evolution_env 
from utils.neat_plotting_utils import plot_stats, plot_species

# --- PARAMETRI ---
ENV_NAME = "spaceinvaders" 
CONFIG_FILE_NAME = "neat_spaceinvaders_config.txt"
NUM_GENERATIONS = 100       
MAX_STEPS = 4000           
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# Percorsi
OUTPUT_DIR = os.path.join(project_root, "evolution_results", "spaceinvaders")
CONFIG_PATH = os.path.join(project_root, "configs", CONFIG_FILE_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def eval_genome_worker(genome, config):
    """
    Worker parallelo con FITNESS SHAPING CORRETTO (Fix variable order).
    """
    # 1. Crea Rete
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # 2. Crea Ambiente
    try:
        env = make_evolution_env(ENV_NAME, render_mode=None)
    except Exception as e:
        print(f"Errore creazione env: {e}")
        return 0.0

    # 3. Reset iniziale
    observation, info = env.reset()
    
    total_reward = 0.0      # Score del gioco
    fitness_bonus = 0.0     # Bonus didattici
    
    steps = 0
    last_x_pos = 0.5
    frames_still = 0
    
    done = False
    
    while not done and steps < MAX_STEPS:
        # --- A. DECISIONE ---
        output = net.activate(observation)
        action = np.argmax(output) 
        
        # Salviamo lo stato PRE-azione per valutare movimento e mira
        player_x = observation[0]
        is_aligned = (observation[7] > 0.5)

        # --- B. STEP FISICO (PRIMA DI CALCOLARE I BONUS!) ---
        # Qui otteniamo il 'reward' reale dal gioco
        next_observation, reward, terminated, truncated, info = env.step(action)
        
        # --- C. FITNESS SHAPING ---
        
        # 1. Kill Bonus (Score reale)
        if reward > 0:
            total_reward += reward
            # Moltiplicatore x2 per incentivare l'uccisione reale
            fitness_bonus += (reward * 2.0) 
            
        # 2. Bonus Cecchino (Solo scuola guida)
        # Lo diamo solo se non ha ancora fatto punti veri per insegnargli a sparare
        if total_reward == 0 and action == 3 and is_aligned:
             fitness_bonus += 0.05 
             
        # 3. Penalità Movimento
        if abs(player_x - last_x_pos) < 0.001:
            frames_still += 1
        else:
            frames_still = 0
            
        if frames_still > 30: 
            fitness_bonus -= 2.0 # Penalità severa per camping

        # Aggiornamento stato per il prossimo loop
        last_x_pos = player_x
        observation = next_observation # Importante: passiamo al frame successivo
        
        steps += 1
        done = terminated or truncated

    env.close()
    
    # --- CALCOLO FITNESS FINALE ---
    final_fitness = total_reward + fitness_bonus
    
    # Penalità morte istantanea
    if steps < 50:
        final_fitness -= 50
        
    return max(0.0, final_fitness)


# === CLASSE PER GESTIONE DINAMICA ===
class DynamicMutationReporter(BaseReporter):
    def __init__(self, switch_threshold=1800.0): 
        self.switch_threshold = switch_threshold
        self.mode = "EXPLORATION"
        self.generation = 0

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        current_best_fitness = best_genome.fitness
        
        print(f"   >>> [DynamicStats] Best Fitness: {current_best_fitness:.1f} | Mode: {self.mode}")

        # --- LOGICA DI SWITCH ---
        if self.mode == "EXPLORATION" and current_best_fitness > self.switch_threshold:
            print(f"\n🚨 SOGLIA ({self.switch_threshold}) SUPERATA! Passaggio a EXPLOITATION (Cecchino) 🚨\n")
            self.mode = "EXPLOITATION"
            
            # Congela struttura
            config.genome_config.node_add_prob = 0.0
            config.genome_config.conn_add_prob = 0.0
            config.genome_config.conn_delete_prob = 0.0
            
            # Raffina pesi
            config.genome_config.weight_mutate_power = 0.05 
            config.genome_config.bias_mutate_power = 0.05
            
            # Selezione dura
            config.reproduction_config.survival_threshold = 0.1 

        elif self.mode == "EXPLOITATION" and current_best_fitness < (self.switch_threshold * 0.8):
            print("\n🔙 PRESTAZIONI CALATE! Ritorno a EXPLORATION 🔙\n")
            self.mode = "EXPLORATION"
            config.genome_config.weight_mutate_power = 0.8
            config.genome_config.node_add_prob = 0.15
            config.genome_config.conn_add_prob = 0.3
            config.reproduction_config.survival_threshold = 0.2


def run_evolution():
    print(f"--- 👾 Avvio Evoluzione SPACE INVADERS (Parallel + Shaped Fitness Corrected) ---")
    print(f"--- 🚀 Core: {NUM_WORKERS} | Pop: {250} | Gen: {NUM_GENERATIONS}")
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)

    p = neat.Population(config)
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix=os.path.join(OUTPUT_DIR, "checkpoint-")))
    
    # Soglia impostata a 350 (circa 5-6 alieni uccisi)
    p.add_reporter(DynamicMutationReporter(switch_threshold=1800.0))
    
    pe = neat.ParallelEvaluator(NUM_WORKERS, eval_genome_worker)
    
    try:
        winner = p.run(pe.evaluate, NUM_GENERATIONS)
        
        timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        winner_path = os.path.join(OUTPUT_DIR, f"best_genome_si_{timestamp_str}.pkl")
        
        with open(winner_path, 'wb') as f:
            pickle.dump(winner, f)
            
        print(f"\n🏆 WINNER SAVED: {winner_path}")
        
        plot_stats(stats, ylog=False, filename=os.path.join(OUTPUT_DIR, f"stats_{timestamp_str}.png"))
        plot_species(stats, filename=os.path.join(OUTPUT_DIR, f"species_{timestamp_str}.png"))
        
    except KeyboardInterrupt:
        print("\n🛑 Stop manuale.")
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_evolution()