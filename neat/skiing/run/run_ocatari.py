import os
import sys
from datetime import datetime
import neat
import pickle
import multiprocessing
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Aggiungi il percorso per importare il wrapper
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from wrapper.wrapper_ocatari import BioSkiingOCAtariWrapper
except ImportError:
    print("❌ Errore: Non trovo 'wrapper/wrapper_ocatari.py'")
    sys.exit(1)

# --- CONFIGURAZIONE ---
NUM_GENERATIONS = 70
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2) # Lascia 2 core liberi
CONFIG_FILENAME = "config_ocatari.txt"
CHECKPOINT_PREFIX = "neat-checkpoint-"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join(parent_dir, 'evolution_results', 'ocatari_run')

def eval_genome(genome, config):
    """
    Funzione eseguita da ogni worker in parallelo.
    Valuta un singolo genoma (cervello) e restituisce la fitness.
    """
    # Crea ambiente (OCAtari non ha bisogno di render durante il training)
    # env = BioSkiingOCAtariWrapper(render_mode=None) 
    # NOTA: Alcune versioni di OCAtari richiedono 'rgb_array' per funzionare internamente
    try:
        env = BioSkiingOCAtariWrapper(render_mode="rgb_array")
    except:
        env = BioSkiingOCAtariWrapper(render_mode=None)
        
    observation, info = env.reset()
    
    # Crea Rete Neurale (Recurrent per memoria)
    net = neat.nn.RecurrentNetwork.create(genome, config)
    
    done = False
    total_reward = 0.0
    steps = 0
    
    while not done and steps < 4000: # Max steps per sicurezza
        # Input (già normalizzati dal wrapper)
        inputs = observation
        
        # Check sicurezza dimensioni
        if len(inputs) != config.genome_config.num_inputs:
             print(f"⚠️ Mismatch: Wrapper dà {len(inputs)}, Config aspetta {config.genome_config.num_inputs}")
             return 0.0

        # Attivazione Rete
        output = net.activate(inputs)
        action = np.argmax(output) # 3 output: Su/Giu(nulla), Dx, Sx
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        done = terminated or truncated
        steps += 1
        
    env.close()
    return total_reward

def plot_results(stats, save_dir):
    """
    Genera grafici robusti leggendo direttamente generation_statistics.
    """
    print(f"\n📊 Generazione grafici in: {save_dir}")
    
    # --- 1. GRAFICO FITNESS ---
    if stats.most_fit_genomes:
        generation = range(len(stats.most_fit_genomes))
        best_fitness = [c.fitness for c in stats.most_fit_genomes]
        avg_fitness = np.array(stats.get_fitness_mean())
        stdev_fitness = np.array(stats.get_fitness_stdev())

        plt.figure(figsize=(10, 6))
        plt.plot(generation, avg_fitness, 'b-', label="Average Fitness", alpha=0.6)
        plt.fill_between(generation, avg_fitness - stdev_fitness, avg_fitness + stdev_fitness, 
                         color='blue', alpha=0.1)
        plt.plot(generation, best_fitness, 'r-', label="Best Fitness", linewidth=2)
        
        plt.title("Evoluzione Fitness Skiing")
        plt.xlabel("Generazioni")
        plt.ylabel("Fitness")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(os.path.join(save_dir, f"fitness_history_{timestamp}.png"), dpi=100)
        plt.close()
        print("✅ Fitness graph salvato.")

    # --- 2. GRAFICO SPECIAZIONE (FIXED) ---
    # Usiamo stats.generation_statistics che è una lista di dizionari {id_specie: OggettoSpecie}
    # Questo metodo è sicuro al 100%
    
    try:
        # Trova tutti gli ID di specie mai esistiti
        all_species = set()
        for gen_data in stats.generation_statistics:
            # gen_data è un dict: {species_id: species_object}
            all_species.update(gen_data.keys())
        
        all_species = sorted(list(all_species))
        
        # Costruisci la matrice della storia (Specie x Generazioni)
        species_history = []
        for gen_data in stats.generation_statistics:
            row = []
            for s_id in all_species:
                if s_id in gen_data:
                    # La dimensione è il numero di membri nella specie
                    row.append(len(gen_data[s_id].members))
                else:
                    row.append(0)
            species_history.append(row)
        
        # Trasponi per matplotlib (vuole Liste di Y per ogni specie)
        species_history = np.array(species_history).T
        
        if len(species_history) > 0:
            plt.figure(figsize=(10, 6))
            plt.stackplot(range(len(stats.generation_statistics)), species_history, 
                          labels=[f"ID {i}" for i in all_species])
            
            plt.title("Evoluzione delle Specie")
            plt.xlabel("Generazioni")
            plt.ylabel("Popolazione")
            
            # Legenda solo se poche specie per non coprire tutto
            if len(all_species) < 15:
                plt.legend(loc='upper left')
                
            plt.savefig(os.path.join(save_dir, f"speciation_{timestamp}.png"), dpi=100)
            plt.close()
            print("✅ Speciation graph salvato.")
            
    except Exception as e:
        print(f"⚠️ Errore durante plot speciazione: {e}")

def run_training():
    # 1. Setup Percorsi
    config_path = os.path.join(parent_dir, 'config', CONFIG_FILENAME)
    results_dir = os.path.join(parent_dir, 'evolution_results', 'ocatari_run')
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"🚀 Avvio Training OCAtari")
    print(f"📂 Config: {config_path}")
    print(f"📂 Output: {results_dir}")
    print(f"⚙️  Workers: {NUM_WORKERS}")

    # 2. Carica Configurazione
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    # Check veloce input
    if config.genome_config.num_inputs != 9:
        print(f"⚠️  ATTENZIONE: Il wrapper OCAtari usa 9 input.")
        print(f"    Nel config c'è scritto {config.genome_config.num_inputs}.")
        print("    Aggiorna config_wrapper.txt a 'num_inputs = 9'!")
        # sys.exit(1) # Decommenta per bloccare se sbagliato

    # 3. Inizializza Popolazione
    # Cerca checkpoint esistenti per riprendere (Opzionale)
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-X')
    p = neat.Population(config)

    # 4. Aggiungi Reporter (Statistiche a video)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=5, filename_prefix=CHECKPOINT_PREFIX))

    # 5. Esegui Evoluzione Parallela
    pe = neat.ParallelEvaluator(NUM_WORKERS, eval_genome)
    
    # RUN!
    winner = p.run(pe.evaluate, NUM_GENERATIONS)

    # 6. Salva il Vincitore
    print(f"\n🏆 Miglior genoma trovato! Fitness: {winner.fitness}")
    #save_path = os.path.join(results_dir, "best_agent_ocatari_{timestamp}.pkl")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(results_dir, f"best_agent_ocatari_{timestamp}.pkl")
    
    with open(save_path, 'wb') as f:
        pickle.dump(winner, f)
    
    print(f"💾 Salvato in: {save_path}")
    #results_dir = os.path.join(parent_dir, 'evolution_results', 'ocatari_run')
    plot_results(stats, results_dir)

if __name__ == "__main__":
    # Fix per multiprocessing su Windows
    multiprocessing.freeze_support()
    run_training()