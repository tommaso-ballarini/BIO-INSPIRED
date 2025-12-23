import os
import sys
import neat
import pickle
import multiprocessing
import gymnasium as gym
import numpy as np

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
NUM_GENERATIONS = 20
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2) # Lascia 2 core liberi
CONFIG_FILENAME = "config_ocatari.txt"
CHECKPOINT_PREFIX = "neat-checkpoint-"

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
    save_path = os.path.join(results_dir, "best_agent_ocatari.pkl")
    
    with open(save_path, 'wb') as f:
        pickle.dump(winner, f)
    
    print(f"💾 Salvato in: {save_path}")

if __name__ == "__main__":
    # Fix per multiprocessing su Windows
    multiprocessing.freeze_support()
    run_training()