# File: run/run_ski_neat.py

import sys
import os
import neat
import numpy as np
import pickle
import datetime
import multiprocessing
import gymnasium as gym

# --- IMPORTA IL WRAPPER PERSONALIZZATO ---
# Assume la struttura: project_root/wrapper/skiing_wrapper.py
script_dir = os.path.dirname(os.path.abspath(__file__))

# La directory radice del SOTTO-PROGETTO NEAT è la cartella 'skiing' (un livello sopra 'run')
project_root = os.path.abspath(os.path.join(script_dir, '..')) 

if project_root not in sys.path:
    # Aggiunge la cartella 'skiing' al PYTHONPATH
    sys.path.insert(0, project_root)

try:
    from wrapper.skiing_wrapper import SkiingCustomWrapper
except ImportError:
    print("❌ Errore nell'importazione: Assicurati che 'wrapper/skiing_wrapper.py' esista e sia accessibile.")
    sys.exit(1)
# ------------------------------------------

# IMPORTANTE: Registra gli ambienti ALE
try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    print("❌ ale-py non installato. Installa con: pip install ale-py")
    sys.exit(1)

# --- CONFIGURAZIONE ESPERIMENTO ---
ENV_ID = "ALE/Skiing-v5"
CONFIG_FILE_NAME = "neat_ski_config.txt"
NUM_GENERATIONS = 15 # Aumentato per un test più significativo
MAX_STEPS = 20000
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

OUTPUT_DIR = os.path.join(project_root, "evolution_results", "skiing")
CONFIG_PATH = os.path.join(project_root, "config", CONFIG_FILE_NAME) # Aggiornato path
os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_ram(ram_state):
    """Normalizza lo stato RAM (128 bytes) in [0, 1]"""
    return np.array(ram_state, dtype=np.float32) / 255.0


def calculate_fitness(wrapped_env, net, max_steps=MAX_STEPS, debug=False):
    """
    🎯 FITNESS per Skiing basata su reward totale + penalità del wrapper.
    
    Fitness calcolata: 
    F_minimizzazione = abs(total_reward) + steering_cost + stability_penalty
    F_massimizzazione = MAX_FITNESS_CAP - F_minimizzazione
    """
    try:
        # Usa il reset del wrapper
        observation, info = wrapped_env.reset() 
    except Exception as e:
        if debug:
            print(f"⚠️ Errore reset: {e}")
        return 100000.0
    
    done = False
    steps = 0
    total_reward = 0.0

    terminated = False
    truncated = False
    
    while not done and steps < max_steps:
        try:
            norm_obs = normalize_ram(observation)
            output = net.activate(norm_obs)
            # L'azione è 0, 1, o 2 (come definito dal wrapper)
            action = int(np.argmax(output)) 
            
            # Lo step usa l'ambiente wrapped, che gestisce la mappatura
            observation, reward, terminated, truncated, info = wrapped_env.step(action)
            
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
        except Exception as e:
            if debug:
                print(f"⚠️ Errore step {steps}: {e}")
            break
    
    # --- CALCOLO FITNESS AGGREGATO ---
    
    # 1. Fitness Base (Target: MINIMIZZAZIONE)
    if terminated:
        fitness_min = abs(total_reward)
    elif steps >= max_steps:
        # Penalizza pesantemente il timeout
        fitness_min = abs(total_reward) + 100000.0
    else:
        # Errore o interruzione anomala
        fitness_min = 1000000.0
        
    # 2. Aggiungi Penalità dal Wrapper
    steering_cost = wrapped_env.steering_cost
    stability_penalty = wrapped_env.get_stability_penalty()

    fitness_min += steering_cost 
    fitness_min += stability_penalty
    fitness_min += wrapped_env.edge_cost
    MOVEMENT_BONUS_MULTIPLIER = 5.0 
    
    movement_bonus = wrapped_env.total_x_movement * MOVEMENT_BONUS_MULTIPLIER
    fitness_min -= movement_bonus # <--- Sottrai il bonus!
    # 3. Conversione a fitness massimizzata (Obiettivo NEAT)
    MAX_FITNESS_CAP = 150000.0 
    final_inverted_fitness = MAX_FITNESS_CAP - fitness_min

    if debug:
        print(f"Steps: {steps}, Terminated: {terminated}")
        print(f"Total Reward: {total_reward}")
        print(f"Bonus Movimento X: -{movement_bonus:.2f}")
        # Usa .unwrapped per accedere ai dati di info originali
        print(f"Lives remaining: {wrapped_env.unwrapped.ale.lives()}") 
        print(f"Cost Summary: Steering={steering_cost:.2f}, Stability={stability_penalty:.2f}")
        print(f"Fitness (MINIMIZATION): {fitness_min:.2f}")
        print(f"Fitness finale invertita (per MAXIMIZATION): {final_inverted_fitness:.2f}")
    
    return float(final_inverted_fitness)

def eval_single_genome(genome, config):
    """Valuta un singolo genoma NEAT."""
    try:
        # 1. Crea l'ambiente base
        env = gym.make(
            ENV_ID,
            obs_type="ram",
            frameskip=1,
            repeat_action_probability=0.0,
            render_mode=None
        )
        
        # 2. WRAPPA L'AMBIENTE e imposta i tuoi parametri di penalità
        # QUI puoi modificare i valori delle penalità per testare le strategie!
        wrapped_env = SkiingCustomWrapper(
            env, 
            enable_steering_cost=False,             # Abilita/Disabilita costo sterzo
            min_change_ratio=0.05,                 # Minima percentuale di azioni non-NOOP richieste
            steering_cost_per_step=1.0,             # Penalità per ogni passo di sterzo,           
            edge_penalty_multiplier=30.0, # AUMENTA QUESTO VALORE per punire i bordi
            edge_threshold=40
        )
        
        # ✅ DEBUG solo per il primo genoma
        debug = not hasattr(eval_single_genome, '_first_eval_done')
        
        if debug:
            print("\n" + "="*80)
            print("🔍 DEBUG PRIMO GENOMA")
            print(f"🎮 Environment: {ENV_ID}")
            print(f"🎮 Action Space (Wrapper): {wrapped_env.action_space}")
            print(f"🎮 Observation Space: {wrapped_env.observation_space}")
            print(f"🎮 Azioni Mappate: 0->NOOP, 1->RIGHT, 2->LEFT")
            print("="*80)
            eval_single_genome._first_eval_done = True
        
        # Creazione rete NEAT
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Valutazione
        fitness = calculate_fitness(wrapped_env, net, max_steps=MAX_STEPS, debug=debug)
        
        if debug:
            print(f"🎯 Fitness finale primo genoma: {fitness}")
            print("="*80 + "\n")
        
        # Chiudi l'ambiente sottostante
        env.close()
        return fitness
        
    except Exception as e:
        print(f"❌ Errore in eval_single_genome: {e}")
        import traceback
        traceback.print_exc()
        # Ritorna un fitness molto basso per eliminare il genoma
        return -1000.0 # Valore negativo dato che si MAXIMIZZA

def run_evolution():
    """Funzione principale per eseguire l'evoluzione NEAT."""
    print("=" * 80)
    print("🧬 NEAT EVOLUTION - SKIING (FeedForward, Gymnasium RAM)")
    print("=" * 80)
    print(f"📁 Config: {CONFIG_PATH}")
    print(f"📊 Output: {OUTPUT_DIR}")
    print(f"⚙️ Workers: {NUM_WORKERS}")
    print(f"🔄 Generazioni: {NUM_GENERATIONS}")
    print(f"⏱ Max Steps: {MAX_STEPS}")
    print(f"🎯 Obiettivo: MINIMIZZARE tempo di discesa")
    print(f"🎮 Ambiente: {ENV_ID}")
    print("=" * 80)
    
    # Caricamento config NEAT
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"❌ Config file non trovato: {CONFIG_PATH}")
    
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )
    
    # Parallel evaluator
    pe = neat.ParallelEvaluator(NUM_WORKERS, eval_single_genome)
    
    # Inizializzazione popolazione
    population = neat.Population(config)
    
    # Reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    # Checkpointer
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_prefix = os.path.join(OUTPUT_DIR, f"checkpoint_{timestamp}_")
    population.add_reporter(neat.Checkpointer(
        generation_interval=5,
        filename_prefix=checkpoint_prefix
    ))
    
    # Evoluzione
    print("\n🚀 Avvio evoluzione...\n")

    try:
        winner = population.run(pe.evaluate, NUM_GENERATIONS)
        
        # Salvataggio risultati
        winner_path = os.path.join(OUTPUT_DIR, f"winner_{timestamp}.pkl")
        with open(winner_path, 'wb') as f:
            pickle.dump((winner, config), f)
        print(f"\n✅ Winner salvato: {winner_path}")
        
        # Statistiche finali
        print("\n" + "=" * 80)
        print("🏆 EVOLUZIONE COMPLETATA")
        print("=" * 80)
        
        # La fitness è massimizzata (più alta è, meglio è)
        # La fitness massimizzata è: 150000.0 - F_minimizzazione
        # Quindi F_minimizzazione = 150000.0 - winner.fitness
        best_time_approx = 150000.0 - winner.fitness
        
        print(f"Best Fitness (Maximized): {winner.fitness:.2f}")
        print(f"Approx. Minimum Cost (Time + Penalties): {best_time_approx:.2f}")
        print(f"Best Genome ID: {winner.key}")
        print("=" * 80)

        return winner, config, stats
    
    except KeyboardInterrupt:
        print("\n\n⏸ Evoluzione interrotta dall'utente")
        return None, config, None
    
    except Exception as e:
        print(f"\n\n❌ Errore fatale durante evoluzione: {e}")
        import traceback
        traceback.print_exc()
        return None, config, None


if __name__ == "__main__":
    # Metodo di avvio multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    # Esegui evoluzione
    winner, config, stats = run_evolution()
    
    if winner is not None:
        print("\n✨ Evoluzione completata con successo!")