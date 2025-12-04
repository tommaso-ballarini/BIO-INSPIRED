import sys
import os
import neat
import numpy as np
import pickle
import datetime
import multiprocessing
import gymnasium as gym

# IMPORTANTE: Registra gli ambienti ALE
try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    print("❌ ale-py non installato. Installa con: pip install ale-py")
    sys.exit(1)

# --- SETUP PATHS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- CONFIGURAZIONE ESPERIMENTO ---
ENV_ID = "ALE/Skiing-v5"
CONFIG_FILE_NAME = "neat_ski_config.txt"
NUM_GENERATIONS = 5
MAX_STEPS = 20000
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

OUTPUT_DIR = os.path.join(project_root, "evolution_results", "skiing")
CONFIG_PATH = os.path.join(project_root, "configs", CONFIG_FILE_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_ram(ram_state):
    """Normalizza lo stato RAM (128 bytes) in [0, 1]"""
    return np.array(ram_state, dtype=np.float32) / 255.0


def calculate_fitness(env, net, max_steps=MAX_STEPS, debug=False):
    """
    🎯 FITNESS per Skiing basata su reward totale
    
    In Skiing:
    - reward è negativo (rappresenta tempo/penalità)
    - Più negativo = più tempo = peggio
    - Goal: MINIMIZZARE abs(total_reward)
    """
    try:
        observation, info = env.reset()
    except Exception as e:
        return 100000.0
    
    done = False
    steps = 0
    total_reward = 0.0

    last_action = -1
    actions_changed = 0

    STEERING_COST_PER_STEP = 1.0 # Costo per ogni passo di sterzo
    steering_cost = 0.0
    
    while not done and steps < max_steps:
        try:
            norm_obs = normalize_ram(observation)
            output = net.activate(norm_obs)
            action = int(np.argmax(output)) % 3
            action = int(np.argmax(output)) % 3 # Azione 0, 1, o 2
    
    # 🎯 NUOVO: Aggiungi un costo se l'azione è RIGHT (1) o LEFT (2)
            if action == 1 or action == 2:
                steering_cost += STEERING_COST_PER_STEP
            # ... dopo aver selezionato 'action' ...
            # Check for action change
            if steps > 0 and action != last_action:
                actions_changed += 1
            last_action = action
            # ...
            
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
        except Exception as e:
            if debug:
                print(f"⚠️ Errore step {steps}: {e}")
            break
    
    # ✅ CALCOLO FITNESS CORRETTO
    # In Skiing, reward negativo = tempo impiegato
    # Vogliamo MINIMIZZARE il tempo (= minimizzare abs(reward))
    
    if terminated:
        fitness = abs(total_reward)
    # ❌ RIMUOVI O COMMENTA QUESTO BLOCCO:
    # if steps < 1000:
    #     fitness += 50000.0
            
    elif steps >= max_steps:
        # Timeout: non ha finito in tempo
        # Penalizza pesantemente ma premia progressi parziali
        fitness = abs(total_reward) + 100000.0
        
    else:
        # Errore o interruzione anomala
        fitness = 100000.0
        
        # --- APPLICA PENALITÀ PER STABILITÀ DELL'AZIONE (PASSIVITÀ) ---
    stability_penalty = 0.0

    if steps > 1:
        change_ratio = actions_changed / (steps - 1) 
        MIN_CHANGE_RATIO = 0.05 

        if change_ratio < MIN_CHANGE_RATIO:
            BASE_PENALTY = 50000.0
            stability_penalty = BASE_PENALTY * (1.0 - (change_ratio / MIN_CHANGE_RATIO))
            stability_penalty = max(100.0, stability_penalty)

    fitness += stability_penalty
    fitness += steering_cost 

    if debug:
        print(f"Cumulative Steering Cost: {steering_cost:.2f}")
    # ...
    MAX_FITNESS_CAP = 150000.0 
    
    # Ora: minimizzare F diventa massimizzare (C - F)
    # F = 30000 (buono) -> Return 120000.0 (alto)
    # F = 80000 (cattivo) -> Return 70000.0 (basso)
    final_inverted_fitness = MAX_FITNESS_CAP - fitness

    if debug:
        print(f"Steps: {steps}, Terminated: {terminated}")
        print(f"Total Reward: {total_reward}")
        print(f"Lives remaining: {info.get('lives', 'N/A')}")
        print(f"Fitness: {fitness}")
        print(f"Fitness finale invertita (per MAXIMIZATION): {final_inverted_fitness:.2f}")
    
    return float(final_inverted_fitness)

def eval_single_genome(genome, config):
    """Valuta un singolo genoma NEAT."""
    try:
        env = gym.make(
            ENV_ID,
            obs_type="ram",
            frameskip=1,
            repeat_action_probability=0.0,
            render_mode=None
        )
        
        # ✅ DEBUG solo per il primo genoma
        debug = not hasattr(eval_single_genome, '_first_eval_done')
        
        if debug:
            print("\n" + "="*80)
            print("🔍 DEBUG PRIMO GENOMA")
            print(f"🎮 Environment: {ENV_ID}")
            print(f"🎮 Action Space: {env.action_space}")
            print(f"🎮 Observation Space: {env.observation_space}")
            print(f"🎮 Azioni disponibili: {env.unwrapped.get_action_meanings()}")
            print("="*80)
            eval_single_genome._first_eval_done = True
        
        # Creazione rete NEAT
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Valutazione con debug per primo genoma
        fitness = calculate_fitness(env, net, max_steps=MAX_STEPS, debug=debug)
        
        if debug:
            print(f"🎯 Fitness finale primo genoma: {fitness}")
            print("="*80 + "\n")
        
        env.close()
        return fitness
        
    except Exception as e:
        print(f"❌ Errore in eval_single_genome: {e}")
        import traceback
        traceback.print_exc()
        return 100000.0

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
        print(f"Best Time: {winner.fitness:.2f} centesimi di secondo")
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