# FILE: experiments/run_pacman_neat.py
"""
Script di esecuzione NEAT per Pacman (ALE) con estrazione feature ottimizzata.

Implementa le strategie descritte nel documento tecnico:
- Feature engineering tramite OCAtari (REM - RAM Extraction Method)
- Fitness function avanzata con anti-camping e exploration
- Parallelizzazione per efficienza computazionale
- Checkpointing e visualizzazione risultati
"""

import sys
import os
import neat
import numpy as np
import pickle
import datetime
import multiprocessing
import math
from itertools import chain

# --- MONKEY PATCH OCATARI (CRITICO) ---
# Fix per bug noto di OCAtari con oggetti None
import ocatari.core

def patched_ns_state(self):
    valid_objects = [o for o in self.objects if o is not None and hasattr(o, '_nsrepr')]
    return list(chain.from_iterable([o._nsrepr for o in valid_objects]))

ocatari.core.OCAtari.ns_state = patched_ns_state
# ---------------------------------------

# --- SETUP PATHS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ocatari.core import OCAtari
from core.wrappers_pacman import PacmanFeatureWrapper
from utils.neat_plotting_utils import plot_stats, plot_species

# --- CONFIGURAZIONE ESPERIMENTO ---
ENV_ID = "Pacman"  # Nome gioco OCAtari (non ALE/Pacman-v5)
CONFIG_FILE_NAME = "neat_pacman_config.txt"
NUM_GENERATIONS = 50  # Generazioni di evoluzione
MAX_STEPS = 3000      # Step massimi per episodio
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)  # Parallelizzazione

# Directory output
OUTPUT_DIR = os.path.join(project_root, "evolution_results", "pacman")
CONFIG_PATH = os.path.join(project_root, "configs", CONFIG_FILE_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_fitness_advanced(env, net, max_steps=MAX_STEPS):
    """
    Fitness Function Avanzata v5.0
    
    Implementa le strategie del documento tecnico (Sezione 6.1.1):
    - Reward shaping per esplorazione
    - Penalità per camping/stuck behavior
    - Bonus per raccolta pellet e ghost hunting
    - Gestione stato "edible ghost" (critico per policy switching)
    
    Returns:
        fitness (float): Score totale dell'episodio
        metrics (dict): Metriche dettagliate per debug
    """
    
    observation, info = env.reset()
    done = False
    steps = 0
    
    # --- Accumulatori Fitness ---
    fitness = 0.0
    game_score = 0.0  # Score reale del gioco
    
    # --- Tracking Comportamento ---
    visited_sectors = set()       # Mappa settori visitati (griglia 20x20)
    steps_without_progress = 0    # Anti-camping counter
    pellets_eaten = 0
    ghosts_eaten = 0
    power_pills_collected = 0
    
    # --- Memoria Stato ---
    prev_player_pos = None
    prev_pellet_count = 244  # Approssimativo per Pacman standard
    
    while not done and steps < max_steps:
        # --- ATTIVAZIONE RETE NEAT ---
        output = net.activate(observation)
        action = np.argmax(output)
        
        # --- STEP AMBIENTE ---
        try:
            observation, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            print(f"⚠️ Errore durante step: {e}")
            break
        
        # --- ESTRAZIONE FEATURES DALL'OSSERVAZIONE ---
        # Il vettore observation è strutturato (vedi PacmanFeatureWrapper)
        player_x = observation[0]  # Normalizzato [0,1]
        player_y = observation[1]
        
        # Distanze minime dai ghosts (feature 30-31)
        min_dist_dangerous = observation[30]
        min_dist_edible = observation[31]
        
        # Pellet rimanenti (feature 32)
        pellets_remaining = observation[32]
        
        # Ghosts edibility (feature 26-29)
        ghost_edible_states = observation[26:30]
        any_ghost_edible = np.any(ghost_edible_states > 0.5)
        
        # --- CALCOLO FITNESS COMPONENTI ---
        
        # 1. SOPRAVVIVENZA BASE (incoraggia a non morire subito)
        fitness += 0.05
        
        # 2. ESPLORAZIONE (Griglia virtuale 20x20 per tracking)
        sector_x = int(player_x * 20)
        sector_y = int(player_y * 20)
        current_sector = (sector_x, sector_y)
        
        explored_new = False
        if current_sector not in visited_sectors:
            visited_sectors.add(current_sector)
            fitness += 3.0  # Bonus esplorazione
            explored_new = True
            steps_without_progress = 0
        
        # 3. GAME SCORE (Reward principale da Atari)
        if reward > 0:
            # Scaling aggressivo per prioritizzare score reale
            fitness += reward * 25.0
            game_score += reward
            steps_without_progress = 0
            
            # Tracking pellet mangiati
            if reward == 10:  # Pellet standard
                pellets_eaten += 1
            elif reward == 50:  # Power pill
                power_pills_collected += 1
            elif reward >= 200:  # Ghost mangiato
                ghosts_eaten += 1
        
        # 4. PELLET COLLECTION PROGRESS
        # Rileva quando i pellet diminuiscono anche senza reward esplicito
        if prev_pellet_count > pellets_remaining:
            delta = prev_pellet_count - pellets_remaining
            fitness += delta * 244 * 5.0  # Denormalizza e scala
        prev_pellet_count = pellets_remaining
        
        # 5. GHOST HUNTING BONUS (quando edible)
        if any_ghost_edible and min_dist_edible < 0.1:
            # Pacman è vicino a un ghost commestibile
            fitness += 2.0  # Incoraggia inseguimento
        
        # 6. DANGER AVOIDANCE (quando ghosts pericolosi)
        if not any_ghost_edible and min_dist_dangerous < 0.1:
            # Troppo vicino a ghost pericoloso
            fitness -= 1.0  # Leggera penalità per rischio
        
        # 7. ANTI-CAMPING / STARVATION
        if not explored_new and reward == 0:
            steps_without_progress += 1
        
        # Stuck detection: se fermo troppo a lungo
        if prev_player_pos is not None:
            pos_delta = np.abs(player_x - prev_player_pos[0]) + np.abs(player_y - prev_player_pos[1])
            if pos_delta < 0.005:  # Movimento quasi nullo
                    # 💡 Correzione: usare steps_without_progress, non self.stuck_counter
                steps_without_progress += 1  # Penalità ridotta (era += 2)
            else:
                    # 💡 Correzione: non esiste self.stuck_counter qui. 
                    # Se c'è progresso, si resetta il contatore di blocco:
                steps_without_progress = 0
        
        prev_player_pos = (player_x, player_y)
        
        # Termina se bloccato troppo a lungo
        if steps_without_progress > 600:
            fitness -= 20.0  # Penalità severa
            done = True
            print(f"⏸ Episodio terminato: STUCK (step {steps})")
        
        # 8. COMPLETION BONUS
        # Se mangia >80% dei pellet, bonus sostanziale
        if pellets_remaining < 0.2:  # <20% rimanenti
            fitness += 100.0
        
        # --- AGGIORNAMENTO CONTATORI ---
        steps += 1
        done = done or terminated or truncated
    
    # --- PENALITÀ MORTE PRECOCE ---
    if steps < 300:
        fitness *= 0.5  # Dimezza fitness se muore troppo presto
    
    # --- METRICHE FINALI ---
    metrics = {
        "fitness": max(0.0, fitness),
        "game_score": game_score,
        "steps_survived": steps,
        "sectors_explored": len(visited_sectors),
        "pellets_eaten": pellets_eaten,
        "ghosts_eaten": ghosts_eaten,
        "power_pills": power_pills_collected,
        "exploration_ratio": len(visited_sectors) / 400.0  # 20x20 grid
    }
    
    return max(0.0, fitness), metrics

def calculate_fitness_baseline(env, net, max_steps=MAX_STEPS):
    """
    🎯 FITNESS BASELINE: Solo score del gioco
    
    Strategia semplice per stabilire una baseline solida:
    - Fitness = Score totale dell'episodio
    - Nessun reward shaping
    - Nessuna penalità/bonus artificiale
    
    Returns:
        fitness (float): Score totale
        metrics (dict): Metriche per logging
    """
    
    observation, info = env.reset()
    done = False
    steps = 0
    
    total_score = 0.0
    total_reward = 0.0
    
    while not done and steps < max_steps:
        # ATTIVAZIONE RETE NEAT
        output = net.activate(observation)
        action = np.argmax(output)
        
        # STEP AMBIENTE
        try:
            observation, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            print(f"⚠️ Errore durante step: {e}")
            break
        
        # ACCUMULA SCORE
        total_reward += reward
        total_score += reward  # In Pacman, reward = score incrementale
        
        steps += 1
        done = done or terminated or truncated
    
    # FITNESS = SCORE DEL GIOCO (semplice e pulito)
    fitness = total_score
    
    # Metriche per logging
    metrics = {
        "fitness": fitness,
        "game_score": total_score,
        "steps_survived": steps,
        "avg_reward_per_step": total_reward / max(steps, 1)
    }
    
    return fitness, metrics


def eval_single_genome(genome, config):
    """
    Valuta un singolo genoma NEAT.
    Chiamata dal ParallelEvaluator per ogni genoma nella popolazione.
    
    Args:
        genome: Genoma NEAT da valutare
        config: Configurazione NEAT
    
    Returns:
        float: Fitness del genoma
    """
    try:
        # --- CREAZIONE AMBIENTE ---
        env = OCAtari(ENV_ID, mode="ram", obs_mode="obj", render_mode="rgb_array", hud=False)
        
        # Disabilita sticky actions (repeat_action_probability)
        if hasattr(env.unwrapped, 'ale'):
            env.unwrapped.ale.setFloat('repeat_action_probability', 0.0)
        
        # Applica il wrapper di feature extraction
        env = PacmanFeatureWrapper(env, grid_rows=10, grid_cols=10)
        
        # --- CREAZIONE RETE NEAT ---
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # --- ESECUZIONE E VALUTAZIONE ---
        fitness, metrics = calculate_fitness_baseline(env, net, max_steps=MAX_STEPS)
        
        env.close()
        
        return fitness
        
    except Exception as e:
        print(f"❌ Errore in eval_single_genome: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def run_evolution():
    """
    Funzione principale per eseguire l'evoluzione NEAT.
    """
    
    print("=" * 80)
    print("🧬 NEAT EVOLUTION - PACMAN (ALE)")
    print("=" * 80)
    print(f"📁 Config: {CONFIG_PATH}")
    print(f"📊 Output: {OUTPUT_DIR}")
    print(f"⚙️  Workers: {NUM_WORKERS}")
    print(f"🔄 Generazioni: {NUM_GENERATIONS}")
    print(f"⏱  Max Steps: {MAX_STEPS}")
    print("=" * 80)
    
    # --- CARICAMENTO CONFIG NEAT ---
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"❌ Config file non trovato: {CONFIG_PATH}")
    
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )
    
    # --- PARALLEL EVALUATOR ---
    # Usa multiprocessing per valutare genomi in parallelo
    pe = neat.ParallelEvaluator(NUM_WORKERS, eval_single_genome)
    
    # --- INIZIALIZZAZIONE POPOLAZIONE ---
    population = neat.Population(config)
    
    # --- REPORTERS (Logging e Statistiche) ---
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    # Checkpointer: salva popolazione ogni 5 generazioni
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_prefix = os.path.join(OUTPUT_DIR, f"checkpoint_{timestamp}_")
    population.add_reporter(neat.Checkpointer(
        generation_interval=5,
        filename_prefix=checkpoint_prefix
        ))
    
    # --- EVOLUZIONE ---
    print("\n🚀 Avvio evoluzione...\n")

    try:
        winner = population.run(pe.evaluate, NUM_GENERATIONS)
        # --- SALVATAGGIO RISULTATI ---
        winner_path = os.path.join(OUTPUT_DIR, f"winner_{timestamp}.pkl")
        with open(winner_path, 'wb') as f:
            pickle.dump((winner, config), f)
        print(f"\n✅ Winner salvato: {winner_path}")
        # --- VISUALIZZAZIONE STATISTICHE ---
        print("\n📈 Generazione grafici...")
        try:
            plot_stats(
                stats,
                ylog=False,
                filename=os.path.join(OUTPUT_DIR, f"fitness_evolution_{timestamp}.png")
            )
            plot_species(
                stats,
                filename=os.path.join(OUTPUT_DIR, f"speciation_{timestamp}.png")
            )
            print("✅ Grafici salvati")
        except Exception as e:
            print(f"⚠️  Errore generazione grafici: {e}")

        # --- STATISTICHE FINALI ---
        print("\n" + "=" * 80)
        print("🏆 EVOLUZIONE COMPLETATA")
        print("=" * 80)
        print(f"Best Fitness: {winner.fitness:.2f}")
        print(f"Best Genome ID: {winner.key}")
        print(f"Generazioni: {len(stats.generation_statistics)}")
        print(f"Specie finali: {len(stats.get_species_sizes())}")
        print("=" * 80)

        return winner, config, stats
    
    except KeyboardInterrupt:
        print("\n\n⏸  Evoluzione interrotta dall'utente")
        return None, config, None
    
    except Exception as e:
        print(f"\n\n❌ Errore fatale durante evoluzione: {e}")
        import traceback
        traceback.print_exc()
        return None, config, None
    
if __name__ == "__main__":
    # Imposta metodo di avvio multiprocessing (necessario per alcuni OS)
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # Già impostato

    # Esegui evoluzione
    winner, config, stats = run_evolution()

    if winner is not None:
        print("\n✨ Usa il genoma winner per testare l'agente!")
        print(f"   Carica con: pickle.load(open('winner_*.pkl', 'rb'))")