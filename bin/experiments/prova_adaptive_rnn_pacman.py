# FILE: experiments/run_pacman_neat_optimized.py

import sys
import os
import neat
import numpy as np
import pickle
import datetime
import multiprocessing

# --- SOLUZIONE PER L'IMPORT ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

from core.evaluator import run_game_simulation
from algorithms.neat_runner import run_neat
from utils.neat_plotting_utils import plot_stats, plot_species, draw_net

import gymnasium as gym

# =============================================================================
# 0. PARAMETRI DELL'ESPERIMENTO
# =============================================================================

ENV_NAME = "ALE/Pacman-v5"
CONFIG_FILE_NAME = "temp_rnn_neat_pacman_config_adaptive.txt"
NUM_GENERATIONS = 100
MAX_STEPS = 10000

# Abilita/disabilita curriculum learning
CURRICULUM_ENABLED = True

# Scegli strategia fitness: "adaptive", "score_first", "longevity_first", "exploration_guided"
FITNESS_STRATEGY = "adaptive"

root_dir = project_root
OUTPUT_DIR = os.path.join(root_dir, "evolution_results")
CONFIG_PATH = os.path.join(root_dir, "configs", CONFIG_FILE_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True) 

current_net = None
generation_counter = 0

# =============================================================================
# 1. CONFIGURAZIONI AMBIENTE (CURRICULUM LEARNING)
# =============================================================================

ENV_CONFIG_EASY = {
    'frameskip': 4,
    'repeat_action_probability': 0.0,
}

ENV_CONFIG_MEDIUM = {
    'frameskip': 3,
    'repeat_action_probability': 0.1,
}

ENV_CONFIG_HARD = {
    'frameskip': 2,
    'repeat_action_probability': 0.25,
}

def get_env_config(generation):
    """Curriculum learning: aumenta difficoltà progressivamente"""
    if not CURRICULUM_ENABLED:
        return {'frameskip': 2, 'repeat_action_probability': 0.0}
    
    if generation < 30:
        return ENV_CONFIG_EASY
    elif generation < 60:
        return ENV_CONFIG_MEDIUM
    else:
        return ENV_CONFIG_HARD

def get_max_steps(generation):
    """Aumenta progressivamente il tempo di gioco"""
    if not CURRICULUM_ENABLED:
        return MAX_STEPS
    
    if generation < 20:
        return 5000
    elif generation < 50:
        return 8000
    else:
        return 12000

# =============================================================================
# 2. STRATEGIE FITNESS
# =============================================================================

# Strategia A: Score-First
COEFFS_SCORE_FIRST = {
    'SURVIVAL_BONUS': 0.005,
    'EXPLORATION_BONUS': 0.01,
    'STUCK_PENALTY': 0.05,
    'EARLY_DEATH_PENALTY': 0.3,
    'MAX_STUCK_STEPS': 200
}

# Strategia B: Exploration-Guided
COEFFS_EXPLORATION = {
    'SURVIVAL_BONUS': 0.02,
    'EXPLORATION_BONUS': 0.1,
    'STUCK_PENALTY': 0.2,
    'EARLY_DEATH_PENALTY': 0.4,
    'MAX_STUCK_STEPS': 100
}

# Strategia C: Longevity-First
COEFFS_LONGEVITY = {
    'SURVIVAL_BONUS': 0.05,
    'EXPLORATION_BONUS': 0.02,
    'STUCK_PENALTY': 0.1,
    'EARLY_DEATH_PENALTY': 0.6,
    'MAX_STUCK_STEPS': 150
}

def get_fitness_coeffs(strategy, generation=0):
    """Restituisce coefficienti in base alla strategia"""
    if strategy == "score_first":
        return COEFFS_SCORE_FIRST
    elif strategy == "exploration_guided":
        return COEFFS_EXPLORATION
    elif strategy == "longevity_first":
        return COEFFS_LONGEVITY
    elif strategy == "adaptive":
        # Coefficienti che cambiano nel tempo
        progress = min(generation / 50.0, 1.0)
        return {
            'SURVIVAL_BONUS': 0.05 * (1.0 - progress),
            'EXPLORATION_BONUS': 0.1 * (1.0 - progress),
            'STUCK_PENALTY': 0.2 * (1.0 + progress),
            'EARLY_DEATH_PENALTY': 0.3 + (0.3 * progress),
            'MAX_STUCK_STEPS': int(200 - generation * 1.5)
        }
    else:
        return COEFFS_SCORE_FIRST

# =============================================================================
# 3. FUNZIONE FITNESS
# =============================================================================


def calculate_fitness_with_shaping(agent_decision_function, env_name, max_steps, 
                                   generation=0, strategy="adaptive"):
    """
    Fitness avanzata con reward shaping e coefficienti configurabili.
    """
    coeffs = get_fitness_coeffs(strategy, generation)
    env_config = get_env_config(generation)
    
    env = gym.make(
        env_name, 
        obs_type="ram", 
        frameskip=env_config['frameskip'],
        repeat_action_probability=env_config['repeat_action_probability']
    )
    
    observation, info = env.reset()
    done = False
    steps = 0
    
    fitness = 0.0
    metrics = {
        "score_grezzo": 0.0,
        "bonus_survival": 0.0,
        "bonus_esplorazione": 0.0,
        "penalita_stuck": 0.0,
        "passi_sopravvissuti": 0,
        "stati_ram_unici": set()
    }
    
    stuck_counter = 0
    prev_ram_state = tuple(observation)
    
    while not done and steps < max_steps:
        features = observation / 255.0
        action = agent_decision_function(features)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        ram_hash = tuple(observation)
        
        # 1. Score grezzo (componente dominante)
        fitness += reward
        metrics["score_grezzo"] += reward
        
        # 2. Bonus sopravvivenza
        bonus_surv = coeffs['SURVIVAL_BONUS']
        fitness += bonus_surv
        metrics["bonus_survival"] += bonus_surv
        
        # 3. Bonus esplorazione
        if ram_hash not in metrics["stati_ram_unici"]:
            metrics["stati_ram_unici"].add(ram_hash)
            bonus_expl = coeffs['EXPLORATION_BONUS']
            fitness += bonus_expl
            metrics["bonus_esplorazione"] += bonus_expl
            stuck_counter = 0
        
        # 4. Anti-stuck
        if ram_hash == prev_ram_state:
            stuck_counter += 1
        else:
            stuck_counter = 0
            
        if stuck_counter > coeffs['MAX_STUCK_STEPS']:
            penalty = coeffs['STUCK_PENALTY']
            fitness -= penalty
            metrics["penalita_stuck"] -= penalty
        
        prev_ram_state = ram_hash
        metrics["passi_sopravvissuti"] = steps

    env.close()
    
    # Penalità morte precoce
    if steps < 300:
        fitness *= (1.0 - coeffs['EARLY_DEATH_PENALTY'])
    
    # Converti set in int per logging
    metrics["stati_ram_unici"] = len(metrics["stati_ram_unici"])
    
    return max(0.1, fitness), metrics

# =============================================================================
# 4. FUNZIONI AGENTE
# =============================================================================

def agent_decision_function_neat(game_state):
    """Funzione agente che usa il network NEAT."""
    if current_net is None:
        return 0

    features = game_state / 255.0
    output = current_net.activate(features)
    action_idx = np.argmax(output)
    ACTIONS = [0, 2, 3, 4, 5]
    return ACTIONS[action_idx]

def eval_single_genome(genome, config):
    """Valutazione singolo genoma per parallelizzazione."""
    try:
        current_net = neat.nn.RecurrentNetwork.create(genome, config)
        
        def agent_decision_local(game_state):
            features = game_state / 255.0
            output = current_net.activate(features)
            action_idx = np.argmax(output)
            ACTIONS = [0, 2, 3, 4, 5]
            return ACTIONS[action_idx]
            
        fitness, metrics = calculate_fitness_with_shaping(
            agent_decision_local,
            ENV_NAME,
            get_max_steps(generation_counter),
            generation=generation_counter,
            strategy=FITNESS_STRATEGY
        )
        
        return fitness
        
    except Exception as e:
        return 0.1

# =============================================================================
# 5. VALUTAZIONE GENERAZIONE
# =============================================================================
def is_valid_genome(genome, config):
    """Verifica se un genoma ha una topologia valida."""
    try:
        # Controlla che ci siano connessioni
        if not genome.connections:
            return False
        
        # Controlla che ci siano output raggiungibili
        enabled_connections = [c for c in genome.connections.values() if c.enabled]
        if not enabled_connections:
            return False
            
        return True
    except:
        return False


def eval_genomes(genomes, config):
    """Valuta ogni genoma nella popolazione."""
    global current_net, generation_counter
    generation_counter += 1
    
    print(f"\n{'='*70}")
    print(f"🧬 GENERAZIONE {generation_counter}/{NUM_GENERATIONS}")
    print(f"{'='*70}")
    
    # Info curriculum
    if CURRICULUM_ENABLED:
        env_config = get_env_config(generation_counter)
        max_steps_current = get_max_steps(generation_counter)
        coeffs = get_fitness_coeffs(FITNESS_STRATEGY, generation_counter)
        
        print(f"🎮 Ambiente:")
        print(f"   • Frameskip: {env_config['frameskip']}")
        print(f"   • Stochastic: {env_config['repeat_action_probability']:.2f}")
        print(f"   • Max steps: {max_steps_current}")
        print(f"\n🎯 Fitness Strategy: {FITNESS_STRATEGY}")
        print(f"   • Survival bonus: {coeffs['SURVIVAL_BONUS']:.4f}")
        print(f"   • Exploration bonus: {coeffs['EXPLORATION_BONUS']:.4f}")
        print(f"   • Stuck penalty: {coeffs['STUCK_PENALTY']:.4f}")
    
    print(f"\n{'='*70}")
    
    fitness_scores = []
    score_history = []
    best_genome_this_gen = None
    best_fitness_this_gen = 0
    best_score_this_gen = 0
    
    for i, (genome_id, genome) in enumerate(genomes):
        try:
            # 🔥 VALIDAZIONE PREVENTIVA
            if not is_valid_genome(genome, config):
                print(f"   ⚠️  Genoma {genome_id}: topologia invalida")
                genome.fitness = 0.1
                fitness_scores.append(0.1)
                score_history.append(0)
                continue
            
            try:
                current_net = neat.nn.RecurrentNetwork.create(genome, config)
            except Exception as net_error:
                print(f"   ⚠️  Genoma {genome_id}: creazione rete fallita")
                genome.fitness = 0.1
                fitness_scores.append(0.1)
                score_history.append(0)
                continue
            
            fitness, metrics = calculate_fitness_with_shaping(
                agent_decision_function_neat,
                ENV_NAME,
                get_max_steps(generation_counter),
                generation=generation_counter,
                strategy=FITNESS_STRATEGY
            )
            
            genome.fitness = fitness
            fitness_scores.append(fitness)
            score_history.append(metrics['score_grezzo'])
            
            # Traccia il migliore
            if fitness > best_fitness_this_gen:
                best_fitness_this_gen = fitness
                best_genome_this_gen = genome_id
                best_score_this_gen = metrics['score_grezzo']
            
            # Stampa progressi ogni 20 genomi
            if (i + 1) % 20 == 0:
                print(f"   [{i+1:3d}/{len(genomes)}] "
                      f"fitness={fitness:.2f}, score={metrics['score_grezzo']:.0f}, "
                      f"steps={metrics['passi_sopravvissuti']}")
                
        except Exception as e:
            print(f"   ❌ Errore genoma {genome_id}: {e}")
            genome.fitness = 0.1
            fitness_scores.append(0.1)
            score_history.append(0)
    
    # Statistiche finali generazione
    print(f"\n{'='*70}")
    print(f"📊 STATISTICHE GENERAZIONE {generation_counter}")
    print(f"{'='*70}")
    print(f"Fitness:")
    print(f"   • Max:  {max(fitness_scores):.2f} (Genoma #{best_genome_this_gen})")
    print(f"   • Mean: {np.mean(fitness_scores):.2f}")
    print(f"   • Min:  {min(fitness_scores):.2f}")
    print(f"\nScore Grezzo:")
    print(f"   • Max:  {max(score_history):.0f}")
    print(f"   • Mean: {np.mean(score_history):.1f}")
    print(f"   • Min:  {min(score_history):.0f}")
    print(f"{'='*70}")

# =============================================================================
# 6. ESECUZIONE PRINCIPALE
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧬 AVVIO EVOLUZIONE NEAT PER MS. PAC-MAN (OPTIMIZED)")
    print("=" * 80)
    print(f"📂 Config: {CONFIG_PATH}")
    print(f"📈 Output: {OUTPUT_DIR}")
    print(f"🎮 Environment: {ENV_NAME}")
    print(f"⏱️  Max Steps: {MAX_STEPS}")
    print(f"🔢 Generazioni: {NUM_GENERATIONS}")
    print(f"👥 Workers: {NUM_WORKERS}")
    print(f"🎯 Fitness Strategy: {FITNESS_STRATEGY}")
    print(f"📚 Curriculum Learning: {'✅ ENABLED' if CURRICULUM_ENABLED else '❌ DISABLED'}")
    print("=" * 80)
    
    if not os.path.exists(CONFIG_PATH):
        print(f"\n❌ ERRORE: File di configurazione non trovato: {CONFIG_PATH}")
        sys.exit(1)
    
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        print("\n🔧 Caricamento configurazione NEAT...")
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            CONFIG_PATH
        )
        print("✅ Configurazione caricata con successo")
        print(f"   • Input: {config.genome_config.num_inputs}")
        print(f"   • Output: {config.genome_config.num_outputs}")
        print(f"   • Popolazione: {config.pop_size}")
        print(f"   • Hidden iniziali: {config.genome_config.num_hidden}")
        
        # Inizializza popolazione
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        
        # Checkpointer
        checkpoint_prefix = os.path.join(OUTPUT_DIR, f"checkpoint_{timestamp_str}_")
        p.add_reporter(neat.Checkpointer(
            generation_interval=10,
            filename_prefix=checkpoint_prefix
        ))

        print("\n🚀 Avvio evoluzione...")
        print(f"⚙️  Modalità: Sequential (per compatibilità con curriculum)")
        
        # Usa eval_genomes sequenziale per avere controllo su generation_counter
        winner = p.run(eval_genomes, NUM_GENERATIONS)
        
        # Salvataggio risultati
        print("\n" + "=" * 80)
        print("💾 SALVATAGGIO RISULTATI")
        print("=" * 80)
        
        winner_file = os.path.join(OUTPUT_DIR, f"best_genome_neat_{timestamp_str}.pkl")
        with open(winner_file, 'wb') as f:
            pickle.dump(winner, f)
        print(f"✅ Genoma vincitore salvato: {winner_file}")
        print(f"   • Fitness: {winner.fitness:.2f}")
        
        # Salva config usata
        config_backup = os.path.join(OUTPUT_DIR, f"config_used_{timestamp_str}.txt")
        import shutil
        shutil.copy(CONFIG_PATH, config_backup)
        print(f"✅ Config salvata: {config_backup}")
        
        # Salva parametri esperimento
        params_file = os.path.join(OUTPUT_DIR, f"experiment_params_{timestamp_str}.txt")
        with open(params_file, 'w') as f:
            f.write(f"Fitness Strategy: {FITNESS_STRATEGY}\n")
            f.write(f"Curriculum Enabled: {CURRICULUM_ENABLED}\n")
            f.write(f"Num Generations: {NUM_GENERATIONS}\n")
            f.write(f"Population: {config.pop_size}\n")
            f.write(f"Workers: {NUM_WORKERS}\n")
        print(f"✅ Parametri salvati: {params_file}")
        
        # Grafici
        print("\n📊 Generazione grafici...")
        plot_stats_file = os.path.join(OUTPUT_DIR, f"neat_fitness_{timestamp_str}.png")
        plot_species_file = os.path.join(OUTPUT_DIR, f"neat_speciation_{timestamp_str}.png")
        
        try:
            plot_stats(stats, ylog=False, filename=plot_stats_file)
            print(f"   ✅ Grafico fitness: {plot_stats_file}")
        except Exception as e:
            print(f"   ❌ Errore grafico fitness: {e}")
        
        try:
            plot_species(stats, filename=plot_species_file)
            print(f"   ✅ Grafico speciazione: {plot_species_file}")
        except Exception as e:
            print(f"   ❌ Errore grafico speciazione: {e}")
        
        print("\n" + "=" * 80)
        print("✅ EVOLUZIONE COMPLETATA CON SUCCESSO")
        print("=" * 80)
        print(f"\n🏆 Best Fitness: {winner.fitness:.2f}")
        print(f"📁 Risultati salvati in: {OUTPUT_DIR}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Evoluzione interrotta dall'utente")
        
    except Exception as e:
        print(f"\n{'=' * 80}")
        print("❌ ERRORE DURANTE L'EVOLUZIONE")
        print(f"{'=' * 80}")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)