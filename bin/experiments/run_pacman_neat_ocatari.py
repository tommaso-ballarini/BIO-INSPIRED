# experiments/run_pacman_neat_ocatari.py
import sys
import os
import neat
import numpy as np
import pickle
import datetime
import multiprocessing
from pathlib import Path

# --- SETUP PATH ---
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import gymnasium as gym
from ocatari.core import OCAtari
from core.wrappers_pacman import NeatPacmanWrapper
from utils.neat_plotting_utils import plot_stats, plot_species

# --- PARAMETRI ---
ENV_ID = "Pacman"  # OCAtari usa "Pacman" come ID
CONFIG_FILE = "neat_pacman_ocatari_config.txt"
NUM_GENERATIONS = 100
MAX_STEPS = 3000
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)

OUTPUT_DIR = project_root / "evolution_results" / "pacman_ocatari"
CONFIG_PATH = project_root / "configs" / CONFIG_FILE
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def eval_single_genome(genome, config):
    """Valuta un singolo genoma"""
    try:
        # Crea pipeline con OCAtari
        # Nota: OCAtari usa direttamente "Pacman" come env_name
        ocatari_env = OCAtari(ENV_ID, mode="ram", obs_mode="obj", render_mode="rgb_array")
        
        if hasattr(ocatari_env.unwrapped, 'ale'):
            ocatari_env.unwrapped.ale.setFloat('repeat_action_probability', 0.0)
        
        env = NeatPacmanWrapper(ocatari_env)
        
        # Crea rete
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Valutazione
        observation, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done and steps < MAX_STEPS:
            output = net.activate(observation)
            action = np.argmax(output)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        env.close()
        
        # Fitness: reward + bonus sopravvivenza
        fitness = total_reward + (steps * 0.01)
        return max(0.0, fitness)
        
    except Exception as e:
        print(f"Errore valutazione: {e}")
        return 0.0


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass

    print("=" * 80)
    print("🧬 EVOLUZIONE NEAT - PACMAN (OCAtari REM)")
    print("=" * 80)
    print(f"🎮 Ambiente: {ENV_ID}")
    print(f"📊 Popolazione: 150")
    print(f"🔄 Generazioni: {NUM_GENERATIONS}")
    print(f"⚡ Workers: {NUM_WORKERS}")
    print(f"📁 Output: {OUTPUT_DIR}")
    print("=" * 80)
    
    if not CONFIG_PATH.exists():
        print(f"❌ Config non trovato: {CONFIG_PATH}")
        sys.exit(1)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(CONFIG_PATH)
        )
        
        pe = neat.ParallelEvaluator(NUM_WORKERS, eval_single_genome)
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(
            10, 
            filename_prefix=str(OUTPUT_DIR / f"checkpoint_{timestamp}_")
        ))
        
        print(f"\n🚀 Avvio evoluzione...\n")
        winner = p.run(pe.evaluate, NUM_GENERATIONS)
        
        print("\n" + "=" * 80)
        print("✅ EVOLUZIONE COMPLETATA")
        print("=" * 80)
        
        # Salva risultati
        winner_file = OUTPUT_DIR / f"winner_{timestamp}.pkl"
        with open(winner_file, 'wb') as f:
            pickle.dump(winner, f)
        print(f"💾 Winner salvato: {winner_file}")
        print(f"📊 Fitness: {winner.fitness:.2f}")
        
        try:
            plot_stats(stats, ylog=False, 
                      filename=str(OUTPUT_DIR / f"fitness_{timestamp}.png"))
            plot_species(stats, 
                        filename=str(OUTPUT_DIR / f"species_{timestamp}.png"))
            print(f"📈 Grafici salvati in {OUTPUT_DIR}")
        except Exception as e:
            print(f"⚠️ Errore grafici: {e}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrotto")
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()