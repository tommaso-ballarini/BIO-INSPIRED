# FILE: experiments/run_pacman_neat.py

import sys
import os
import neat
import numpy as np
import pickle
import datetime

# --- SOLUZIONE PER L'IMPORT ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- FINE SOLUZIONE ---

from core.evaluator import run_game_simulation
from algorithms.neat_runner import run_neat
from utils.neat_plotting_utils import plot_stats, plot_species, draw_net

# --- 0. Parametri dell'Esperimento ---
ENV_NAME = "ALE/MsPacman-v5"
CONFIG_FILE_NAME = "neat_pacman_config.txt"
NUM_GENERATIONS = 5
MAX_STEPS = 3000

root_dir = project_root
OUTPUT_DIR = os.path.join(root_dir, "evolution_results")
CONFIG_PATH = os.path.join(root_dir, "configs", CONFIG_FILE_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True) 

current_net = None
generation_counter = 0  # Aggiungi contatore

def agent_decision_function_neat(game_state):
    """
    Funzione agente che usa il network NEAT.
    """
    if current_net is None:
        return 0

    # Normalizza l'input
    features = game_state / 255.0
    
    # Attiva il network
    output = current_net.activate(features)
    
    # Restituisci l'azione migliore
    return np.argmax(output)


def eval_genomes(genomes, config):
    """
    La funzione fitness richiesta da NEAT.
    Valuta ogni genoma nella popolazione.
    """
    global current_net, generation_counter
    
    generation_counter += 1
    print(f"\n{'='*60}")
    print(f"🔄 GENERAZIONE {generation_counter}/{NUM_GENERATIONS}")
    print(f"{'='*60}")
    
    fitness_scores = []
    
    # Loop su ogni genoma
    for i, (genome_id, genome) in enumerate(genomes):
        try:
            # 1. Crea il network
            current_net = neat.nn.FeedForwardNetwork.create(genome, config)
            
            # 2. Esegui la simulazione
            fitness, metrics = run_game_simulation(
                agent_decision_function=agent_decision_function_neat,
                env_name=ENV_NAME,
                max_steps=MAX_STEPS,
                obs_type="ram"
            )
            
            # 3. Assegna la fitness
            genome.fitness = fitness
            fitness_scores.append(fitness)
            
            # Print ogni 10 genomi per non intasare
            if (i + 1) % 10 == 0:
                print(f"   Genoma {i+1}/{len(genomes)}: fitness={fitness:.2f}")
                
        except Exception as e:
            print(f"❌ Errore nel genoma {genome_id}: {e}")
            genome.fitness = 0.0
            fitness_scores.append(0.0)
    
    # Statistiche della generazione
    if fitness_scores:
        print(f"\n📊 Statistiche Generazione {generation_counter}:")
        print(f"   • Max:  {max(fitness_scores):.2f}")
        print(f"   • Mean: {np.mean(fitness_scores):.2f}")
        print(f"   • Min:  {min(fitness_scores):.2f}")


# --- 3. Esecuzione ---
if __name__ == "__main__":
    print("=" * 70)
    print("🧬 AVVIO EVOLUZIONE NEAT PER MS. PAC-MAN")
    print("=" * 70)
    print(f"📂 Config: {CONFIG_PATH}")
    print(f"📈 Output: {OUTPUT_DIR}")
    print(f"🎮 Environment: {ENV_NAME}")
    print(f"⏱️  Max Steps: {MAX_STEPS}")
    print(f"🔢 Generazioni: {NUM_GENERATIONS}")
    print(f"👥 Popolazione: 100")
    print("=" * 70)
    
    # Verifica che il file di config esista
    if not os.path.exists(CONFIG_PATH):
        print(f"\n❌ ERRORE: File di configurazione non trovato: {CONFIG_PATH}")
        sys.exit(1)
    
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        # Carica la config per verificare
        print("\n🔍 Caricamento configurazione NEAT...")
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
        
        winner, config, stats = run_neat(
            eval_function=eval_genomes,
            config_file_path=CONFIG_PATH,
            num_generations=NUM_GENERATIONS,
            output_dir=OUTPUT_DIR,
            timestamp_str=timestamp_str
        )
        
        # --- 4. Salvataggio Risultati ---
        print("\n" + "=" * 70)
        print("💾 SALVATAGGIO RISULTATI")
        print("=" * 70)
        
        # Salva il genoma vincitore
        winner_file = os.path.join(OUTPUT_DIR, f"best_genome_neat_{timestamp_str}.pkl")
        with open(winner_file, 'wb') as f:
            pickle.dump(winner, f)
        print(f"✅ Genoma vincitore salvato: {winner_file}")
        print(f"   • Fitness: {winner.fitness:.2f}")
        
        # Verifica stats
        if stats is None:
            print("❌ ATTENZIONE: L'oggetto stats è None!")
        else:
            print(f"✅ Stats raccolte: {len(stats.generation_statistics)} generazioni")
        
        # Salva i grafici
        plot_stats_file = os.path.join(OUTPUT_DIR, f"neat_fitness_{timestamp_str}.png")
        plot_species_file = os.path.join(OUTPUT_DIR, f"neat_speciation_{timestamp_str}.png")
        
        print("\n📈 Generazione grafici...")
        try:
            plot_stats(stats, ylog=False, filename=plot_stats_file)
            print(f"   ✅ Grafico fitness: {plot_stats_file}")
        except Exception as e:
            print(f"   ❌ Errore nel grafico fitness: {e}")
        
        try:
            plot_species(stats, filename=plot_species_file)
            print(f"   ✅ Grafico speciazione: {plot_species_file}")
        except Exception as e:
            print(f"   ❌ Errore nel grafico speciazione: {e}")
        
        print("\n" + "=" * 70)
        print("✅ EVOLUZIONE COMPLETATA CON SUCCESSO")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n⚠️  Evoluzione interrotta dall'utente")
        
    except Exception as e:
        print(f"\n{'=' * 70}")
        print("❌ ERRORE DURANTE L'EVOLUZIONE")
        print(f"{'=' * 70}")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)