# ======================================================================
# file: run_ga_bankheist.py
# ======================================================================

from pylab import *
from random import Random
import sys
import traceback
import json
from datetime import datetime
import os

# Importa il 'run_ga'
from lab_ga_runner import run_ga

# Importa il NOSTRO problema
from bank_heist_problem import BankHeistProblem
# Importa la dimensione corretta
from agent_policy import TOTAL_WEIGHTS 

print("--- 🧬 Avvio Evoluzione per BankHeist (RAM Mode) ---")

# --- 0. Cartella di output ---
OUTPUT_DIR = "ga_results"
os.makedirs(OUTPUT_DIR, exist_ok=True) # <-- 2. CREA LA CARTELLA

# --- 1. Parametri per il GA ---
args = {}

# Quanti geni ha il nostro cromosoma?
# 18 azioni * 128 features RAM +18 bias = 2322
args["num_vars"] = TOTAL_WEIGHTS

# Parametri del GA
args["pop_size"] = 20        # <-- Aumentato (20 è troppo poco)
args["max_generations"] = 10 # <-- Aumentato (10 è troppo poco)

# --- NOTA BENE ---
# Un problema con 2304 dimensioni è ENORME.
# Questi parametri (50 pop, 25 gen) sono ANCORA TROPPO PICCOLI
# per trovare una soluzione, ma servono per un test rapido (durerà qualche minuto).
# Per un'evoluzione REALE servirebbero pop_size=500, max_generations=1000+

args["gaussian_stdev"] = 0.3 
args["crossover_rate"] = 0.8 
args["tournament_size"] = 2 
args["mutation_rate"] = 0.3 
args["num_elites"] = 1 

args["use_bounder"] = True
display = True

args["fig_title"] = 'GA - BankHeist (RAM)'

# --- 2. Esecuzione ---
seed = None
rng = Random(seed)

try:
    best_individual, best_fitness, final_pop = run_ga(rng, 
                                                      display=display,
                                                      problem_class=BankHeistProblem,
                                                      **args)
    
    print("\n--- 🏆 Evoluzione Terminata ---")
    print(f"Miglior Fitness Trovata: {best_fitness:.2f}")
    print(f"Miglior Individuo (primi 5 pesi): {best_individual[:5]}...")
    
    # Dopo l'evoluzione
    results = {
        "timestamp": datetime.now().isoformat(),
        "best_fitness": float(best_fitness),
        "parameters": {k: v for k, v in args.items() if not k.startswith('plot')},
        "best_weights": best_individual.tolist()
    }

    with open(f"best_solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n💾 Soluzione salvata!")
    if display:
        print("\nSalvataggio grafico fitness...")
        plot_filename = os.path.join(OUTPUT_DIR, f"fitness_trend_{timestamp_str}.png")
        
        try:
            # 'savefig' è importato da 'from pylab import *'
            savefig(plot_filename) 
            print(f"📈 Grafico salvato in: {plot_filename}")

            # Chiudi la figura per liberare memoria
            fig_name = args["fig_title"] + ' (fitness trend)'
            close(fig_name) 
        except Exception as e:
            print(f"Attenzione: non è stato possibile salvare il grafico. Errore: {e}")
            
except Exception as e:
    print(f"\n❌ Si è verificato un errore durante l'evoluzione: {e}")
    traceback.print_exc()