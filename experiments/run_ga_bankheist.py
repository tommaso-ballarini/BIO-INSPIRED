# FILE: experiments/run_bankheist_ga.py

import sys
import os
# 1. Trova il percorso della cartella 'experiments' (dove si trova questo file)
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Trova la cartella root del progetto (un livello sopra, 'BIO-INSPIRED')
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# 3. Aggiungi la root al percorso di ricerca di Python
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from random import Random

import json
from pylab import *
from datetime import datetime

# 1. Importa i componenti giusti
from core.problem import GenericGymProblem
from core.policy import LinearPolicy
from algorithms.inspyred_runner import run_ga 


# --- 0. Cartella di output ---
# (Questa logica va bene, ma assicurati che i percorsi siano corretti)
root_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(root_dir, "..", "evolution_results") # Vai su e poi giù
os.makedirs(OUTPUT_DIR, exist_ok=True) 

print("--- 🧬 Avvio Evoluzione per BankHeist (RAM Mode) ---")
print(f"--- 📂 Output sarà salvato in: {OUTPUT_DIR} ---")

# --- 1. Parametri per il GA ---
ga_args = {}
ga_args["pop_size"] = 20
ga_args["max_generations"] = 10
ga_args["gaussian_stdev"] = 0.3
ga_args["crossover_rate"] = 0.8
ga_args["tournament_size"] = 2
ga_args["mutation_rate"] = 0.3
ga_args["num_elites"] = 1
ga_args["use_bounder"] = True
ga_args["fig_title"] = 'GA - BankHeist (RAM)'

# --- 2. Parametri per il PROBLEMA ---
problem_args = {
    "env_name": "ALE/BankHeist-v5",
    "policy_class": LinearPolicy,
    "obs_type": "ram",           # Obbligatorio per BankHeist
    "max_steps": 1500
}

# Inizializza il problema (ora non servono più num_vars!)
problem = GenericGymProblem(**problem_args)

# Passa num_vars al GA, preso dal problema
ga_args["num_vars"] = problem.dimensions 

# --- 3. Esecuzione ---
seed = None
rng = Random(seed)
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

try:
    best_individual, best_fitness, final_pop = run_ga(
        rng, 
        problem=problem, # Passa l'istanza del problema
        display=True,
        **ga_args
    )
    
    print("\n--- 🏆 Evoluzione Terminata ---")
    # (Il resto del tuo codice per salvare JSON e PNG va bene)
    # ... (codice di salvataggio omesso per brevità) ...
    
except Exception as e:
    print(f"\n❌ Si è verificato un errore: {e}")
    import traceback
    traceback.print_exc()