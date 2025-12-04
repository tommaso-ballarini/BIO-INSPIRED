import sys
import os
import pathlib
import pickle

# Setup percorsi
project_root = pathlib.Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from core.gp_engine import run_gp_evolution
from core.gp_evaluator import run_gp_simulation, REWARD_FACTOR, COLLISION_PENALTY

# experiments/run_gp_freeway.py

def save_best_agent_as_code(hof_individual, pset, filename="best_gp_agent.py"):
    """Converte l'albero vincente in uno script Python usabile da view_agent.py"""
    tree_code = str(hof_individual)
    
    # Template che rende l'albero eseguibile come script standalone
    # AGGIUNTA IMPORTANTE: Mappiamo le funzioni DEAP agli operatori Python
    code_content = f"""
import math
import operator

# --- Primitive del Genetic Programming ---
def add(a, b): return a + b
def sub(a, b): return a - b
def mul(a, b): return a * b
def neg(a): return -a
def lt(a, b): return 1.0 if a < b else 0.0
def gt(a, b): return 1.0 if a > b else 0.0

def protected_div(left, right):
    if abs(right) < 1e-6: return 1
    return left / right

def if_then_else(input, output1, output2):
    return output1 if input else output2

# --- Agente Evoluto ---
def get_action(observation):
    # Mapping variabili (Input Terminali)
    chicken_y = observation[0]
    car_1 = observation[1]
    car_2 = observation[2]
    car_3 = observation[3]
    car_4 = observation[4]
    car_5 = observation[5]
    car_6 = observation[6]
    car_7 = observation[7]
    car_8 = observation[8]
    car_9 = observation[9]
    car_10 = observation[10]
    
    # Costanti effimere (se presenti nell'albero)
    # Il GP potrebbe aver generato costanti come numeri diretti, 
    # ma se usa 'rand101' dobbiamo gestirlo o sperare che DEAP lo abbia convertito in numero.
    # Solitamente DEAP converte le costanti in numeri nella stringa, quindi ok.

    # Logica dell'Albero
    output = {tree_code}
    
    # Policy di Attuazione
    return 1 if output > 0 else 0
"""
    
    output_path = project_root / 'evolution_history' / filename
    with open(output_path, "w") as f:
        f.write(code_content)
    print(f"Agente salvato in: {output_path}")

if __name__ == "__main__":
    
    # Esecuzione
    # Consigliato: Aumenta drasticamente
    pop, log, hof, pset = run_gp_evolution(
        n_gen=50, 
        pop_size=300,   # Raddoppia la popolazione
        cx_prob=0.5, 
        initial_mut_prob=0.6   
    )
    
    print("\n--- Evoluzione Terminata ---")
    best_ind = hof[0]
    print(f"Miglior Fitness: {best_ind.fitness.values[0]}")
    print(f"Codice Albero: {best_ind}")
    
    history_dir = project_root / 'evolution_history'
    os.makedirs(history_dir, exist_ok=True)

    logbook_path = history_dir / 'gp_logbook.pkl'
    with open(logbook_path, "wb") as f:
        pickle.dump(log, f)
    print(f"Logbook salvato per visualizzazione in: {logbook_path}")

    # Salvataggio
    save_best_agent_as_code(best_ind, pset)
    
    # Test Visivo Finale
    print("\nEsecuzione Test Visivo del Migliore...")
    import time
    # Compiliamo ed eseguiamo al volo
    from deap import gp
    func = gp.compile(best_ind, pset)
    run_gp_simulation(func, render=True)