# experiments_openevolve/meta_evaluator.py
import sys
import os
import importlib.util
import json
from functools import partial               
from core.evaluator import run_game_simulation  
from openevolve.evaluation_result import EvaluationResult

# --- Aggiungi il root del tuo progetto al PYTHONPATH ---
# Questo permette di importare 'core.problem', 'core.policy', ecc.
# Si aspetta che questo script sia in /experiments_openevolve
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)
# ---

try:
    from core.problem import GenericGymProblem
    from core.policy import LinearPolicy
except ImportError:
    print("Errore: Impossibile importare i moduli da /core.")
    print("Assicurati che il PYTHONPATH sia corretto e che /core abbia __init__.py")
    sys.exit(1)

# --- Configurazione del "Problema di Test" ---
# Usiamo un problema FACILE e VELOCE per il "Capo Allenatore". 

ENV_NAME = 'Freeway-v4'  
MAX_STEPS_PER_GAME = 1000 # Limite 

# Quante generazioni farà girare l'algoritmo dell'LLM per ogni valutazione
NUM_GENS_META_EVAL = 10 
POP_SIZE_META_EVAL = 20

# Prepariamo il problema UNA VOLTA SOLA per efficienza.
# Questo è il "problem" che passeremo all'algoritmo dell'LLM.
# Sostituisci il vecchio blocco "Prepariamo il problema..." con questo:
try:
    print(f"Pre-caricamento del problema di test: {ENV_NAME}...")

    # 1. Creiamo il problema come prima
    PROBLEM = GenericGymProblem(
        env_name=ENV_NAME, 
        policy_class=LinearPolicy,
        max_steps=MAX_STEPS_PER_GAME,
        obs_type="ram"
    )

    # 2. INIETTIAMO (MONKEY-PATCH) LA FUNZIONE "evaluate" MANCANTE
    #    Questa funzione fa ciò che fa "evaluator" ma per un solo genoma.
    def single_evaluate_func(genome):
        # Logica copiata da core/problem.py -> evaluator
        specific_agent_func = partial(PROBLEM.policy.decide_move, weights=genome)

        fitness, metrics = run_game_simulation(
            agent_decision_function=specific_agent_func,
            env_name=PROBLEM.env_name,
            max_steps=PROBLEM.max_steps,
            obs_type=PROBLEM.obs_type
        )
        return {"fitness": fitness} # Restituisce un dizionario come da specifiche

    # 3. "Attacchiamo" la funzione e la proprietà all'oggetto PROBLEM
    #    Così l'LLM e initial_program.py possono trovarle
    PROBLEM.evaluate = single_evaluate_func
    # (La proprietà 'dimensions' è già sull'oggetto, non serve
    #  iniettarla, ma il system_message e l'initial_program
    #  ora la chiameranno correttamente.)

    print("Problema di test caricato con successo e potenziato.")

except Exception as e:
    print(f"Errore critico durante l'inizializzazione di GenericGymProblem: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ---

def evaluate(program_path: str):
    """
    Funzione chiamata da OpenEvolve.
    Esegue l'algoritmo evolutivo ('initial_program_ea.py' evoluto)
    e ne misura la performance.
    """
    try:
        # 1. Importa dinamicamente la funzione 'evolve_neural_network'
        #    dal file .py generato dall'LLM
        spec = importlib.util.spec_from_file_location("evolved_module", program_path)
        evolved_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evolved_module)
        
        evolve_nn_func = evolved_module.evolve_neural_network
        
        # 2. Esegui l'algoritmo evoluto sul nostro problema di test (CartPole)
        #    Questo è il test del "Capo Allenatore"
        result_dict = evolve_nn_func(
            PROBLEM, 
            max_generations=NUM_GENS_META_EVAL, 
            pop_size=POP_SIZE_META_EVAL
        )
        
        final_fitness = result_dict.get("best_fitness_achieved", -float('inf'))
        
        # 3. Restituisce il punteggio a OpenEvolve
        #    Il "punteggio" è la fitness massima che l'algoritmo 
        #    è riuscito a trovare per CartPole.
        return EvaluationResult(
            metrics={"score": final_fitness, "correctness": 1.0}
        )
        
    except Exception as e:
        import traceback
        print(f"\n--- ERRORE NASCOSTO NEL META-EVALUATOR ---")
        traceback.print_exc() # Questo stamperà l'errore completo
        # L'LLM ha prodotto codice rotto (es. SyntaxError, TypeError)
        # Penalizzalo severamente e manda l'errore indietro.
        return EvaluationResult(
            metrics={"score": -1000.0, "correctness": 0.0},
            artifacts={"stderr": str(e)} # Feedback per l'LLM
        )

# Blocco 'main' per il testing e per OpenEvolve
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Chiamato da OpenEvolve
        program_path = sys.argv[1]
        result = evaluate_program(program_path)
        print(json.dumps(result.to_dict()))
    else:
        # Test manuale (esegui questo file direttamente per debug)
        print("Esecuzione test manuale del meta-evaluator...")
        test_program_path = os.path.join(script_dir, 'initial_program_ea.py')
        result = evaluate_program(test_program_path)
        print(f"Test completato. Risultato da 'initial_program_ea.py':")
        print(json.dumps(result.to_dict(), indent=2))