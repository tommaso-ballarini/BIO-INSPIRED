import sys
import os
import importlib.util
import json
from functools import partial

# --- Aggiungi il root del progetto al PYTHONPATH ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# ---

try:
    # Assumiamo che la tua funzione 'run_game_simulation' esista ancora
    from core.evaluator import run_game_simulation
    from openevolve.evaluation_result import EvaluationResult
except ImportError:
    print("Errore: Impossibile importare i moduli da /core o openevolve.")
    sys.exit(1)

# --- Configurazione del Test ---
ENV_NAME = 'Freeway-v4'
MAX_STEPS_PER_GAME = 1500  # Diamo un po' più di tempo per partita
NUM_GAMES_PER_EVAL = 3     # Eseguiamo 3 partite per avere un punteggio medio

def evaluate(program_path: str):
    """
    Funzione chiamata da OpenEvolve per valutare UN agente.
    Questo processo è VELOCE.
    """
    try:
        # 1. Importa dinamicamente la funzione 'get_action'
        #    dal file .py generato dall'LLM (es. initial_agent.py)
        spec = importlib.util.spec_from_file_location("evolved_agent", program_path)
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        
        # Prendi la funzione "cervello" dell'agente
        agent_decision_function = agent_module.get_action
        
        total_fitness = 0.0
        
        # 2. Esegui la simulazione N volte per ottenere un punteggio stabile
        for _ in range(NUM_GAMES_PER_EVAL):
            fitness, metrics = run_game_simulation(
                agent_decision_function=agent_decision_function,
                env_name=ENV_NAME,
                max_steps=MAX_STEPS_PER_GAME,
                obs_type="ram"
            )
            total_fitness += fitness
        
        final_score = total_fitness / NUM_GAMES_PER_EVAL

        # 3. Restituisce il punteggio a OpenEvolve
        return EvaluationResult(
            metrics={
                "combined_score": final_score, 
                "score": final_score,
                "correctness": 1.0
            }
        )
        
    except Exception as e:
        import traceback
        print(f"\n--- ERRORE DURANTE LA VALUTAZIONE DELL'AGENTE ---")
        traceback.print_exc() 
        # L'LLM ha prodotto codice rotto
        return EvaluationResult(
            metrics={
                "combined_score": -1000.0,  # <-- AGGIUNTO
                "score": -1000.0, 
                "correctness": 0.0
            },
            artifacts={"stderr": str(e)} # Feedback per l'LLM
        )

# Blocco 'main' per OpenEvolve
if __name__ == "__main__":
    if len(sys.argv) > 1:
        program_path = sys.argv[1]
        result = evaluate(program_path)
        print(json.dumps(result.to_dict()))
    else:
        print("Errore: Questo script deve essere chiamato da OpenEvolve con un percorso al programma.")
        sys.exit(1)