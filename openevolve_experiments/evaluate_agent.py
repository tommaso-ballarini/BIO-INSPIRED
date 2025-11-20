import sys
import os
import importlib.util
import json
import shutil
import time
import csv
from pathlib import Path

# --- Aggiungi il root del progetto al PYTHONPATH ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# ---

try:
    from core.evaluator import run_game_simulation
    from openevolve.evaluation_result import EvaluationResult
except ImportError:
    # Fallback silenzioso o print su stderr per non rompere il parsing JSON del genitore
    sys.stderr.write("Errore: Impossibile importare i moduli da /core o openevolve.\n")
    sys.exit(1)

# --- Configurazione ---
ENV_NAME = 'Freeway-v4'
MAX_STEPS_PER_GAME = 1500
NUM_GAMES_PER_EVAL = 3

# Cartelle per Checkpoint e Storia
HISTORY_DIR = Path(project_root) / 'evolution_history'
HISTORY_CSV = HISTORY_DIR / 'fitness_history.csv'
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# Inizializza CSV se non esiste
if not HISTORY_CSV.exists():
    with open(HISTORY_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "score", "filename"])

def save_checkpoint(program_path, score):
    """Salva una copia del codice generato per analisi futura."""
    timestamp = int(time.time() * 1000)
    # Nome file: attempt_TIMESTAMP_SCORE.py (così li ordiniamo facilmente e vediamo subito la qualità)
    dest_name = f"attempt_{timestamp}_score_{score:.1f}.py"
    dest_path = HISTORY_DIR / dest_name
    
    try:
        shutil.copy(program_path, dest_path)
        
        # Aggiorna il CSV
        with open(HISTORY_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, score, dest_name])
            
    except Exception as e:
        sys.stderr.write(f"Errore salvataggio checkpoint: {e}\n")

def evaluate(program_path: str):
    final_score = -1000.0
    correctness = 0.0
    error_log = None

    try:
        # 1. Importa dinamicamente
        spec = importlib.util.spec_from_file_location("evolved_agent", program_path)
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        
        agent_decision_function = agent_module.get_action
        
        # 2. Esegui simulazione
        total_fitness = 0.0
        for _ in range(NUM_GAMES_PER_EVAL):
            fitness, metrics = run_game_simulation(
                agent_decision_function=agent_decision_function,
                env_name=ENV_NAME,
                max_steps=MAX_STEPS_PER_GAME,
                obs_type="ram"
            )
            total_fitness += fitness
        
        final_score = total_fitness / NUM_GAMES_PER_EVAL
        correctness = 1.0

        # Salva checkpoint del successo
        save_checkpoint(program_path, final_score)

        return EvaluationResult(
            metrics={
                "combined_score": final_score, 
                "score": final_score,
                "correctness": 1.0
            }
        )
        
    except Exception as e:
        # Salva checkpoint anche del fallimento (utile per debuggare errori di sintassi gemma)
        save_checkpoint(program_path, -1000.0)
        
        return EvaluationResult(
            metrics={
                "combined_score": -1000.0,
                "score": -1000.0, 
                "correctness": 0.0
            },
            artifacts={"stderr": str(e)}
        )

if __name__ == "__main__":
    if len(sys.argv) > 1:
        program_path = sys.argv[1]
        result = evaluate(program_path)
        print(json.dumps(result.to_dict()))
    else:
        sys.exit(1)