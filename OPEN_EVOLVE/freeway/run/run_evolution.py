import sys
import os
import pathlib
from openevolve import run_evolution
# Se vuoi fare plotting qui, importa matplotlib, altrimenti separa in visualize.py

# --- PERCORSI DINAMICI ---
# Base: OPEN_EVOLVE/freeway/
base_path = pathlib.Path(__file__).parent.resolve()

initial_program_path = base_path / 'src' / 'initial_agent.py'
evaluator_path = base_path / 'src' / 'evaluator.py'
config_path = base_path / 'configs' / 'config.yaml'

# Output dentro la cartella dell'esperimento
output_dir = base_path / 'results' / 'run_gemma_test'
history_csv = base_path / 'history' / 'fitness_history.csv'

def setup_env():
    # Crea cartelle se non esistono
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (base_path / 'history').exists():
        (base_path / 'history').mkdir()

def run_experiment():
    print(f"--- Avvio OpenEvolve Freeway ---")
    print(f"Config: {config_path}")
    
    try:
        run_evolution(
            initial_program=str(initial_program_path),
            evaluator=str(evaluator_path),
            config=str(config_path),
            output_dir=str(output_dir)
        )
    except Exception as e:
        print(f"Errore Run: {e}")

if __name__ == '__main__':
    setup_env()
    # Verifica esistenza file critici
    if initial_program_path.exists() and config_path.exists():
        run_experiment()
    else:
        print(f"ERRORE: File mancanti.\nCercato in: {base_path}")