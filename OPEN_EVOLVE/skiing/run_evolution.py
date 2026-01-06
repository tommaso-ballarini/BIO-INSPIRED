import sys
import os
import pathlib
from openevolve import run_evolution

# --- PERCORSI DINAMICI ---
base_path = pathlib.Path(__file__).parent.resolve()

initial_program_path = base_path / 'src' / 'initial_agent.py'
evaluator_path = base_path / 'src' / 'evaluator.py'
config_path = base_path / 'configs' / 'config.yaml'

output_dir = base_path / 'results' / 'run_skiing_llm'

def setup_env():
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (base_path / 'history').exists():
        (base_path / 'history').mkdir()

def run_experiment():
    print(f"--- Avvio OpenEvolve SKIING ---")
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
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    setup_env()
    run_experiment()