# experiments/run_openevolve_ea.py
import sys
import os
import pathlib
from openevolve import run_evolution

# --- Definizione Funzioni e Costanti ---
# Mettiamo i percorsi qui fuori, così sono "globali"
project_root = pathlib.Path(__file__).parent.parent.resolve()
initial_program_path = project_root / 'openevolve_experiments' / 'initial_program_ea.py'
evaluator_path = project_root / 'openevolve_experiments' / 'meta_evaluator.py'
config_path = project_root / 'configs' / 'openevolve_ea_config.yaml'
output_dir = project_root / 'evolution_results' / 'openevolve_ea_test_gemma3_1b'

def setup_paths():
    """Aggiunge il root del progetto al sys.path"""
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    print(f"Project Root impostato su: {project_root}")

def check_files():
    """Controlla che tutti i file necessari esistano prima di iniziare."""
    try:
        assert initial_program_path.exists(), f"File non trovato: {initial_program_path}"
        assert evaluator_path.exists(), f"File non trovato: {evaluator_path}"
        assert config_path.exists(), f"File non trovato: {config_path}"
        os.makedirs(output_dir, exist_ok=True)
        return True
    except AssertionError as e:
        print(f"\nErrore di configurazione: {e}")
        print("Controlla i percorsi e assicurati che i file siano al posto giusto.")
        sys.exit(1)

def run_experiment():
    """La funzione principale che esegue l'evoluzione."""
    print("--- Avvio Esperimento OpenEvolve (modalità Libreria) ---")
    print(f"Programma Iniziale: {initial_program_path}")
    print(f"Meta-Valutatore: {evaluator_path}")
    print(f"Configurazione: {config_path}")
    print(f"Output in: {output_dir}")
    print("-------------------------------------------------------")

    try:
        run_evolution(
            initial_program=str(initial_program_path),
            evaluator=str(evaluator_path),
            config=str(config_path),
            output_dir=str(output_dir)
        )
    except Exception as e:
        print(f"\nErrore catastrofico durante l'esecuzione di run_evolution: {e}")
        import traceback
        traceback.print_exc()

    print("--- Esperimento OpenEvolve Terminato ---")

# --- IL GUARDIANO "if __name__ == '__main__':" ---
# Questo è il punto di ingresso.
# Questo blocco viene eseguito SOLO quando lanci "python experiments/run_openevolve_ea.py"
# NON viene eseguito dai processi figli.
if __name__ == '__main__':
    setup_paths()
    if check_files():
        run_experiment()