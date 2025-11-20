# experiments/run_openevolve_freeway.py
import sys
import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from openevolve import run_evolution

# --- Percorsi ---
project_root = pathlib.Path(__file__).parent.parent.resolve()
initial_program_path = project_root / 'openevolve_experiments' / 'initial_agent.py' 
evaluator_path = project_root / 'openevolve_experiments' / 'evaluate_agent.py' 
config_path = project_root / 'configs' / 'openevolve_ea_config.yaml'
output_dir = project_root / 'evolution_results' / 'openevolve_ea_test_gemma3_1b'
history_csv = project_root / 'evolution_history' / 'fitness_history.csv'

def setup_paths():
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

def check_files():
    try:
        assert initial_program_path.exists(), f"Manca: {initial_program_path}"
        assert evaluator_path.exists(), f"Manca: {evaluator_path}"
        assert config_path.exists(), f"Manca: {config_path}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Pulizia opzionale della history precedente per nuovi run puliti
        # if history_csv.exists():
        #     os.remove(history_csv)
            
        return True
    except AssertionError as e:
        print(f"Errore Config: {e}")
        sys.exit(1)

def plot_results():
    """Legge la history prodotta dall'evaluator e plotta i risultati."""
    print("\n--- Generazione Grafico Fitness ---")
    if not history_csv.exists():
        print("Nessun dato storico trovato (il file csv non esiste).")
        return

    try:
        df = pd.read_csv(history_csv)
        if df.empty:
            print("Il file CSV è vuoto.")
            return

        # Filtriamo i -1000 (errori di sintassi/runtime) per il grafico principale
        valid_runs = df[df['score'] > -900].reset_index()
        
        plt.figure(figsize=(10, 6))
        
        # Plot di TUTTI i tentativi (in rosso gli errori, in blu i validi)
        plt.scatter(df.index, df['score'], c='red', label='Errori / Crash', alpha=0.3, s=10)
        plt.scatter(valid_runs['index'], valid_runs['score'], c='blue', label='Esecuzioni Valide', alpha=0.7)
        
        # Linea del miglior punteggio progressivo
        if not valid_runs.empty:
            df['best_so_far'] = df['score'].cummax()
            plt.plot(df.index, df['best_so_far'], c='green', linewidth=2, label='Best So Far')

        plt.title("Evoluzione Agente Freeway (Gemma 1B)")
        plt.xlabel("Numero Tentativo (Valutazione)")
        plt.ylabel("Fitness (Score Medio)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_plot = output_dir / 'fitness_plot.png'
        plt.savefig(output_plot)
        print(f"Grafico salvato in: {output_plot}")
        plt.show()
        
    except Exception as e:
        print(f"Errore durante il plotting: {e}")

def run_experiment():
    print(f"--- Avvio OpenEvolve ---\nOutput: {output_dir}")
    
    try:
        run_evolution(
            initial_program=str(initial_program_path),
            evaluator=str(evaluator_path),
            config=str(config_path),
            output_dir=str(output_dir)
        )
    except Exception as e:
        print(f"Errore Run: {e}")
        # Non usciamo subito, proviamo a plottare quello che abbiamo fatto finora

    print("--- Esperimento Terminato ---")
    plot_results()

if __name__ == '__main__':
    setup_paths()
    if check_files():
        run_experiment()