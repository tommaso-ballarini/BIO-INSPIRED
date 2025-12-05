import sys
import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from openevolve import run_evolution

# --- 1. SETUP PERCORSI (Dalla tua versione NUOVA) ---
# Percorso di QUESTO file (dentro .../freeway/run/)
current_dir = pathlib.Path(__file__).parent.resolve()

# Percorso BASE dell'esperimento (la cartella .../freeway/)
experiment_root = current_dir.parent 

# Percorsi relativi alla root dell'esperimento
initial_program_path = experiment_root / 'src' / 'initial_agent.py'
evaluator_path = experiment_root / 'src' / 'evaluator.py'
config_path = experiment_root / 'configs' / 'config.yaml'

# Output: salvati dentro la cartella run/results
output_dir = current_dir / 'results' 
history_csv = current_dir / 'history' / 'fitness_history.csv'

def setup_env():
    """Crea le cartelle di output se non esistono."""
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = current_dir / 'history'
    history_dir.mkdir(parents=True, exist_ok=True)

# --- 2. LOGICA PLOTTING (Dalla versione VECCHIA) ---
def plot_results():
    """Legge la history prodotta dall'evaluator e plotta i risultati."""
    print("\n--- Generazione Grafico Fitness ---")
    
    if not history_csv.exists():
        print(f"Nessun dato storico trovato in: {history_csv}")
        return

    try:
        df = pd.read_csv(history_csv)
        if df.empty:
            print("Il file CSV è vuoto.")
            return

        plt.figure(figsize=(10, 6))

        # --- Separazione Crash vs Esecuzioni Valide ---
        # Crash / errori: score <= -999.9 (convenzione evaluator)
        error_runs = df[df['score'] <= -999.9]
        # Valid runs
        valid_runs = df[df['score'] > -999.9].reset_index()

        # 1. Errori / crash in rosso
        if not error_runs.empty:
            plt.scatter(
                error_runs.index,
                error_runs['score'],
                c='red',
                label='Errori / Crash',
                alpha=0.3,
                s=10
            )

        # 2. Esecuzioni valide in blu
        if not valid_runs.empty:
            plt.scatter(
                valid_runs['index'],
                valid_runs['score'],
                c='blue',
                label='Esecuzioni Valide',
                alpha=0.7
            )

        # 3. Linea del miglior punteggio progressivo (Best So Far)
        if not df.empty:
            df['best_so_far'] = df['score'].cummax()
            plt.plot(df.index, df['best_so_far'], c='green', linewidth=2, label='Best So Far')

        plt.title("Evoluzione Agente Freeway")
        plt.xlabel("Numero Tentativo (Valutazione)")
        plt.ylabel("Fitness (Score Medio)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Salvataggio grafico
        output_plot = output_dir / 'fitness_plot.png'
        plt.savefig(output_plot)
        print(f"Grafico salvato in: {output_plot}")
        # plt.show() # Decommenta se vuoi vederlo a schermo

    except Exception as e:
        print(f"Errore durante il plotting: {e}")

# --- 3. RUNNER PRINCIPALE ---
def run_experiment():
    print(f"--- Avvio OpenEvolve Freeway ---")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    
    try:
        run_evolution(
            initial_program=str(initial_program_path),
            evaluator=str(evaluator_path),
            config=str(config_path),
            output_dir=str(output_dir)
        )
    except Exception as e:
        print(f"Errore Run: {e}")
    finally:
        # Generiamo il grafico anche se la run crasha o viene interrotta
        print("--- Esperimento Terminato (o Interrotto) ---")
        plot_results()

if __name__ == '__main__':
    setup_env()
    
    # Verifica esistenza file critici
    if initial_program_path.exists() and config_path.exists():
        run_experiment()
    else:
        print("ERRORE: File mancanti.")
        print(f"Initial Program: {initial_program_path}")
        print(f"Config: {config_path}")