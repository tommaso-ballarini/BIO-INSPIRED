import sys
import os
import pathlib
import logging
import re
import datetime
import time
import pandas as pd
import matplotlib.pyplot as plt

# --- FIX WINDOWS ---
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

# Import opzionale tqdm
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from openevolve import run_evolution

# --- CONFIGURAZIONE CARTELLE ---
base_path = pathlib.Path(__file__).parent.resolve()

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"run_si_{timestamp}"
output_dir = base_path / 'results' / run_name

# File history specifico per Space Invaders
history_csv = output_dir / 'history.csv'
os.environ["SI_HISTORY_PATH"] = str(history_csv)

initial_program_path = base_path / 'src' / 'initial_agent.py'
evaluator_path = base_path / 'src' / 'evaluator.py'
config_path = base_path / 'configs' / 'config.yaml'

class ProgressBarHandler(logging.Handler):
    """Barra di progresso per OpenEvolve."""
    def __init__(self, total_iterations):
        super().__init__()
        if tqdm:
            self.pbar = tqdm(total=total_iterations, desc=f"Run: {run_name}", unit="gen", colour="green")
        self.last_seen = 0

    def emit(self, record):
        if not tqdm: return
        msg = self.format(record)
        match = re.search(r"Iteration (\d+):", msg)
        if match:
            current_iter = int(match.group(1))
            if current_iter > self.last_seen:
                increment = current_iter - self.last_seen
                self.pbar.update(increment)
                self.last_seen = current_iter

    def close(self):
        if tqdm: self.pbar.close()
        super().close()

def format_duration(seconds):
    """Converte secondi in formato H:M:S."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {int(s)}s"

def get_config_details():
    """Legge il config.yaml per estrarre info utili per il plot."""
    details = { "max_iter": 100, "pop_size": "N/A", "model": "Unknown" }
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
            iter_match = re.search(r"max_iterations:\s*(\d+)", content)
            if iter_match: details["max_iter"] = int(iter_match.group(1))
            pop_match = re.search(r"population_size:\s*(\d+)", content)
            if pop_match: details["pop_size"] = pop_match.group(1)
            model_match = re.search(r'- name:\s*["\']?([^"\']+)["\']?', content)
            if model_match: details["model"] = model_match.group(1)
    except Exception:
        pass
    return details

def plot_results(duration_str=""):
    """Genera grafico Fitness e Statistiche finali con Metadata."""
    print("\n--- Generazione Grafico Fitness ---")
    if not history_csv.exists():
        print(f"Nessun dato storico trovato in: {history_csv}")
        return

    config_data = get_config_details()
    
    try:
        df = pd.read_csv(history_csv)
        if df.empty: return

        df['attempt'] = range(1, len(df) + 1)
        
        # Setup Figura
        plt.figure(figsize=(12, 8))
        
        # Filtri Crash vs Validi
        error_runs = df[df['score'] <= -9]
        valid_runs = df[df['score'] > -9000]

        if not error_runs.empty:
            plt.scatter(error_runs['attempt'], error_runs['score'], c='red', label='Crash / Errori', alpha=0.3, s=15, marker='x')

        if not valid_runs.empty:
            plt.scatter(valid_runs['attempt'], valid_runs['score'], c='blue', label='Validi', alpha=0.6, s=20)

        # Best So Far
        df['best_so_far'] = df['score'].cummax()
        plt.plot(df['attempt'], df['best_so_far'], c='green', linewidth=2, label='Best So Far')

        plt.subplots_adjust(top=0.85)
        
        plt.suptitle(f"Evoluzione Space Invaders - {timestamp}", fontsize=16, fontweight='bold', y=0.95)
        
        subtitle = (f"Model: {config_data['model']} | Pop: {config_data['pop_size']} | "
                    f"Max Gen: {config_data['max_iter']} | Duration: {duration_str}")
        
        plt.title(subtitle, fontsize=10, pad=10, backgroundcolor='#eeeeee')

        plt.xlabel("Numero Valutazioni (Tentativi)")
        plt.ylabel("Fitness (Survival + Kills)")
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        safe_model_name = re.sub(r'[^\w\-]', '_', config_data['model'])
        plot_filename = f'fitness_plot_{safe_model_name}_pop{config_data["pop_size"]}.png'
        
        output_plot = output_dir / plot_filename
        plt.savefig(output_plot, dpi=150)
        plt.close()
        print(f"✅ Grafico salvato in: {output_plot}")

    except Exception as e:
        print(f"❌ Errore plot: {e}")
        import traceback
        traceback.print_exc()

def setup_env():
    output_dir.mkdir(parents=True, exist_ok=True)

def run_experiment():
    print(f"--- Avvio OpenEvolve SPACE INVADERS ---")
    print(f"Output Directory: {output_dir}")
    
    start_time = time.time()
    
    config_info = get_config_details()
    max_iter = config_info["max_iter"]
    
    print(f"Model: {config_info['model']} | Pop: {config_info['pop_size']}")
    
    bar_handler = ProgressBarHandler(max_iter)
    bar_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(bar_handler)
    
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
        end_time = time.time()
        duration_sec = end_time - start_time
        duration_str = format_duration(duration_sec)
        
        bar_handler.close()
        plot_results(duration_str)
        
        print(f"\n" + "="*40)
        print(f"⏱️  TEMPO TOTALE RUN: {duration_str}")
        print(f"="*40 + "\n")

if __name__ == '__main__':
    setup_env()
    run_experiment()