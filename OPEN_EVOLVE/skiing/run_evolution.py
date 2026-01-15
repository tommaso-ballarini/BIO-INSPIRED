import sys
import os
import pathlib
import logging
import re
import datetime
import time
import threading
import pandas as pd
import matplotlib.pyplot as plt

# --- WINDOWS FIX ---
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from openevolve import run_evolution

# --- PATH CONFIGURATION ---
base_path = pathlib.Path(__file__).parent.resolve()

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"run_skiing_{timestamp}"
output_dir = base_path / 'results' / run_name

history_csv = output_dir / 'history.csv'
gen_stats_csv = output_dir / 'generation_stats.csv'
os.environ["SKIING_HISTORY_PATH"] = str(history_csv)

initial_program_path = base_path / 'src' / 'initial_agent.py'
#initial_program_path = base_path / 'src' / 'seed_agent.py'
evaluator_path = base_path / 'src' / 'evaluator.py'
config_path = base_path / 'configs' / 'config.yaml'

class LogStatsWatcher(threading.Thread):
    """
    Real-time log parser. Extracts generation stats (Avg, Best, Diversity) to CSV.
    """
    def __init__(self, log_dir, csv_file):
        super().__init__()
        self.log_dir = log_dir
        self.csv_file = csv_file
        self.stop_event = threading.Event()
        self.daemon = True

    def run(self):
        # Wait for log file creation
        log_file = None
        while not self.stop_event.is_set():
            if os.path.exists(self.log_dir):
                logs = sorted([f for f in os.listdir(self.log_dir) if f.endswith('.log')])
                if logs:
                    log_file = os.path.join(self.log_dir, logs[-1])
                    break
            time.sleep(2)
        
        if not log_file: return

        # Init CSV
        with open(self.csv_file, 'w', encoding='utf-8') as f:
            f.write("generation,best_score,avg_score,diversity\n")

        print(f"📊 Stats Watcher active on: {os.path.basename(log_file)}")

        # Regex to parse: INFO - ... best=1599.75, avg=837.44, diversity=147.18, gen=4074
        pattern = re.compile(r"best=([\d\.-]+),\s*avg=([\d\.-]+),\s*diversity=([\d\.-]+),\s*gen=(\d+)")

        with open(log_file, 'r', encoding='utf-8') as f:
            while not self.stop_event.is_set():
                line = f.readline()
                if not line:
                    time.sleep(1)
                    continue
                
                if "Island 0:" in line and "avg=" in line:
                    match = pattern.search(line)
                    if match:
                        with open(self.csv_file, 'a', encoding='utf-8') as csv_f:
                            csv_f.write(f"{match.group(4)},{match.group(1)},{match.group(2)},{match.group(3)}\n")

    def stop(self):
        self.stop_event.set()

class ProgressBarHandler(logging.Handler):
    """Progress bar integration for OpenEvolve."""
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
    """Converts seconds to H:M:S."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {int(s)}s"

def get_config_details():
    """Extracts metadata from config.yaml for plotting."""
    details = { "max_iter": 30, "pop_size": "N/A", "model": "Unknown" }
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
    """Generates Fitness and Stats plots."""
    print("\n--- Generating Plots ---")
    config_data = get_config_details()
    
    # --- 1. HISTORY PLOT ---
    if history_csv.exists():
        try:
            df = pd.read_csv(history_csv)
            if not df.empty:
                df['attempt'] = range(1, len(df) + 1)
                
                plt.figure(figsize=(12, 8))
                
                error_runs = df[df['score'] <= -9000]
                valid_runs = df[df['score'] > -9000]

                if not error_runs.empty:
                    plt.scatter(error_runs['attempt'], error_runs['score'], c='red', label='Crash / Error', alpha=0.3, s=15, marker='x')

                if not valid_runs.empty:
                    plt.scatter(valid_runs['attempt'], valid_runs['score'], c='blue', label='Valid', alpha=0.6, s=20)

                df['best_so_far'] = df['score'].cummax()
                plt.plot(df['attempt'], df['best_so_far'], c='green', linewidth=2, label='Best So Far')

                plt.subplots_adjust(top=0.85)
                plt.suptitle(f"Skiing Evolution - {timestamp}", fontsize=16, fontweight='bold', y=0.95)
                
                subtitle = (f"Model: {config_data['model']} | Pop: {config_data['pop_size']} | "
                            f"Max Gen: {config_data['max_iter']} | Duration: {duration_str}")
                
                plt.title(subtitle, fontsize=10, pad=10, backgroundcolor='#eeeeee')

                plt.xlabel("Evaluations (Attempts)")
                plt.ylabel("Score")
                plt.legend(loc='lower right')
                plt.grid(True, alpha=0.3)

                safe_model_name = re.sub(r'[^\w\-]', '_', config_data['model'])
                plot_filename = f'fitness_plot_{safe_model_name}_pop{config_data["pop_size"]}.png'
                
                output_plot = output_dir / plot_filename
                plt.savefig(output_plot, dpi=150)
                plt.close()
                print(f"✅ History plot saved to: {output_plot}")
        except Exception as e:
            print(f"❌ History plot error: {e}")

    # --- 2. GENERATION STATS PLOT ---
    if gen_stats_csv.exists():
        try:
            df = pd.read_csv(gen_stats_csv)
            if not df.empty:
                plt.figure(figsize=(10, 6))
                plt.plot(df['generation'], df['best_score'], 'g-', label='Best Score')
                plt.plot(df['generation'], df['avg_score'], 'b-', label='Avg Score')
                plt.fill_between(df['generation'], df['avg_score'], df['best_score'], color='green', alpha=0.1)
                
                plt.title(f"Generation Stats (Avg vs Best) - Skiing")
                plt.xlabel("Generation")
                plt.ylabel("Score")
                plt.legend()
                plt.grid(True)
                plt.savefig(output_dir / 'generation_plot.png')
                plt.close()
                print("✅ Generation stats plot saved.")
        except Exception as e:
            print(f"Gen stats plot error: {e}")

def setup_env():
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (base_path / 'history').exists():
        (base_path / 'history').mkdir()

def run_experiment():
    print(f"--- Starting OpenEvolve SKIING ---")
    print(f"Output Directory: {output_dir}")
    
    start_time = time.time()
    
    config_info = get_config_details()
    max_iter = config_info["max_iter"]
    
    print(f"Model: {config_info['model']} | Pop: {config_info['pop_size']}")
    
    # --- START WATCHER ---
    log_dir = output_dir / "logs"
    watcher = LogStatsWatcher(str(log_dir), str(gen_stats_csv))
    watcher.start()
    
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
        print(f"Run Error: {e}")
    finally:
        # --- STOP WATCHER ---
        watcher.stop()
        
        end_time = time.time()
        duration_sec = end_time - start_time
        duration_str = format_duration(duration_sec)
        
        bar_handler.close()
        plot_results(duration_str)
        
        print(f"\n" + "="*40)
        print(f"⏱️  TOTAL RUN TIME: {duration_str}")
        print(f"="*40 + "\n")

if __name__ == '__main__':
    setup_env()
    run_experiment()