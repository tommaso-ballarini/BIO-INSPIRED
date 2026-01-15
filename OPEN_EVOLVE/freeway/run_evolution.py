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

# --- Windows terminal encoding fix ---
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from openevolve import run_evolution
from openevolve.config import Config, LLMModelConfig

# --- Path configuration ---
run_dir = pathlib.Path(__file__).parent.resolve()
project_dir = run_dir.parent 
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"run_freeway_{timestamp}"
output_dir = run_dir / 'results' / run_name
history_csv = output_dir / 'history.csv'
gen_stats_csv = output_dir / 'generation_stats.csv'
os.environ["FREEWAY_HISTORY_PATH"] = str(history_csv)

initial_program_path = project_dir / 'src' / 'initial_agent.py'
evaluator_path = project_dir / 'src' / 'evaluator.py'

config_path = project_dir / 'configs' / 'config.YAML' 
if not config_path.exists():
    config_path = project_dir / 'configs' / 'config.yaml'

class LogStatsWatcher(threading.Thread):
    """Background thread to extract stats from logs to CSV"""
    def __init__(self, log_dir, csv_file):
        super().__init__()
        self.log_dir = log_dir
        self.csv_file = csv_file
        self.stop_event = threading.Event()
        self.daemon = True

    def run(self):
        log_file = None
        while not self.stop_event.is_set():
            if os.path.exists(self.log_dir):
                logs = sorted([f for f in os.listdir(self.log_dir) if f.endswith('.log')])
                if logs:
                    log_file = os.path.join(self.log_dir, logs[-1])
                    break
            time.sleep(2)
        
        if not log_file: return

        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', encoding='utf-8') as f:
                f.write("generation,best_score,avg_score,diversity\n")

        print(f"📊 Stats Watcher active on: {os.path.basename(log_file)}")
        pattern = re.compile(r"best=([\d\.-]+),\s*avg=([\d\.-]+),\s*diversity=([\d\.-]+),\s*gen=(\d+)")

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                f.seek(0, 2)
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
        except Exception as e:
            print(f"Watcher error: {e}")

    def stop(self):
        self.stop_event.set()

class ProgressBarHandler(logging.Handler):
    """Integrates OpenEvolve logging with tqdm progress bar"""
    def __init__(self, total_iterations):
        super().__init__()
        if tqdm:
            self.pbar = tqdm(total=total_iterations, desc=f"Run: {run_name}", unit="gen", colour="blue")
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

def plot_results():
    """Generates fitness evolution plots"""
    print("\n--- Generating Plots ---")
    if history_csv.exists():
        try:
            df = pd.read_csv(history_csv)
            if not df.empty:
                df['attempt'] = range(1, len(df) + 1)
                plt.figure(figsize=(12, 8))
                
                valid_runs = df[df['score'] > -400]
                if not valid_runs.empty:
                    plt.scatter(valid_runs['attempt'], valid_runs['score'], c='blue', alpha=0.5, s=10)

                df['best_so_far'] = df['score'].cummax()
                plt.plot(df['attempt'], df['best_so_far'], c='green', linewidth=2, label='Best So Far')
                
                plt.title(f"Freeway Evolution - {timestamp}")
                plt.xlabel("Attempts")
                plt.ylabel("Score (Shaped)")
                plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / 'fitness_history.png')
                plt.close()
                print("✅ History plot saved.")
        except Exception as e:
            print(f"❌ Plotting error: {e}")

def setup_env():
    output_dir.mkdir(parents=True, exist_ok=True)

def run_experiment():
    print(f"--- Starting OpenEvolve FREEWAY ---")
    print(f"Run Dir: {run_dir}")
    print(f"Project Dir: {project_dir}")
    print(f"Config Path: {config_path}")


    # 1. Load Configuration
    config = None
    try:
        if config_path.exists():
            config = Config.from_yaml(str(config_path))
            print("✅ Config loaded from YAML.")
        else:
            print(f"⚠️ YAML not found in {config_path}, using base config.")
            config = Config()
    except Exception as e:
        print(f"⚠️ YAML error: {e}. Using default config.")
        config = Config()

    # 2. Manual LLM Override (Ollama)
    if not hasattr(config, 'llm') or not config.llm.models:
        print("🔧 Configurazione LLM manuale (Ollama)...")
        # Ricostruiamo la sezione LLM se manca del tutto
        if not hasattr(config, 'llm'):
             from openevolve.config import LLMConfig
             config.llm = LLMConfig()
             
        config.llm.models = [
            LLMModelConfig(
                name="qwen2.5-coder:7b", 
                weight=1.0,
                api_base="http://localhost:11434/v1",
                api_key="ollama"
            )
        ]
        config.llm.api_base = "http://localhost:11434/v1"
        config.llm.api_key = "ollama"
        config.llm.temperature = 0.8
        

    # 3. Setup Monitoring
    log_dir = output_dir / "logs"
    watcher = LogStatsWatcher(str(log_dir), str(gen_stats_csv))
    watcher.start()
    
    max_iter = config.max_iterations if hasattr(config, 'max_iterations') else 5000
    bar_handler = ProgressBarHandler(max_iter)
    bar_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(bar_handler)
    
    # 4. Start Evolution Process
    try:
        run_evolution(
            initial_program=str(initial_program_path),
            evaluator=str(evaluator_path),
            config=config, 
            output_dir=str(output_dir)
        )
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        watcher.stop()
        bar_handler.close()
        plot_results()

if __name__ == '__main__':
    setup_env()
    run_experiment()