import sys
import os
import importlib.util
import time
import re
import random
from pathlib import Path

# --- SETUP IMPORT WRAPPER ---
current_dir = os.path.dirname(os.path.abspath(__file__))
wrapper_dir = os.path.abspath(os.path.join(current_dir, '..', 'wrapper'))
sys.path.append(wrapper_dir)

try:
    from skiing_wrapper import SkiingOCAtariWrapper
except ImportError as e:
    print(f"Errore Import Wrapper: {e}")
    sys.exit(1)

from openevolve.evaluation_result import EvaluationResult

# --- CONFIGURAZIONE ---
ENV_NAME = 'ALE/Skiing-v5'
MAX_STEPS_PER_GAME = 2000
NUM_GAMES_PER_EVAL = 3 # Media su 3 partite diverse
EVAL_SEEDS = [3, 1, 22] # Semi deterministici per la riproducibilità

def log_to_csv(score):
    """Scrive il risultato su un CSV condiviso."""
    csv_path = os.environ.get("SKIING_HISTORY_PATH", "history_backup.csv")
    for _ in range(10):
        try:
            file_exists = os.path.exists(csv_path)
            with open(csv_path, 'a', encoding='utf-8') as f:
                if not file_exists:
                    f.write("timestamp,score\n")
                f.write(f"{time.time()},{score}\n")
            break
        except PermissionError:
            time.sleep(random.random() * 0.1)

def save_interesting_agent(code_string, score):
    """
    Salva il codice dell'agente se è interessante.
    LOGICA ANTI-CLONI: Non salva se esistono già >= 2 agenti con lo stesso score intero.
    """
    try:
        # Recupera il percorso della history per capire dove salvare
        csv_path = os.environ.get("SKIING_HISTORY_PATH", None)
        if csv_path:
            results_dir = os.path.dirname(csv_path)
            save_dir = os.path.join(results_dir, "interesting_agents")
        else:
            save_dir = "interesting_agents_backup"
            
        os.makedirs(save_dir, exist_ok=True)
        
        # --- LOGICA ANTI-CLONI ---
        score_int = int(score)
        prefix = f"agent_{score_int}_pts_"
        
        # Conta quanti file esistono già con questo punteggio esatto
        existing_agents = [f for f in os.listdir(save_dir) if f.startswith(prefix)]
        
        # Se abbiamo già 2 o più agenti con questo score, IGNORA il salvataggio
        if len(existing_agents) >= 2:
            return

        # Nome file con score e timestamp univoco
        filename = f"{prefix}{int(time.time()*1000)}.py"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code_string)
            
    except Exception as e:
        print(f"Errore salvataggio agente: {e}")

def clean_llm_code(code_string: str) -> str:
    """Rimuove i backticks di markdown."""
    pattern = r"```(?:python)?\s*(.*?)```"
    match = re.search(pattern, code_string, re.DOTALL)
    if match:
        return match.group(1).strip()
    return code_string.strip()

def run_custom_simulation(action_function, game_idx=0, visualization=False):
    """
    Esegue una simulazione singola.
    Usa game_idx per selezionare il seed deterministico dalla lista EVAL_SEEDS.
    """
    render_mode = "human" if visualization else None
    try:
        env = SkiingOCAtariWrapper(render_mode=render_mode)
    except Exception:
        return 0.0

    # --- LOGICA STOCASTICITÀ ---
    # Ruota i seed: 42 -> 43 -> 44 -> 42 ...
    current_seed = EVAL_SEEDS[game_idx % len(EVAL_SEEDS)]
    
    observation, info = env.reset(seed=current_seed)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    try:
        while not (terminated or truncated) and steps < MAX_STEPS_PER_GAME:
            try:
                action = int(action_function(observation))
            except Exception:
                return -10000.0

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if visualization: time.sleep(0.01)
    except Exception:
        return -10000.0
    finally:
        env.close()

    return total_reward

def evaluate(input_data: str) -> EvaluationResult:
    # 1. GESTIONE INPUT E CARICAMENTO CODICE
    code_to_exec = input_data
    if os.path.exists(input_data) and input_data.endswith('.py'):
        try:
            with open(input_data, 'r', encoding='utf-8') as f:
                code_to_exec = f.read()
        except Exception:
            log_to_csv(-9999.0)
            return EvaluationResult(metrics={'combined_score': -9999.0})

    # 2. PULIZIA
    cleaned_code = clean_llm_code(code_to_exec)

    # 3. COMPILAZIONE
    try:
        spec = importlib.util.spec_from_loader("agent_module", loader=None)
        agent_module = importlib.util.module_from_spec(spec)
        exec(cleaned_code, agent_module.__dict__)
        
        if not hasattr(agent_module, 'get_action'):
             log_to_csv(-9999.0)
             return EvaluationResult(metrics={'combined_score': -9999.0})
             
        get_action_func = agent_module.get_action

    except Exception:
        log_to_csv(-9999.0) 
        return EvaluationResult(metrics={'combined_score': -9999.0})

    # 4. SIMULAZIONE (LOOP STOCASTICO)
    total_score = 0
    for i in range(NUM_GAMES_PER_EVAL):
        score = run_custom_simulation(get_action_func, game_idx=i, visualization=False)
        total_score += score

    avg_score = total_score / NUM_GAMES_PER_EVAL
    
    # 5. LOGGING
    log_to_csv(avg_score)

    # 6. SALVATAGGIO CON NUOVA SOGLIA
    # Salva solo se supera 1000 punti (miglioramenti rispetto al seed_agent)
    if avg_score > 1000.0:
        save_interesting_agent(cleaned_code, avg_score)

    return EvaluationResult(metrics={'combined_score': avg_score})

if __name__ == "__main__":
    # --- MODIFICA: USIAMO SEED_AGENT ---
    try:
        import initial_agent
        print("Testing initial_agent (best from previous run)...")
        # Test visuale sulla prima mappa (Seed 42)
        score = run_custom_simulation(initial_agent.get_action, game_idx=0, visualization=True)
        print(f"Seed Agent Score (Game 1): {score}")
    except ImportError:
        print("❌ Errore: initial_agent.py non trovato nella cartella src!")