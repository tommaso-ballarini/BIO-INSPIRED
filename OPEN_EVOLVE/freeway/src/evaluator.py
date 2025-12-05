import sys
import os
import importlib.util
import json
import shutil
import time
import csv
import gymnasium as gym
import numpy as np
from pathlib import Path

# --- SETUP PERCORSI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
wrapper_dir = os.path.abspath(os.path.join(current_dir, '..', 'wrapper'))
sys.path.append(wrapper_dir)

# --- IMPORT WRAPPER ---
try:
    from freeway_wrapper import FreewayOCAtariWrapper
except ImportError as e:
    print(f"ERRORE CRITICO: Impossibile importare il wrapper da {wrapper_dir}")
    print(f"Dettaglio: {e}")
    sys.exit(1)

# Importiamo EvaluationResult
try:
    from openevolve.evaluation_result import EvaluationResult
except ImportError:
    print("ERRORE: Libreria 'openevolve' non trovata.")
    sys.exit(1)


# --- CONFIGURAZIONE COSTANTI ---
ENV_NAME = 'Freeway-v4'
MAX_STEPS_PER_GAME = 1000 
NUM_GAMES_PER_EVAL = 3 

# Parametri Fitness Aggiornati
REWARD_FACTOR = 100.0         # Punti per aver attraversato la strada
COLLISION_PENALTY = 0.2       # RIDOTTO: Piccola penalità, non blocca l'apprendimento
PROGRESS_FACTOR = 50.0        # NUOVO: Punti extra basati sull'altezza massima raggiunta
IDLE_PENALTY_PER_FRAME = 0.1  
MAX_IDLE_FRAMES = 60          

# Setup History
experiment_root = os.path.abspath(os.path.join(current_dir, '..'))
HISTORY_DIR = Path(experiment_root) / 'history'
HISTORY_CSV = HISTORY_DIR / 'fitness_history.csv'
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

if not HISTORY_CSV.exists():
    with open(HISTORY_CSV, 'w', newline='') as f:
        # Aggiungiamo max_y al log per curiosità
        csv.writer(f).writerow(["timestamp", "score", "collisions", "idle_penalty", "max_y", "filename"])

# Registrazione ALE
import ale_py
gym.register_envs(ale_py)

# --- FUNZIONI DI SUPPORTO ---

def save_checkpoint(program_path, score, collisions, idle_penalty, max_y):
    """Salva una copia dell'agente nella cartella history."""
    timestamp = int(time.time() * 1000)
    dest_name = f"attempt_{timestamp}_score_{score:.1f}.py"
    dest_path = HISTORY_DIR / dest_name
    try:
        shutil.copy(program_path, dest_path)
        with open(HISTORY_CSV, 'a', newline='') as f:
            csv.writer(f).writerow([timestamp, score, collisions, idle_penalty, max_y, dest_name])
    except Exception as e:
        print(f"Errore salvataggio checkpoint: {e}")

def run_custom_simulation(agent_func):
    """
    Esegue una partita. Restituisce: reward, collisions, idle_pen, max_y
    """
    try:
        env = gym.make(ENV_NAME, obs_type='ram', render_mode=None)
    except:
        env = gym.make('ALE/Freeway-v5', obs_type='ram', render_mode=None)

    env = FreewayOCAtariWrapper(env)

    observation, _ = env.reset()
    total_reward = 0.0
    collisions = 0
    idle_penalty_total = 0.0
    
    # max_y_reached: traccia il punto più alto raggiunto (0.0 - 1.0)
    max_y_reached = 0.0
    
    # obs[0] è la Y del pollo
    prev_y = observation[0] 
    
    idle_counter = 0
    last_action = None
    same_action_count = 0

    for _ in range(MAX_STEPS_PER_GAME):
        
        # 1. DECISIONE AGENTE
        try:
            action = agent_func(observation)
            if isinstance(action, (list, tuple, np.ndarray)):
                action = action[0]
            action = int(action)
        except Exception:
            action = 1 
        
        if action not in (0, 1, 2): action = 1

        # Check anti-loop
        if last_action is not None and action == last_action:
            same_action_count += 1
        else:
            last_action = action
            same_action_count = 1
        
        # 2. STEP
        observation, reward, terminated, truncated, _ = env.step(action)
        
        # 3. ANALISI
        curr_y = observation[0]
        
        # Aggiorna record altezza
        if curr_y > max_y_reached:
            max_y_reached = curr_y

        # Collisione (Y scende)
        if curr_y < (prev_y - 0.05):
            collisions += 1
            idle_counter = 0
        # Immobilità
        elif abs(curr_y - prev_y) < 0.001:
            idle_counter += 1
            if idle_counter > MAX_IDLE_FRAMES:
                idle_penalty_total += IDLE_PENALTY_PER_FRAME
        else:
            idle_counter = 0

        prev_y = curr_y
        total_reward += reward

        if same_action_count > 600: 
            break
        if terminated or truncated:
            break

    env.close()
    # Ritorna anche max_y_reached
    return total_reward, collisions, idle_penalty_total, max_y_reached


# --- FUNZIONE PRINCIPALE: EVALUATE ---
def evaluate(program_path: str):
    try:
        spec = importlib.util.spec_from_file_location("evolved_agent", program_path)
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        
        if not hasattr(agent_module, 'get_action'):
            raise ValueError("Funzione 'get_action' non trovata.")
            
        agent_func = agent_module.get_action

        avg_fitness = 0.0
        avg_collisions = 0.0
        avg_idle_penalty = 0.0
        avg_max_y = 0.0

        for _ in range(NUM_GAMES_PER_EVAL):
            # Unpack dei 4 valori
            g_reward, n_coll, idle_pen, max_y = run_custom_simulation(agent_func)

            # --- NUOVA LOGICA FITNESS ---
            # 1. Base: Reward del gioco (attraversamento completo)
            score = g_reward * REWARD_FACTOR
            
            # 2. Progress: Premia l'avanzamento anche se muore
            # Se arriva a metà (0.5), prende 25 punti. Incentiva il movimento.
            score += (max_y * PROGRESS_FACTOR)
            
            # 3. Collisioni: Penalità leggera
            # Puniamo poco (-2) così capisce che muoversi è meglio che stare fermi
            score -= (n_coll * COLLISION_PENALTY)
            
            # 4. Idleness: Penalità standard
            score -= idle_pen
            
            # 5. Penalità extra PIGRIZIA (se non si muove dalla base)
            if max_y < 0.05:
                score -= 50.0

            avg_fitness += score
            avg_collisions += n_coll
            avg_idle_penalty += idle_pen
            avg_max_y += max_y

        final_score = avg_fitness / NUM_GAMES_PER_EVAL
        final_collisions = avg_collisions / NUM_GAMES_PER_EVAL
        final_idle = avg_idle_penalty / NUM_GAMES_PER_EVAL
        final_max_y = avg_max_y / NUM_GAMES_PER_EVAL

        save_checkpoint(program_path, final_score, final_collisions, final_idle, final_max_y)

        return EvaluationResult(
            metrics={
                "combined_score": final_score,
                "score": final_score,
                "collisions": final_collisions,
                "idle_penalty": final_idle,
                "max_y": final_max_y,
                "correctness": 1.0,
            }
        )

    except Exception as e:
        print(f"EVALUATION ERROR: {e}")
        save_checkpoint(program_path, -1000.0, 0, 0, 0)
        return EvaluationResult(
            metrics={"combined_score": -1000.0, "score": -1000.0, "correctness": 0.0},
            artifacts={"stderr": str(e)}
        )

if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = evaluate(sys.argv[1])
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("Uso: python evaluator.py <path_to_agent_script.py>")
        sys.exit(1)