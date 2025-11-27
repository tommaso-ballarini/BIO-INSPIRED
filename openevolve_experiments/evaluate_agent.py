# openevolve_experiments/evaluate_agent.py
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

# --- IMPORTANTE: Assicurati che questo import funzioni ---
# Se core.wrappers non viene trovato, controlla la struttura delle cartelle
try:
    from core.wrappers import FreewayOCAtariWrapper
except ImportError:
    # Fallback se esegui lo script da una cartella diversa
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.wrappers import FreewayOCAtariWrapper

from openevolve.evaluation_result import EvaluationResult

# --- Configurazione ---
ENV_NAME = 'Freeway-v4'
MAX_STEPS_PER_GAME = 1500
NUM_GAMES_PER_EVAL = 3

# Parametri Fitness (Resi più severi)
REWARD_FACTOR = 100.0         # Premiare tantissimo il successo
COLLISION_PENALTY = 10.0      # Punire fortemente l'impatto
IDLE_PENALTY_PER_FRAME = 0.1  # Punire lo stare fermi troppo a lungo
MAX_IDLE_FRAMES = 60          # Abbassato a 2 secondi di tolleranza

# Setup Percorsi
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
HISTORY_DIR = Path(project_root) / 'evolution_history'
HISTORY_CSV = HISTORY_DIR / 'fitness_history.csv'
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

if not HISTORY_CSV.exists():
    with open(HISTORY_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(["timestamp", "score", "collisions", "idle_penalty", "filename"])

try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    pass

def save_checkpoint(program_path, score, collisions, idle_penalty):
    timestamp = int(time.time() * 1000)
    dest_name = f"attempt_{timestamp}_score_{score:.1f}.py"
    dest_path = HISTORY_DIR / dest_name
    try:
        shutil.copy(program_path, dest_path)
        with open(HISTORY_CSV, 'a', newline='') as f:
            csv.writer(f).writerow([timestamp, score, collisions, idle_penalty, dest_name])
    except Exception:
        pass

def run_custom_simulation(agent_func):
    """Esegue il gioco usando il Wrapper per l'agente ma la RAM per l'arbitro."""
    try:
        env = gym.make(ENV_NAME, obs_type='ram')
    except:
        env = gym.make('ALE/Freeway-v5', obs_type='ram')

    # --- ATTIVAZIONE WRAPPER ---
    translator = FreewayOCAtariWrapper(env)
    # ---------------------------

    observation, _ = env.reset()
    total_reward = 0.0
    collisions = 0
    idle_penalty_total = 0.0

    prev_y = observation[14]
    idle_counter = 0
    last_action = None
    same_action_count = 0

    for _ in range(MAX_STEPS_PER_GAME):
        
        # 1. TRADUZIONE: RAM (128) -> Features (11 floats)
        agent_view = translator.observation(observation)
        
        # 2. DECISIONE AGENTE
        try:
            # L'agente vede solo il vettore semplificato!
            action = agent_func(agent_view)
            action = int(action)
        except Exception:
            action = 1 # Fallback UP
        
        if action not in (0, 1, 2): action = 1

        # Check anti-loop (se l'agente si blocca a premere sempre lo stesso tasto)
        if last_action is None or action != last_action:
            last_action = action
            same_action_count = 1
        else:
            same_action_count += 1
        
        # 3. FISICA GIOCO (Input all'ambiente reale)
        observation, reward, terminated, truncated, _ = env.step(action)
        
        # 4. VALUTAZIONE (Guardiamo la verità nella RAM)
        curr_y = observation[14]

        # Collisione: Y scende bruscamente
        if curr_y < prev_y:
            collisions += 1
            idle_counter = 0
        # Immobilità
        elif curr_y == prev_y:
            idle_counter += 1
            if idle_counter > MAX_IDLE_FRAMES:
                idle_penalty_total += IDLE_PENALTY_PER_FRAME
        else:
            idle_counter = 0

        prev_y = curr_y
        total_reward += reward

        if same_action_count > 600: # Tagliamo se è bloccato da 20 secondi
            break
        if terminated or truncated:
            break

    env.close()
    return total_reward, collisions, idle_penalty_total

def evaluate(program_path: str):
    try:
        spec = importlib.util.spec_from_file_location("evolved_agent", program_path)
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        agent_func = agent_module.get_action

        avg_fitness = 0.0
        avg_collisions = 0.0
        avg_idle_penalty = 0.0

        for _ in range(NUM_GAMES_PER_EVAL):
            game_reward, n_collisions, idle_pen = run_custom_simulation(agent_func)

            # Fitness aggiornata:
            # Reward molto alto (100) per incentivare l'attraversamento
            # Penalità collisione alta (10) per incentivare la prudenza
            fitness = (game_reward * REWARD_FACTOR) - (n_collisions * COLLISION_PENALTY) - idle_pen
            
            # Se prende troppe botte senza segnare, penalità extra
            if game_reward == 0 and n_collisions > 3:
                fitness -= 200.0

            avg_fitness += fitness
            avg_collisions += n_collisions
            avg_idle_penalty += idle_pen

        final_score = avg_fitness / NUM_GAMES_PER_EVAL
        final_collisions = avg_collisions / NUM_GAMES_PER_EVAL
        final_idle = avg_idle_penalty / NUM_GAMES_PER_EVAL

        save_checkpoint(program_path, final_score, final_collisions, final_idle)

        return EvaluationResult(
            metrics={
                "combined_score": final_score,
                "score": final_score,
                "collisions": final_collisions,
                "idle_penalty": final_idle,
                "correctness": 1.0,
            }
        )

    except Exception as e:
        save_checkpoint(program_path, -1000.0, 0, 0)
        return EvaluationResult(
            metrics={"combined_score": -1000.0, "score": -1000.0, "correctness": 0.0},
            artifacts={"stderr": str(e)}
        )

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(json.dumps(evaluate(sys.argv[1]).to_dict()))
    else:
        sys.exit(1)