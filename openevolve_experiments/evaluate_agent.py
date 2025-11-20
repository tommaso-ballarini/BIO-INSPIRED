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

# --- Configurazione ---
ENV_NAME = 'Freeway-v4'
MAX_STEPS_PER_GAME = 1500
NUM_GAMES_PER_EVAL = 3

# Punteggi
REWARD_FACTOR = 10.0      # Punti per ogni attraversamento riuscito
COLLISION_PENALTY = 2.0   # Punti persi per ogni collisione

# Setup Percorsi
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
HISTORY_DIR = Path(project_root) / 'evolution_history'
HISTORY_CSV = HISTORY_DIR / 'fitness_history.csv'
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# Inizializza CSV
if not HISTORY_CSV.exists():
    with open(HISTORY_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(["timestamp", "score", "collisions", "filename"])

# Fix per Gym/Atari
try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    pass

def save_checkpoint(program_path, score, collisions):
    timestamp = int(time.time() * 1000)
    dest_name = f"attempt_{timestamp}_score_{score:.1f}.py"
    dest_path = HISTORY_DIR / dest_name
    try:
        shutil.copy(program_path, dest_path)
        with open(HISTORY_CSV, 'a', newline='') as f:
            csv.writer(f).writerow([timestamp, score, collisions, dest_name])
    except Exception:
        pass

def run_custom_simulation(agent_func):
    """Esegue il gioco e conta le collisioni guardando la RAM."""
    try:
        env = gym.make(ENV_NAME, obs_type='ram')
    except:
        env = gym.make('ALE/Freeway-v5', obs_type='ram')

    observation, _ = env.reset()
    total_reward = 0
    collisions = 0
    prev_y = 0
    
    # Inizializziamo prev_y
    prev_y = observation[14] # Byte 14 è la Y in Freeway

    for _ in range(MAX_STEPS_PER_GAME):
        action = agent_func(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        
        curr_y = observation[14]
        
        # Rilevamento Collisione:
        # Se la Y diminuisce bruscamente, siamo stati respinti da un'auto
        # (O l'agente ha premuto GIÙ, ma penalizziamo comunque il tornare indietro)
        if curr_y < prev_y:
            collisions += 1
        
        prev_y = curr_y
        total_reward += reward
        
        if terminated or truncated:
            break
            
    env.close()
    return total_reward, collisions

def evaluate(program_path: str):
    try:
        # Import dinamico
        spec = importlib.util.spec_from_file_location("evolved_agent", program_path)
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        agent_func = agent_module.get_action
        
        avg_fitness = 0.0
        avg_collisions = 0.0
        
        # Loop di valutazione multipla
        for _ in range(NUM_GAMES_PER_EVAL):
            game_reward, n_collisions = run_custom_simulation(agent_func)
            
            # Calcolo Fitness Personalizzata
            # Es: 2 attraversamenti (2.0) e 10 collisioni
            # Score = (2.0 * 10) - (10 * 1) = 10.0
            fitness = (game_reward * REWARD_FACTOR) - (n_collisions * COLLISION_PENALTY)
            
            avg_fitness += fitness
            avg_collisions += n_collisions
            
        final_score = avg_fitness / NUM_GAMES_PER_EVAL
        final_collisions = avg_collisions / NUM_GAMES_PER_EVAL

        save_checkpoint(program_path, final_score, final_collisions)

        # Output per OpenEvolve (formato JSON obbligatorio)
        from openevolve.evaluation_result import EvaluationResult
        return EvaluationResult(
            metrics={
                "combined_score": final_score, 
                "score": final_score,
                "collisions": final_collisions,
                "correctness": 1.0
            }
        )
        
    except Exception as e:
        # Gestione errori (codice rotto dall'LLM)
        save_checkpoint(program_path, -1000.0, 0)
        from openevolve.evaluation_result import EvaluationResult
        return EvaluationResult(
            metrics={"combined_score": -1000.0, "score": -1000.0, "correctness": 0.0},
            artifacts={"stderr": str(e)}
        )

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(json.dumps(evaluate(sys.argv[1]).to_dict()))
    else:
        sys.exit(1)