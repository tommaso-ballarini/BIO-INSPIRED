import sys
import os
import importlib.util
import time
import re
import random
import numpy as np
from pathlib import Path
from ocatari.core import OCAtari

# --- SETUP IMPORT WRAPPER ---
current_dir = os.path.dirname(os.path.abspath(__file__))
wrapper_dir = os.path.abspath(os.path.join(current_dir, '..', 'wrapper'))
sys.path.append(wrapper_dir)

try:
    from si_wrapper import SpaceInvadersEgocentricWrapper
except ImportError:
    try:
        from wrapper_si_ego import SpaceInvadersEgocentricWrapper
    except ImportError as e:
        print(f"❌ Errore Import Wrapper: {e}")
        sys.exit(1)

from openevolve.evaluation_result import EvaluationResult

# --- CONFIGURAZIONE ---
ENV_NAME = 'ALE/SpaceInvaders-v5'
MAX_STEPS_PER_GAME = 4000
NUM_GAMES_PER_EVAL = 3
EVAL_SEEDS = [42, 43, 44]

def log_to_csv(score):
    """Scrive il risultato su un CSV condiviso."""
    csv_path = os.environ.get("SI_HISTORY_PATH", "history_si_backup.csv")
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
    Salva il codice dell'agente se supera una certa soglia.
    LIMITAZIONE: Non salva più di 2 agenti con lo stesso punteggio intero.
    """
    try:
        csv_path = os.environ.get("SI_HISTORY_PATH", None)
        if csv_path:
            results_dir = os.path.dirname(csv_path)
            save_dir = os.path.join(results_dir, "interesting_agents")
        else:
            save_dir = "interesting_agents_backup"
            
        os.makedirs(save_dir, exist_ok=True)
        
        # --- LOGICA ANTI-CLONI ---
        score_int = int(score)
        # Cerca file che iniziano con "agent_{score}_pts_"
        prefix = f"agent_{score_int}_pts_"
        existing_agents = [f for f in os.listdir(save_dir) if f.startswith(prefix)]
        
        # Se abbiamo già 2 o più agenti con questo score esatto, NON salvare
        if len(existing_agents) >= 2:
            return

        # Nome file univoco
        filename = f"{prefix}{int(time.time()*1000)}.py"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code_string)
            
    except Exception as e:
        print(f"Errore salvataggio agente interessante: {e}")

def clean_llm_code(code_string: str) -> str:
    """Rimuove i backticks di markdown."""
    pattern = r"```(?:python)?\s*(.*?)```"
    match = re.search(pattern, code_string, re.DOTALL)
    if match:
        return match.group(1).strip()
    return code_string.strip()

def run_custom_simulation(action_function, game_idx=0, visualization=False):
    render_mode = "human" if visualization else None
    
    try:
        env = OCAtari(ENV_NAME, mode="ram", hud=False, render_mode=render_mode)
        env = SpaceInvadersEgocentricWrapper(env, skip=4)
    except Exception as e:
        print(f"Errore Init Env: {e}")
        return 0.0

    current_seed = EVAL_SEEDS[game_idx % len(EVAL_SEEDS)]
    observation, info = env.reset(seed=current_seed)
    
    initial_delay = (game_idx * 10) % 30
    for _ in range(initial_delay):
        observation, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated: break

    episode_fitness = 0.0
    steps = 0
    terminated = False
    truncated = False

    try:
        while not (terminated or truncated) and steps < MAX_STEPS_PER_GAME:
            try:
                action = int(action_function(observation))
            except Exception:
                # Penalità ridotta per evitare di scartare logiche promettenti ma con bug minori
                return -10.0

            # CALCOLO FITNESS (Aiming + Survival)
            danger_level = observation[3]
            is_safe = danger_level < 0.25
            
            if action == 1:
                episode_fitness -= 0.05
            
            if is_safe:
                rel_x = observation[11]
                if abs(rel_x) < 0.15:
                    episode_fitness += 0.02
            else:
                episode_fitness -= (danger_level * 0.2)

            observation, reward, terminated, truncated, info = env.step(action)
            
            if reward > 0:
                episode_fitness += reward
                episode_fitness += (reward * 0.5)

            steps += 1
            if visualization: time.sleep(0.01)

        if episode_fitness <= 0:
            episode_fitness = max(0.001, steps / 10000.0)

    except Exception as e:
        print(f"Errore runtime: {e}")
        return -10000.0
    finally:
        env.close()

    return episode_fitness

def evaluate(input_data: str) -> EvaluationResult:
    code_to_exec = input_data
    if os.path.exists(input_data) and input_data.endswith('.py'):
        try:
            with open(input_data, 'r', encoding='utf-8') as f:
                code_to_exec = f.read()
        except Exception:
            return EvaluationResult(metrics={'combined_score': -9999.0})

    cleaned_code = clean_llm_code(code_to_exec)
    
    try:
        spec = importlib.util.spec_from_loader("agent_module", loader=None)
        agent_module = importlib.util.module_from_spec(spec)
        exec(cleaned_code, agent_module.__dict__)
        
        if not hasattr(agent_module, 'get_action'):
             return EvaluationResult(metrics={'combined_score': -9999.0})
             
        get_action_func = agent_module.get_action

    except Exception:
        return EvaluationResult(metrics={'combined_score': -9999.0})

    total_score = 0
    for i in range(NUM_GAMES_PER_EVAL):
        score = run_custom_simulation(get_action_func, game_idx=i, visualization=False)
        total_score += score

    avg_score = total_score / NUM_GAMES_PER_EVAL
    
    log_to_csv(avg_score)

    # --- SALVATAGGIO INTELLIGENTE ---
    # Salva se score > 1600 (per prendere anche comportamenti base)
    # La funzione save_interesting_agent gestirà la limitazione dei duplicati
    if avg_score > 1600.0:
        save_interesting_agent(cleaned_code, avg_score)

    return EvaluationResult(metrics={'combined_score': avg_score})

if __name__ == "__main__":
    import initial_agent
    print("Testing initial agent...")
    score = run_custom_simulation(initial_agent.get_action, visualization=True)
    print(f"Initial Agent Fitness: {score}")