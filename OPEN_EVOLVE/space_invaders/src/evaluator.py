import sys
import os
import importlib.util
import time
import re
import random
import numpy as np
from pathlib import Path
from ocatari.core import OCAtari

# --- WRAPPER SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
wrapper_dir = os.path.abspath(os.path.join(current_dir, '..', 'wrapper'))
sys.path.append(wrapper_dir)

try:
    from si_wrapper import SpaceInvadersEgocentricWrapper
except ImportError:
    try:
        from wrapper_si_ego import SpaceInvadersEgocentricWrapper
    except ImportError as e:
        print(f"❌ Wrapper Import Error: {e}")
        sys.exit(1)

from openevolve.evaluation_result import EvaluationResult

# --- CONFIG ---
ENV_NAME = 'ALE/SpaceInvaders-v5'
MAX_STEPS_PER_GAME = 4000
NUM_GAMES_PER_EVAL = 3

def log_to_csv(score):
    """Writes result to a shared CSV."""
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
    Saves agent code if it passes the threshold.
    """
    try:
        csv_path = os.environ.get("SI_HISTORY_PATH", None)
        if csv_path:
            results_dir = os.path.dirname(csv_path)
            save_dir = os.path.join(results_dir, "interesting_agents")
        else:
            save_dir = "interesting_agents_backup"
            
        os.makedirs(save_dir, exist_ok=True)
        
        # --- ANTI-CLONE LOGIC ---
        score_int = int(score)
        prefix = f"agent_{score_int}_pts_"
        existing_agents = [f for f in os.listdir(save_dir) if f.startswith(prefix)]
        
        # If we already have 2 or more agents with this exact score, DO NOT save
        if len(existing_agents) >= 2:
            return

        filename = f"{prefix}{int(time.time()*1000)}.py"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code_string)
            
    except Exception as e:
        print(f"Error saving interesting agent: {e}")

def clean_llm_code(code_string: str) -> str:
    """Removes markdown backticks."""
    pattern = r"```(?:python)?\s*(.*?)```"
    match = re.search(pattern, code_string, re.DOTALL)
    if match:
        return match.group(1).strip()
    return code_string.strip()

def run_custom_simulation(action_function, specific_seed=None, visualization=False):
    """
    Runs a single simulation.
    If specific_seed is None: use random seed > 100 (Training).
    If specific_seed is Set: use specific seed (Test/Validation).
    """
    render_mode = "human" if visualization else None
    
    try:
        env = OCAtari(ENV_NAME, mode="ram", hud=False, render_mode=render_mode)
        env = SpaceInvadersEgocentricWrapper(env, skip=4)
    except Exception as e:
        print(f"Env Init Error: {e}")
        return 0.0

    # --- RANDOM SEED LOGIC (TRAINING vs TEST) ---
    if specific_seed is None:
        # Training: Random safe seed (> 100)
        current_seed = random.randint(100, 1000000)
    else:
        # Test: Specific seed
        current_seed = specific_seed

    observation, info = env.reset(seed=current_seed)
    
    
    episode_fitness = 0.0
    steps = 0
    terminated = False
    truncated = False

    try:
        while not (terminated or truncated) and steps < MAX_STEPS_PER_GAME:
            try:
                action = int(action_function(observation))
            except Exception:
                return -10.0

            # FITNESS CALCULATION (Aiming + Survival)
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
        print(f"Runtime error: {e}")
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
        # Calls simulation with RANDOM seed (default behavior)
        score = run_custom_simulation(action_function=get_action_func, specific_seed=None, visualization=False)
        total_score += score

    avg_score = total_score / NUM_GAMES_PER_EVAL
    
    log_to_csv(avg_score)

    # --- INTELLIGENT SAVING ---
    if avg_score > 1600.0:
        save_interesting_agent(cleaned_code, avg_score)

    return EvaluationResult(metrics={'combined_score': avg_score})

if __name__ == "__main__":
    try:
        import initial_agent
        print("Testing initial_agent (Validation Run)...")
        
        # Use reserved seed (< 100) for visual test like other scripts
        TEST_SEED = 42 
        score = run_custom_simulation(initial_agent.get_action, specific_seed=TEST_SEED, visualization=True)
        
        print(f"Initial Agent Fitness (Seed {TEST_SEED}): {score}")
    except ImportError:
        print("❌ Error: initial_agent.py not found!")