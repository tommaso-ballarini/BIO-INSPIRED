import sys
import os
import importlib.util
import time
import re
import random
import gymnasium as gym
from pathlib import Path

# --- WRAPPER SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
wrapper_dir = os.path.abspath(os.path.join(current_dir, '..', 'wrapper'))
sys.path.append(wrapper_dir)

try:
    from freeway_wrapper import FreewayEvoWrapper
except ImportError as e:
    print(f"Wrapper Import Error: {e}")
    sys.exit(1)

from openevolve.evaluation_result import EvaluationResult

# --- CONFIG ---
ENV_NAME = 'ALE/Freeway-v5'
MAX_STEPS_PER_GAME = 2000 
NUM_GAMES_PER_EVAL = 3 
EVAL_SEEDS = [42, 101, 999]

def log_to_csv(score):
    csv_path = os.environ.get("FREEWAY_HISTORY_PATH", "history_backup.csv")
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
    try:
        csv_path = os.environ.get("FREEWAY_HISTORY_PATH", None)
        if csv_path:
            results_dir = os.path.dirname(csv_path)
            save_dir = os.path.join(results_dir, "interesting_agents")
        else:
            save_dir = "interesting_agents_backup"
            
        os.makedirs(save_dir, exist_ok=True)
        
        score_int = int(score)
        prefix = f"agent_{score_int}_pts_"
        existing_agents = [f for f in os.listdir(save_dir) if f.startswith(prefix)]
        if len(existing_agents) >= 2: return

        filename = f"{prefix}{int(time.time()*1000)}.py"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code_string)
            
    except Exception as e:
        print(f"Agent save error: {e}")

def clean_llm_code(code_string: str) -> str:
    pattern = r"```(?:python)?\s*(.*?)```"
    match = re.search(pattern, code_string, re.DOTALL)
    if match: return match.group(1).strip()
    return code_string.strip()

def run_custom_simulation(action_function, specific_seed=None, visualization=False):
    render_mode = "human" if visualization else None
    try:
        raw_env = gym.make(ENV_NAME, obs_type="ram", render_mode=render_mode)
        env = FreewayEvoWrapper(raw_env)
    except Exception as e:
        print(f"Env create error: {e}")
        return 0.0

    if specific_seed is None:
        current_seed = random.randint(100, 1000000)
    else:
        current_seed = specific_seed
    
    observation, info = env.reset(seed=current_seed)
    
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    try:
        while not (terminated or truncated) and steps < MAX_STEPS_PER_GAME:
            try:
                # Mapped actions: 0 NOOP, 1 UP, 2 DOWN
                action = int(action_function(observation))
            except Exception:
                return -500.0 # Code crash penalty

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if visualization: time.sleep(0.015)
    except Exception as e:
        return -500.0
    finally:
        env.close()

    return total_reward

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
             log_to_csv(-9999.0)
             return EvaluationResult(metrics={'combined_score': -9999.0})
             
        get_action_func = agent_module.get_action
    except Exception:
        log_to_csv(-9999.0) 
        return EvaluationResult(metrics={'combined_score': -9999.0})

    total_score = 0
    for i in range(NUM_GAMES_PER_EVAL):
        score = run_custom_simulation(action_function=get_action_func, visualization=False)
        total_score += score

    avg_score = total_score / NUM_GAMES_PER_EVAL
    log_to_csv(avg_score)

    # Save if score is decent (> 50.0)
    if avg_score > 50.0:
        save_interesting_agent(cleaned_code, avg_score)

    return EvaluationResult(metrics={'combined_score': avg_score})

if __name__ == "__main__":
    try:
        import initial_agent
        print("Testing initial_agent (Validation Run)...")
        TEST_SEED = 42 
        score = run_custom_simulation(initial_agent.get_action, specific_seed=TEST_SEED, visualization=True)
        print(f"Seed Agent Score (Seed {TEST_SEED}): {score}")
    except ImportError:
        print("❌ Error: initial_agent.py not found!")