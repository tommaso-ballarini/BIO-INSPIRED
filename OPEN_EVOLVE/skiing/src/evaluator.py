import sys
import os
import importlib.util
import time
import gymnasium as gym
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
MAX_STEPS_PER_GAME = 2000 # Come nel tuo NEAT setup
NUM_GAMES_PER_EVAL = 1    # 1 va bene se deterministico, 3 per robustezza

base_dir = os.path.abspath(os.path.join(current_dir, '..'))
HISTORY_DIR = Path(base_dir) / 'history'
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

def run_custom_simulation(action_function, visualization=False):
    """
    Esegue una simulazione usando il wrapper fedele a NEAT.
    """
    render_mode = "human" if visualization else None
    
    try:
        # Usiamo il wrapper personalizzato
        env = SkiingOCAtariWrapper(render_mode=render_mode)
    except Exception as e:
        print(f"Errore init env: {e}")
        return 0.0

    observation, info = env.reset(seed=42) # Seed fisso per consistenza
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    try:
        while not (terminated or truncated) and steps < MAX_STEPS_PER_GAME:
            
            # Chiamata alla funzione generata dall'LLM
            try:
                action = int(action_function(observation))
            except Exception as e:
                # Se il codice generato crasha, penalità massima
                return -10000.0

            observation, reward, terminated, truncated, info = env.step(action)
            
            # Qui reward è il "custom_reward" del wrapper (Magnetico + Bonus)
            total_reward += reward
            steps += 1
            
            if visualization:
                time.sleep(0.01)

    except Exception as e:
        print(f"Runtime Error: {e}")
        return -10000.0
    finally:
        env.close()

    return total_reward

def evaluate(code_string: str) -> EvaluationResult:
    """Valuta il codice Python fornito dall'LLM."""
    
    # 1. Compilazione dinamica
    try:
        # Crea un modulo temporaneo
        spec = importlib.util.spec_from_loader("agent_module", loader=None)
        agent_module = importlib.util.module_from_spec(spec)
        exec(code_string, agent_module.__dict__)
        
        if not hasattr(agent_module, 'get_action'):
             return EvaluationResult(score=-9999, feedback="Function 'get_action' not found.")
             
        get_action_func = agent_module.get_action

    except Exception as e:
        return EvaluationResult(score=-9999, feedback=f"Syntax Error: {str(e)}")

    # 2. Esecuzione Simulazione
    total_score = 0
    
    for _ in range(NUM_GAMES_PER_EVAL):
        score = run_custom_simulation(get_action_func, visualization=False)
        total_score += score

    avg_score = total_score / NUM_GAMES_PER_EVAL
    
    # Feedback semplice per l'LLM basato sullo score
    feedback = f"Score achieved: {avg_score:.2f}. "
    if avg_score < -100:
        feedback += "Try to follow the Target Delta X (obs[3]) more closely."
    elif avg_score > 500:
        feedback += "Good job hitting gates! Now optimize for speed."

    return EvaluationResult(score=avg_score, feedback=feedback)

if __name__ == "__main__":
    # Test rapido se eseguito direttamente
    import initial_agent
    print("Testing initial agent...")
    score = run_custom_simulation(initial_agent.get_action, visualization=True)
    print(f"Initial Agent Score: {score}")