# FILE: core/evaluator.py
import gymnasium as gym
import ale_py
import numpy as np
from core.env_factory import make_evolution_env # <--- IMPORT FONDAMENTALE

# Registra tutti gli ambienti Atari in Gymnasium
gym.register_envs(ale_py)

def run_game_simulation(agent_decision_function, env_name, max_steps=1500, obs_type="ram", render=False, frameskip=None, 
                        repeat_action_probability=None,
                        mode=None, 
                        difficulty=None):
    """
    Esegue una simulazione completa del gioco.
    Usa make_evolution_env per garantire che wrapper e OCAtari siano configurati.
    """
    
    # Usa la factory per creare l'ambiente configurato correttamente
    try:
        render_mode = "human" if render else None
        env = make_evolution_env(env_name, render_mode=render_mode)
    except Exception as e:
        print(f"Errore creazione ambiente '{env_name}': {e}")
        return 0.0, {}

    # Reset iniziale
    try:
        game_state, info = env.reset()
    except Exception as e:
        print(f"Errore durante il reset dell'ambiente: {e}")
        env.close()
        return 0.0, {}

    total_reward = 0.0
    
    # Gestione dinamica dello spazio azioni
    if hasattr(env.action_space, 'n'):
        num_actions = env.action_space.n
    else:
        num_actions = 4 # Fallback sicuro per SpaceInvaders wrapper

    steps_done = 0

    for step in range(max_steps):
        # L'agente riceve il game_state elaborato dal wrapper (es. gli 8 float di SpaceInvaders)
        move = agent_decision_function(game_state)

        # Validazione azione
        if isinstance(move, np.integer):
            move = int(move)
        # Se l'azione è fuori range, usa NOOP (0)
        if not 0 <= move < num_actions:
            move = 0

        try:
            game_state, reward, terminated, truncated, info = env.step(move)
        except Exception as step_error:
            print(f"Errore durante env.step: {step_error}")
            break

        total_reward += reward
        steps_done = step + 1

        if render:
            try:
                env.render()
            except:
                pass

        if terminated or truncated:
            break

    env.close()

    metrics = {
        "fitness_score": total_reward,
        "tempo_sopravvissuto": steps_done,
    }
    return total_reward, metrics