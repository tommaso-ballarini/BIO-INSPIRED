# FILE: core/evaluator.py
import gymnasium as gym
import ale_py
import numpy as np

# Registra tutti gli ambienti Atari in Gymnasium
gym.register_envs(ale_py)

def run_game_simulation(agent_decision_function, env_name, max_steps=1500, obs_type="ram"):
    """
    Esegue una simulazione completa del gioco usando la funzione-agente fornita.
    
    Args:
        agent_decision_function: La funzione (es. policy.decide_move)
        env_name (str): Il nome dell'ambiente (es. "ALE/BankHeist-v5", "CartPole-v1")
        max_steps (int): Limite di passi
        obs_type (str): "ram" per Atari, "rgb" per pixel, None per default (es. CartPole)
    """
    try:
        if obs_type:
            env = gym.make(env_name, obs_type=obs_type)
        else:
            env = gym.make(env_name)
    except Exception as e:
        print(f"Errore creazione ambiente '{env_name}': {e}")
        return 0.0, {}

    game_state, info = env.reset()
    terminated = False
    truncated = False
    
    total_reward = 0.0
    total_time_survived = 0
    num_actions = env.action_space.n # Ottieni il numero di azioni DALL'AMBIENTE
    
    for step in range(max_steps):
        move = agent_decision_function(game_state)
        
        if isinstance(move, np.integer):
            move = int(move)
            
        # Controllo di validità generico
        if not 0 <= move < num_actions:
            move = 0 # Azione di default
            
        game_state, reward, terminated, truncated, info = env.step(move)
        
        total_reward += reward
        total_time_survived = step

        if terminated or truncated:
            break
            
    env.close()

    fitness_score = total_reward
    metrics = {
        "fitness_score": fitness_score,
        "tempo_sopravvissuto": total_time_survived
    }
    
    return fitness_score, metrics