# FILE: core/evaluator.py
import gymnasium as gym
import ale_py
import numpy as np

# Registra tutti gli ambienti Atari in Gymnasium
gym.register_envs(ale_py)


def run_game_simulation(agent_decision_function, env_name, max_steps=1500, obs_type="ram", render=False):
    """
    Esegue una simulazione completa del gioco usando la funzione-agente fornita.
    """
    # crea l'ambiente una sola volta
    try:
        if render:
            # quando vogliamo visualizzare
            if obs_type:
                env = gym.make(env_name, obs_type=obs_type, render_mode="human")
            else:
                env = gym.make(env_name, render_mode="human")
        else:
            if obs_type:
                env = gym.make(env_name, obs_type=obs_type)
            else:
                env = gym.make(env_name)
    except Exception as e:
        print(f"Errore creazione ambiente '{env_name}': {e}")
        return 0.0, {}

    # reset obbligatorio
    game_state, info = env.reset()

    total_reward = 0.0
    num_actions = env.action_space.n
    steps_done = 0

    for step in range(max_steps):
        move = agent_decision_function(game_state)

        # cast e controllo azione
        if isinstance(move, np.integer):
            move = int(move)
        if not 0 <= move < num_actions:
            move = 0

        game_state, reward, terminated, truncated, info = env.step(move)

        total_reward += reward
        steps_done = step + 1  # abbiamo fatto questo step

        if render:
            env.render()

        if terminated or truncated:
            break

    env.close()

    metrics = {
        "fitness_score": total_reward,
        "tempo_sopravvissuto": steps_done,
    }
    return total_reward, metrics