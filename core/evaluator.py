# FILE: core/evaluator.py
import gymnasium as gym
import ale_py
import numpy as np

# Registra tutti gli ambienti Atari in Gymnasium
gym.register_envs(ale_py)


def run_game_simulation(agent_decision_function, env_name, max_steps=1500, obs_type="ram", render=False, frameskip=None, 
                        repeat_action_probability=None,
                        mode=None, 
                        difficulty=None):
    """
    Esegue una simulazione completa del gioco usando la funzione-agente fornita.
    """
    
    # 1. Costruisci il dizionario degli argomenti dinamici
    make_kwargs = {}
    if obs_type:
        make_kwargs['obs_type'] = obs_type
    if frameskip is not None:
        make_kwargs['frameskip'] = frameskip
    if repeat_action_probability is not None:
        make_kwargs['repeat_action_probability'] = repeat_action_probability
    if mode is not None:
        make_kwargs['mode'] = mode
    if difficulty is not None:
        make_kwargs['difficulty'] = difficulty
    # crea l'ambiente una sola volta
    try:
        if render:
            make_kwargs['render_mode'] = "human"
            env = gym.make(env_name, **make_kwargs) # Usa make_kwargs
        else:
            env = gym.make(env_name, **make_kwargs) # Usa make_kwargs
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