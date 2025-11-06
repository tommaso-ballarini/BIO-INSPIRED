# ======================================================================
# 🏛️ FILE: evaluator.py (L'Arena / La Fitness Function)
# ======================================================================
import gymnasium as gym
import ale_py  # <-- Importa la libreria Atari
import numpy as np

# Registra tutti gli ambienti Atari in Gymnasium
gym.register_envs(ale_py)

def run_game_simulation(agent_decision_function):
    """
    Esegue una simulazione completa del gioco usando la funzione-agente fornita.
    """
    try:
        # Crea l'ambiente REALE, specificando "ram" come osservazione
        env = gym.make("ALE/BankHeist-v5", obs_type="ram")
    except Exception as e:
        print(f"Errore creazione ambiente: {e}")
        return 0.0, {}

    # Lo stato ora è un array numpy di 128 bytes (la RAM)
    game_state, info = env.reset()
    terminated = False
    truncated = False
    
    total_reward = 0.0 # Questa è la nostra nuova fitness
    total_time_survived = 0
    
    MAX_STEPS = 1500 # Limite di passi
    
    for step in range(MAX_STEPS):
        # La funzione agente riceve la RAM (128 bytes)
        move = agent_decision_function(game_state)
        
        # Assicurati che la mossa sia un intero Python
        if isinstance(move, np.integer):
            move = int(move)
            
        # L'ambiente ha 18 azioni (0-17)
        # Assicuriamoci che la mossa sia valida
        if not 0 <= move < 18:
            move = 0 # Azione di default (NOOP)
            
        game_state, reward, terminated, truncated, info = env.step(move)
        
        total_reward += reward
        total_time_survived = step

        if terminated or truncated:
            break
            
    env.close()

    # LA NUOVA FORMULA DI FITNESS
    # L'unica metrica affidabile è il punteggio totale.
    fitness_score = total_reward

    metrics = {
        "fitness_score": fitness_score,
        "tempo_sopravvissuto": total_time_survived
        # Le altre metriche (banche, polizia) non sono disponibili
    }
    
    # Restituiamo solo il punteggio, che è ciò che serve al GA
    return fitness_score, metrics