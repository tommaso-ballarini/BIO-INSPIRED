import gymnasium as gym
import numpy as np
from core.wrappers import FreewayOCAtariWrapper

# --- NUOVI PARAMETRI FITNESS (Modalità "Guida Sicura") ---
# Aumentiamo il reward per compensare il fatto che sarà più difficile attraversare
REWARD_FACTOR = 300.0       

# Penalità drastica: se vieni colpito, perdi tantissimo. 
# Se attraversi (300 pti) ma prendi 6 botte (6*50 = 300), hai fatto 0 punti netti.
# Questo costringe l'agente a evitare le auto.
COLLISION_PENALTY = 20.0    

# Penalità per immobilità leggermente rilassata perché a Difficoltà 1 
# a volte DEVI aspettare che passi un'auto veloce.
IDLE_PENALTY_PER_FRAME = 0.05 
MAX_IDLE_FRAMES = 60        

# Bonus per invogliare a salire (fondamentale all'inizio)
PROGRESS_BONUS = 100.0      

def run_gp_simulation(agent_func, render=False):
    try:
        # --- MODIFICA A: Difficulty=1 ---
        # difficulty=1 rende le auto più veloci e il pattern più difficile
        env = gym.make('Freeway-v4', obs_type='ram', render_mode='human' if render else None, difficulty=1)
    except:
        # Fallback per versioni diverse di ALE
        env = gym.make('ALE/Freeway-v5', obs_type='ram', render_mode='human' if render else None, difficulty=1)

    translator = FreewayOCAtariWrapper(env)
    observation, _ = env.reset()
    
    total_reward = 0.0
    collisions = 0
    idle_penalty_total = 0.0
    max_y_reached = 0.0

    prev_y_ram = observation[14]
    idle_counter = 0
    
    # Aumentiamo leggermente gli step perché a diff 1 il gioco è più lento (più attese)
    MAX_STEPS = 1500 

    for _ in range(MAX_STEPS):
        agent_view = translator.observation(observation)
        
        # Calcolo Max Y
        if agent_view[0] > max_y_reached:
            max_y_reached = agent_view[0]

        # Esecuzione Agente
        try:
            output = agent_func(*agent_view)
            action = 1 if output > 0 else 0 # 1=UP, 0=NOOP
        except Exception:
            action = 0 # Fallback

        observation, reward, terminated, truncated, _ = env.step(action)
        
        curr_y_ram = observation[14]

        # Rilevamento Collisione
        if curr_y_ram < prev_y_ram:
            collisions += 1
            idle_counter = 0
        # Rilevamento Idle
        elif curr_y_ram == prev_y_ram:
            # Non penalizziamo l'idle se siamo "al sicuro" (Y molto bassa) 
            # o se stiamo aspettando un varco, ma puniamo il camperare
            idle_counter += 1
            if idle_counter > MAX_IDLE_FRAMES:
                idle_penalty_total += IDLE_PENALTY_PER_FRAME
        else:
            idle_counter = 0

        prev_y_ram = curr_y_ram
        total_reward += reward

        if terminated or truncated:
            break

    env.close()
    
    # Formula Fitness Aggiornata
    fitness = (total_reward * REWARD_FACTOR) + \
              (max_y_reached * PROGRESS_BONUS) - \
              (collisions * COLLISION_PENALTY) - \
              idle_penalty_total
              
    return fitness,