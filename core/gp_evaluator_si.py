import gymnasium as gym
import numpy as np
from core.wrappers import SpaceInvadersOCAtariWrapper

# --- PARAMETRI FITNESS ---
SCORE_FACTOR = 10.0       # Punti per ogni 10 score del gioco
SURVIVAL_BONUS = 0.05     # Bonus per ogni frame vivo (ridotto per non premiare chi scappa e basta)
DEATH_PENALTY = 50.0      # Penalità se muori subito

# --- NUOVO: Penalità Anti-Camper ---
IDLE_PENALTY = 1.0        # Punti persi per ogni frame in cui si sta fermi (dopo la tolleranza)
MAX_IDLE_FRAMES = 30      # Massimo frame concessi nello stesso punto (circa 1 secondo)

def run_gp_si_simulation(agent_func, render=False):
    try:
        # Difficulty=1 rende i nemici più aggressivi o veloci, disincentivando la staticità
        env = gym.make('SpaceInvaders-v4', obs_type='ram', render_mode='human' if render else None, difficulty=1)
    except:
        env = gym.make('ALE/SpaceInvaders-v5', obs_type='ram', render_mode='human' if render else None, difficulty=1)

    translator = SpaceInvadersOCAtariWrapper(env)
    observation, _ = env.reset()
    
    total_score = 0.0
    steps = 0
    MAX_STEPS = 2500 # Aumentiamo un po' la durata massima
    
    # Variabili per Anti-Camping
    last_x = 0.5 # Posizione iniziale presunta (centro)
    idle_counter = 0
    idle_penalty_total = 0.0

    for _ in range(MAX_STEPS):
        agent_view = translator.observation(observation)
        
        # Tracking Posizione (agent_view[0] è player_x nel nostro wrapper)
        curr_x = agent_view[0]
        
        # Logica Decisionale
        try:
            val = agent_func(*agent_view)
            if val < -0.5: action = 3   # LEFT
            elif val > 0.5: action = 2  # RIGHT
            else: action = 1            # FIRE
        except:
            action = 1 

        observation, reward, terminated, truncated, info = translator.step(action)
        
        # --- CALCOLO PENALITA' IDLE ---
        # Se la X non cambia significativamente
        if abs(curr_x - last_x) < 0.001:
            idle_counter += 1
            if idle_counter > MAX_IDLE_FRAMES:
                # Penalità progressiva: più stai fermo, più fa male
                idle_penalty_total += IDLE_PENALTY 
        else:
            idle_counter = 0 # Reset se ti muovi
            
        last_x = curr_x
        total_score += reward
        steps += 1
        
        if terminated or truncated:
            break

    env.close()
    
    # Formula Fitness Aggiornata
    # Score + Sopravvivenza - Pigrizia
    fitness = (total_score * SCORE_FACTOR) + (steps * SURVIVAL_BONUS) - idle_penalty_total
    
    if steps < 100: fitness -= DEATH_PENALTY 
    
    return max(0.0, fitness),