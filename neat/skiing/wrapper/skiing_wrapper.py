import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SkiingCustomWrapper(gym.Wrapper):
    """
    Wrapper personalizzato per l'ambiente ALE/Skiing-v5.
    Gestisce penalità bordi, costi sterzo e bonus movimento laterale.
    """
    
    # Mappatura: 0 -> NOOP, 1 -> RIGHT, 2 -> LEFT
    ATARI_ACTION_MAP = [0, 3, 2] 

    def __init__(self, env, enable_steering_cost=True, min_change_ratio=0.05, steering_cost_per_step=1.0, 
                 edge_penalty_multiplier=5.0, edge_threshold=30):
        super().__init__(env)
        
        # Ridefinisci lo spazio di azione a 3 (NOOP, RIGHT, LEFT)
        self.action_space = spaces.Discrete(3) 
        
        # Inizializzazione variabili di stato
        self.last_action = -1
        self.actions_changed = 0
        self.steps = 0
        self.steering_cost = 0.0
        self.edge_cost = 0.0

        # Variabili per il movimento X (per il bonus in run_ski_neat.py)
        self.prev_x_position = None
        self.total_x_movement = 0.0
        
        # Parametri
        self.enable_steering_cost = enable_steering_cost
        self.min_change_ratio = min_change_ratio
        self.steering_cost_per_step = steering_cost_per_step
        self.edge_penalty_multiplier = edge_penalty_multiplier
        self.edge_threshold = edge_threshold


    def step(self, action):
        """Esegue uno step e calcola penalità/bonus in modo sicuro."""
        atari_action = self.ATARI_ACTION_MAP[action]
        
        # 1. Esegue lo step standard dell'ambiente
        observation, reward, terminated, truncated, info = self.env.step(atari_action)
        self.steps += 1

        # 2. LOGICA RAM BLINDATA (Anti-Crash)
        # Usiamo un try-except generale per ignorare errori di lettura RAM durante il rendering
        try:
            # Ottieni la RAM direttamente dall'emulatore (più sicuro di observation)
            ram_state = self.unwrapped.ale.getRAM()
            
            # Esegui la logica SOLO se siamo oltre il primo step e la RAM è valida
            if self.steps > 1 and hasattr(ram_state, '__len__') and len(ram_state) == 128:
                
                # --- A. Lettura Posizione X (Indice 78) ---
                x_position = ram_state[78]
                
                # --- B. Calcolo Movimento X (Bonus) ---
                if self.prev_x_position is not None:
                    # Calcola spostamento assoluto rispetto al frame precedente
                    movement = abs(int(x_position) - int(self.prev_x_position))
                    self.total_x_movement += movement
                
                # Aggiorna posizione per il prossimo frame
                self.prev_x_position = x_position

                # --- C. Penalità Bordi ---
                if x_position < self.edge_threshold or x_position > (255 - self.edge_threshold):
                    self.edge_cost += self.edge_penalty_multiplier

                # --- D. Penalità Sterzo (Se abilitata) ---
                if self.enable_steering_cost:
                    if action == 1 or action == 2: # 1=RIGHT, 2=LEFT
                        self.steering_cost += self.steering_cost_per_step
                
                # --- E. Tracking Cambio Azione (Stabilità) ---
                if action != self.last_action:
                    self.actions_changed += 1

        except Exception:
            # Se la RAM non è leggibile (es. primo frame renderizzato), ignoriamo lo step
            # questo evita il crash "list index out of range"
            pass
        
        self.last_action = action 
        
        return observation, reward, terminated, truncated, info


    def reset(self, **kwargs):
        """Resetta l'ambiente e tutte le variabili di stato."""
        self.last_action = -1
        self.actions_changed = 0
        self.steps = 0
        self.steering_cost = 0.0
        self.edge_cost = 0.0

        # Reset variabili movimento
        self.prev_x_position = None
        self.total_x_movement = 0.0
        
        return self.env.reset(**kwargs)
    
    
    def get_stability_penalty(self):
        """Calcola la penalità per l'eccessiva passività."""
        stability_penalty = 0.0
        
        if self.steps > 1:
            change_ratio = self.actions_changed / (self.steps - 1) 
            
            if change_ratio < self.min_change_ratio:
                BASE_PENALTY = 50000.0
                stability_penalty = BASE_PENALTY * (1.0 - (change_ratio / self.min_change_ratio))
                stability_penalty = max(100.0, stability_penalty)

        return stability_penalty