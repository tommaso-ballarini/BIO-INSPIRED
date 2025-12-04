# File: wrapper/skiing_wrapper.py

import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SkiingCustomWrapper(gym.Wrapper):
    """
    Wrapper personalizzato per l'ambiente ALE/Skiing-v5.
    
    Gestisce:
    1. Mappatura delle azioni da (0, 1, 2) alle azioni Atari rilevanti.
    2. Tracking e calcolo di penalità personalizzate (Sterzo e Stabilità).
    """
    
    # Mappatura delle 3 azioni del NEAT (0, 1, 2) alle azioni complete di Atari (0-17)
    # 0 -> NOOP (0), 1 -> RIGHT (3), 2 -> LEFT (2)
    ATARI_ACTION_MAP = [0, 3, 2] 

    def __init__(self, env, enable_steering_cost=True, min_change_ratio=0.05, steering_cost_per_step=1.0, edge_penalty_multiplier=5.0, # NUOVO PARAMETRO DI CONTROLLO
                 edge_threshold=30):
        super().__init__(env)
        
        # Ridefinisci lo spazio di azione a 3 (quello che il NEAT vedrà)
        self.action_space = spaces.Discrete(3) 
        
        # Variabili di stato per le penalità
        self.last_action = -1
        self.actions_changed = 0
        self.steps = 0
        self.steering_cost = 0.0
        self.edge_cost = 0.0
        
        # Parametri di penalità (facilmente modificabili)
        self.enable_steering_cost = enable_steering_cost
        self.min_change_ratio = min_change_ratio
        self.steering_cost_per_step = steering_cost_per_step

        self.edge_penalty_multiplier = edge_penalty_multiplier
        self.edge_threshold = edge_threshold


    def step(self, action):
        """Esegue un passo nell'ambiente con l'azione mappata e aggiorna i costi."""
        
        # 1. Mappa l'azione (0, 1, 2) all'azione Atari (0, 3, 2)
        atari_action = self.ATARI_ACTION_MAP[action]
        
        # 2. Esegue lo step nell'ambiente
        observation, reward, terminated, truncated, info = self.env.step(atari_action)
        
        # 3. Aggiorna lo stato per il calcolo delle penalità
        self.steps += 1

        # --- NUOVO: CALCOLO PENALITÀ BORDI ---
        # L'osservazione è lo stato RAM, che è un array di 128 byte
        ram_state = observation # observation è lo stato RAM non normalizzato
        edge_cost_per_step = 0.0

        if len(ram_state) == 128:
            # L'indice 78 (0-based) contiene la posizione X dello sciatore
            x_position = ram_state[78] 
            
            # Se la posizione X è troppo vicina ai bordi (0) o (255)
            if x_position < self.edge_threshold or x_position > (255 - self.edge_threshold):
                # Penalità proporzionale a quanto è lontano dal centro o semplicemente un costo fisso
                
                # Scegliamo un Costo Fisso per Step sui Bordi (perché è più facile)
                edge_cost_per_step = self.edge_penalty_multiplier
                self.edge_cost += edge_cost_per_step # Riutilizzo steering_cost per semplicità
                
                # OPZIONE 2: Penalità Proporzionale (più forte se è proprio all'estremo)
                # distance_from_center = abs(x_position - 128)
                # penalty = (distance_from_center / 128) * self.edge_penalty_multiplier * 5
                # self.steering_cost += penalty
        
        # 4. Tracking del costo di sterzo (penalità progressiva)
        if self.enable_steering_cost:
            # L'azione 0 è NOOP, 1 è RIGHT, 2 è LEFT (nel contesto del wrapper/NEAT output)
            if action == 1 or action == 2:
                self.steering_cost += self.steering_cost_per_step
        
        # 5. Tracking del cambio di azione (per penalità di stabilità)
        if self.steps > 0 and action != self.last_action:
            self.actions_changed += 1
        self.last_action = action
        
        return observation, reward, terminated, truncated, info


    def reset(self, **kwargs):
        """Resetta l'ambiente e le variabili di stato del wrapper."""
        self.last_action = -1
        self.actions_changed = 0
        self.steps = 0
        self.steering_cost = 0.0
        self.edge_cost = 0.0
        
        # Passa il reset all'ambiente sottostante
        return self.env.reset(**kwargs)
    
    
    def get_stability_penalty(self):
        """Calcola la penalità per l'eccessiva passività (mancanza di cambi di direzione)."""
        stability_penalty = 0.0
        if self.steps > 1:
            change_ratio = self.actions_changed / (self.steps - 1) 
            # Se la percentuale di cambi di azione è troppo bassa
            if change_ratio < self.min_change_ratio:
                BASE_PENALTY = 50000.0
                stability_penalty = BASE_PENALTY * (1.0 - (change_ratio / self.min_change_ratio))
                stability_penalty = max(100.0, stability_penalty) # Assicura un minim

        return stability_penalty