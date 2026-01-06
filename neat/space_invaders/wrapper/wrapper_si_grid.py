import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from collections import deque

class SpaceInvadersGridWrapper(gym.ObservationWrapper):
    def __init__(self, env, grid_shape=(20, 20), skip=4):
        super().__init__(env)
        self.h, self.w = grid_shape
        self.skip = skip 
        
        self.W = 160.0
        self.H = 210.0
        
        # --- MODIFICA 1: Stacking (Input x 2) ---
        # 20 * 20 * 2 frame = 800 input
        self.n_features = self.h * self.w * 2 
        
        self.observation_space = Box(
            low=-1.0, high=1.0, 
            shape=(self.n_features,), 
            dtype=np.float32
        )
        
        # MANTENUTO: 4 Output come richiesto
        self.action_map = {
            0: 0, # NOOP
            1: 1, # FIRE
            2: 2, # RIGHT
            3: 3  # LEFT
        }

        # Valori della griglia (Tabella Report)
        self.VALUES = {
            "player": 1.0,
            "mystery": 0.75,
            "alien": 0.5,
            "shield": 0.25,
            "bullet": -1.0, # Proiettili (sia amici che nemici considerati pericoli/tracce)
            "bomb": -1.0
        }

        # Buffer per la memoria a breve termine
        self.frame_stack = deque(maxlen=2)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Riempie il buffer con griglie vuote all'inizio
        empty_grid = np.zeros((self.h, self.w), dtype=np.float32)
        self.frame_stack.clear()
        self.frame_stack.append(empty_grid)
        self.frame_stack.append(empty_grid)
        
        # Genera la prima griglia reale
        first_grid = self._generate_grid()
        self.frame_stack.append(first_grid)
        
        return self._get_stacked_obs(), info
        
    def step(self, action_idx):
        real_action = self.action_map.get(action_idx, 0)
        
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        # Frame Skip Deterministico
        for _ in range(self.skip):
            obs, reward, term, trunc, info = self.env.step(real_action)
            total_reward += reward
            terminated = term or terminated
            truncated = trunc or truncated
            
            if terminated or truncated:
                break

        # Aggiorna lo stack con la nuova griglia
        current_grid = self._generate_grid()
        self.frame_stack.append(current_grid)

        return self._get_stacked_obs(), total_reward, terminated, truncated, info

    def observation(self, obs):
        return self._get_stacked_obs()

    def _get_stacked_obs(self):
        # Concatena le griglie nel buffer in un unico vettore lungo
        return np.concatenate([g.flatten() for g in self.frame_stack])

    def _generate_grid(self):
        # Logica di creazione della singola griglia 20x20
        objects = getattr(self.env, "objects", getattr(self.env.unwrapped, "objects", []))
        grid = np.zeros((self.h, self.w), dtype=np.float32)
        
        cell_w = self.W / self.w
        cell_h = self.H / self.h

        for obj in objects:
            cat = obj.category.lower()
            
            # Filtro Esplosioni (non sono ostacoli)
            if "explosion" in cat:
                continue

            c = int(min(obj.x / cell_w, self.w - 1))
            r = int(min(obj.y / cell_h, self.h - 1))
            
            val = 0.0
            if "shield" in cat: val = self.VALUES["shield"]
            elif "alien" in cat: val = self.VALUES["alien"] # Include "enemy"
            elif "mystery" in cat: val = self.VALUES["mystery"]
            elif "player" in cat and "score" not in cat: val = self.VALUES["player"]
            elif "bullet" in cat or "missile" in cat: val = self.VALUES["bullet"]
            elif "bomb" in cat: val = self.VALUES["bomb"]
            
            # --- MODIFICA 3: Priorità ---
            # Se la cella è già occupata, sovrascriviamo SOLO se il nuovo oggetto 
            # ha un valore assoluto maggiore (es. Proiettile > Scudo)
            if abs(val) > abs(grid[r, c]):
                grid[r, c] = val

        return grid