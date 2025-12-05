import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

class SpaceInvadersGridWrapper(gym.ObservationWrapper):
    def __init__(self, env, grid_shape=(10, 10), skip=4):
        super().__init__(env)
        self.grid_h, self.grid_w = grid_shape
        self.skip = skip # Quanti frame ripetere l'azione (4 è standard Atari)
        
        self.W = 160.0
        self.H = 210.0
        
        self.n_features = self.grid_h * self.grid_w
        self.observation_space = Box(
            low=-2.0, high=2.0, 
            shape=(self.n_features,), 
            dtype=np.float32
        )
        
        # Action Map
        self.action_map = {
            0: 0, # NOOP
            1: 1, # FIRE
            2: 2, # RIGHT
            3: 3  # LEFT
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._get_obs(), info
        
    def step(self, action_idx):
        real_action = self.action_map.get(action_idx, 0)
        
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        # --- DETERMINISTIC FRAME SKIP ---
        # Ripetiamo l'azione N volte. 
        # In NoFrameskip-v4, questo emula il comportamento standard ma SENZA randomness.
        for _ in range(self.skip):
            obs, reward, term, trunc, info = self.env.step(real_action)
            total_reward += reward
            terminated = term or terminated
            truncated = trunc or truncated
            
            if terminated or truncated:
                break
        # -------------------------------

        return self._get_obs(), total_reward, terminated, truncated, info

    def observation(self, obs):
        return self._get_obs()

    def _get_obs(self):
        # ... (Il codice della griglia rimane identico a prima) ...
        objects = getattr(self.env, "objects", getattr(self.env.unwrapped, "objects", []))
        grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        cell_w = self.W / self.grid_w
        cell_h = self.H / self.grid_h

        for obj in objects:
            c = int(min(obj.x / cell_w, self.grid_w - 1))
            r = int(min(obj.y / cell_h, self.grid_h - 1))
            cat = obj.category.lower()
            
            if "player" in cat and "score" not in cat:
                grid[r, c] = 2.0
            elif "bullet" in cat:
                grid[r, c] = -1.0
            elif ("alien" in cat or "enemy" in cat) and grid[r, c] != -1.0:
                grid[r, c] = 1.0
            elif "shield" in cat and grid[r, c] == 0.0:
                grid[r, c] = 0.5

        return grid.flatten()