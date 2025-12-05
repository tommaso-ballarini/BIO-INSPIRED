import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

try:
    from ocatari.core import OCAtari
    OCATARI_AVAILABLE = True
except ImportError:
    OCATARI_AVAILABLE = False

class FreewayOCAtariWrapper(gym.ObservationWrapper):
    def __init__(self, env_name="Freeway-v4"):
        if not OCATARI_AVAILABLE:
            raise ImportError("OCAtari non installato. Installa con pip install ocatari")
        
        # Gestione sia stringa che env già istanziato
        if isinstance(env_name, str):
             self.ocatari_env = OCAtari(env_name, mode="ram", hud=False, render_mode="rgb_array")
             self.ocatari_env.reset()
             super().__init__(self.ocatari_env)
        else:
             super().__init__(env_name)

        self.num_lanes = 10
        # Osservazione: [Chicken Y, Car 1 X, Car 2 X, ..., Car 10 X]
        self.observation_space = Box(low=0.0, high=1.0, shape=(1 + self.num_lanes,), dtype=np.float32)
        self.SCREEN_W = 160.0

    def observation(self, obs):
        # Accesso agli oggetti OCAtari
        objects = getattr(self.env, "objects", getattr(self.env.unwrapped, "objects", []))
        state_vector = np.zeros(1 + self.num_lanes, dtype=np.float32)
        
        chicken = None
        cars = []

        for obj in objects:
            name = obj.category.lower()
            if "chicken" in name: chicken = obj
            elif "car" in name or "enemy" in name: cars.append(obj)
        
        # 1. Chicken Y Position (Normalized 0-1, 1 is Top)
        if chicken:
            y_norm = (175.0 - chicken.y) / (175.0 - 15.0)
            state_vector[0] = np.clip(y_norm, 0.0, 1.0)
        
        # 2. Cars Positions
        cars.sort(key=lambda c: c.y, reverse=True) # Ordina per corsia (dal basso all'alto)
        for i in range(min(len(cars), self.num_lanes)):
            state_vector[1 + i] = cars[i].x / self.SCREEN_W
            
        return state_vector

    def step(self, action):
        obs, reward, truncated, terminated, info = self.env.step(action)
        return self.observation(obs), reward, truncated, terminated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info