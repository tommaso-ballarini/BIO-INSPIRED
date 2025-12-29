import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

try:
    from ocatari.core import OCAtari
    from ocatari.ram.skiing import Player, Flag, Tree
    OCATARI_AVAILABLE = True
except ImportError:
    OCATARI_AVAILABLE = False

class SkiingOCAtariWrapper(gym.Wrapper):
    """
    SkiingOCAtariWrapper - Fedele all'implementazione NEAT originale.
    Include: Magnet Reward, Gate Bonus, Collision Penalty, Anti-Camping.
    """
    
    def __init__(self, render_mode=None):
        if not OCATARI_AVAILABLE:
            raise ImportError("OCAtari non installato. Installa con pip install ocatari[all]")
            
        self.env = OCAtari("ALE/Skiing-v5", mode="ram", hud=False, render_mode=render_mode)
        super().__init__(self.env)
        
        self.input_shape = 9
        # Definiamo lo spazio per compatibilità, anche se l'LLM userà i valori grezzi
        self.observation_space = Box(low=-1.0, high=1.0, shape=(self.input_shape,), dtype=np.float32)
        
        self.prev_gates = 32
        self.prev_x = 0
        self.stuck_counter = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_gates = 32
        self.stuck_counter = 0
        
        objects = self.env.objects
        player = [o for o in objects if isinstance(o, Player)]
        if player: self.prev_x = player[0].x
        else: self.prev_x = 80
        
        return self._get_smart_obs(), info

    def step(self, action):
        # Eseguiamo lo step nell'ambiente reale
        obs, native_reward, terminated, truncated, info = self.env.step(action)
        
        # Salviamo il reward originale per logging (opzionale)
        info['native_reward'] = native_reward
        
        ram = self.env._env.unwrapped.ale.getRAM()
        current_gates = int(ram[107])
        
        objects = self.env.objects
        player = [o for o in objects if isinstance(o, Player)]
        p = player[0] if player else None

        # --- CALCOLO STATO (Osservazione intelligente) ---
        current_obs = self._get_smart_obs() 
        target_delta_x = current_obs[3]
        target_exists = current_obs[5]
        
        custom_reward = 0.0
        
        # 1. GATE BONUS (+500)
        if current_gates < self.prev_gates:
            custom_reward += 500.0
            self.stuck_counter = 0
            
        # 2. MAGNET REWARD (+1.0 se allineato)
        if target_exists > 0.5:
            alignment_error = abs(target_delta_x)
            if alignment_error < 0.1:
                custom_reward += 1.0
            else:
                custom_reward -= (alignment_error * 0.5)

        # 3. COLLISION DETECTION & PENALTIES
        others = [o for o in objects if isinstance(o, (Tree, Flag)) and not isinstance(o, Player)]
        if p:
            for o in others:
                dist = abs(p.x - o.x) + abs(p.y - o.y) 
                if dist < 5: 
                    custom_reward -= 10.0 # Collisione
                    
            # Penalità Bordi
            if p.x < 10 or p.x > 150:
                custom_reward -= 5.0
            
            # Anti-Camping
            if abs(p.x - self.prev_x) < 0.1:
                self.stuck_counter += 1
                custom_reward -= 1.0 
            else:
                self.stuck_counter = 0
                custom_reward += 0.1
            
            self.prev_x = p.x

        custom_reward -= 0.2 # Penalità tempo leggera
        
        if self.stuck_counter > 100:
            truncated = True
            custom_reward -= 50.0

        self.prev_gates = current_gates
        
        # Ritorniamo custom_reward come reward principale per guidare l'evoluzione
        return current_obs, custom_reward, terminated, truncated, info

    def _get_smart_obs(self):
        objects = self.env.objects
        player = [o for o in objects if isinstance(o, Player)]
        flags = [o for o in objects if isinstance(o, Flag)]
        trees = [o for o in objects if isinstance(o, Tree)]
        
        obs = np.zeros(self.input_shape, dtype=np.float32)
        
        if not player: return obs
        p = player[0]
        
        # 1. PLAYER
        obs[0] = (p.x - 80) / 80.0
        orientation = getattr(p, "orientation", 0)
        obs[1] = (orientation - 128.0) / 128.0
        obs[2] = (p.x - self.prev_x)
        
        # 2. TARGET TUNNEL (Logica geometrica originale)
        upcoming_flags = sorted([f for f in flags if f.y > p.y], key=lambda f: f.y)
        gate_found = False
        target_x = 0
        target_dist = 0
        
        for i in range(len(upcoming_flags)-1):
            f1 = upcoming_flags[i]
            f2 = upcoming_flags[i+1]
            if abs(f1.y - f2.y) < 5 and abs(f1.x - f2.x) < 60:
                gate_found = True
                target_x = (f1.x + f2.x) / 2.0 
                target_dist = f1.y - p.y
                break
        
        if gate_found:
            obs[3] = (target_x - p.x) / 80.0 # Delta X (Cruciale per il magnete)
            obs[4] = target_dist / 200.0     
            obs[5] = 1.0                     
        else:
            obs[3] = 0.0 
            obs[4] = 1.0 
            obs[5] = 0.0 
            
        # 3. THREATS
        threats = trees + flags
        upcoming_threats = [t for t in threats if t.y > p.y]
        if upcoming_threats:
            nearest = min(upcoming_threats, key=lambda t: (t.x-p.x)**2 + (t.y-p.y)**2)
            obs[6] = (nearest.x - p.x) / 80.0
            obs[7] = (nearest.y - p.y) / 200.0
            if isinstance(nearest, Tree): obs[8] = -1.0 
            else: obs[8] = -0.5 
            
        return obs