import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from collections import deque

class SpaceInvadersColumnWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_columns=10, skip=1):
        super().__init__(env)
        self.n_cols = n_columns
        self.skip = skip 
        
        # Standard Atari Dimensions
        self.W = 160.0
        self.H = 210.0
        self.col_width = self.W / self.n_cols
        
        # --- FEATURE DEFINITION ---
        # 1. Columns (30 features): EnemyY, BulletY, PlayerPos per column
        # 2. Globals (2 features): UFO X position, UFO Active (bool)
        self.features_per_col = 3
        self.global_features = 2 
        
        # Total per single frame: 30 + 2 = 32
        self.features_single_frame = (self.n_cols * self.features_per_col) + self.global_features
        
        # Total with Stack (x3): 96 Inputs
        self.n_features = self.features_single_frame * 3
        
        self.observation_space = Box(
            low=0.0, high=1.0, 
            shape=(self.n_features,), 
            dtype=np.float32
        )
        
        self.action_map = {
            0: 0, # NOOP
            1: 1, # FIRE
            2: 2, # RIGHT
            3: 3  # LEFT
        }

        # Buffer for stacking (Memory for FFNN)
        self.frame_stack = deque(maxlen=3)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Clear buffer
        empty_features = np.zeros(self.features_single_frame, dtype=np.float32)
        self.frame_stack.clear()
        
        # Fill buffer with empty/initial frames
        self.frame_stack.append(empty_features)
        self.frame_stack.append(empty_features)
        
        # Generate first real state
        first_features = self._generate_features()
        self.frame_stack.append(first_features)
        
        return self._get_stacked_obs(), info
    
    def step(self, action_idx):
        real_action = self.action_map.get(action_idx, 0)
        
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        # Manual Frame Skip (Temporal consistency)
        for _ in range(self.skip):
            obs, reward, term, trunc, info = self.env.step(real_action)
            total_reward += reward
            terminated = term or terminated
            truncated = trunc or truncated
            if terminated or truncated:
                break

        # Update stack
        current_features = self._generate_features()
        self.frame_stack.append(current_features)

        return self._get_stacked_obs(), total_reward, terminated, truncated, info

    def observation(self, obs):
        return self._get_stacked_obs()

    def _get_stacked_obs(self):
        # Flattens the deque into a single 1D array [Frame1, Frame2, Frame3]
        return np.concatenate(self.frame_stack)

    def _generate_features(self):
        # Retrieve objects from OCAtari
        objects = getattr(self.env, "objects", getattr(self.env.unwrapped, "objects", []))
        
        enemy_y = np.zeros(self.n_cols, dtype=np.float32)
        bullet_y = np.zeros(self.n_cols, dtype=np.float32)
        player_pos = np.zeros(self.n_cols, dtype=np.float32)
        
        # UFO Variables
        ufo_x = 0.0
        ufo_active = 0.0
        
        for obj in objects:
            cat = obj.category.lower()
            
            # --- UFO HANDLING (Satellite) ---
            if "satellite" in cat or "ufo" in cat:
                ufo_active = 1.0
                ufo_x = obj.x / self.W
                continue # UFO is not part of standard columns

            # --- COLUMN HANDLING ---
            # Calculate column index and normalize Y
            c = int(min(obj.x / self.col_width, self.n_cols - 1))
            norm_y = obj.y / self.H 
            
            if "alien" in cat:
                if norm_y > enemy_y[c]:
                    enemy_y[c] = norm_y
            
            elif "bullet" in cat or "bomb" in cat or "missile" in cat:
                if norm_y > bullet_y[c]:
                    bullet_y[c] = norm_y
            
            elif "player" in cat and "score" not in cat:
                player_pos[c] = 1.0

        # Column Vector [30]
        col_features = np.column_stack((enemy_y, bullet_y, player_pos)).flatten()
        
        # Global Vector [2]
        global_features = np.array([ufo_x, ufo_active], dtype=np.float32)
        
        # Concatenate: [30] + [2] = [32]
        return np.concatenate((col_features, global_features))