import numpy as np
import gymnasium as gym

class FreewaySpeedWrapper(gym.ObservationWrapper):
    """
    Freeway RAM Wrapper with Velocity - 22 features.
    """

    CHICKEN_Y_IDX = 14
    COLLISION_STATE_IDX = 16
    CARS_X_START = 108
    N_CARS = 10
    
    MAX_CHICKEN_Y = 177.0
    MIN_CHICKEN_Y = 14.0
    MAX_CAR_X = 159.0

    def __init__(self, env, normalize: bool = True, mirror_last_5: bool = True):
        super().__init__(env)
        self.normalize = normalize
        self.mirror_last_5 = mirror_last_5
        
        self.prev_cars_x = None

        # 1 (Y) + 1 (Coll) + 10 (Pos X) + 10 (Vel X) = 22
        self.num_features = 22
        
        low = np.zeros((self.num_features,), dtype=np.float32)
        high = np.ones((self.num_features,), dtype=np.float32)
        
        if not normalize:
            high = np.array([self.MAX_CHICKEN_Y, 1.0] + [self.MAX_CAR_X]*10 + [10.0]*10, dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_cars_x = None 
        return self.observation(obs), info

    def observation(self, obs):
        obs = np.asarray(obs)
        
        # 1. Base data extraction
        chicken_y = float(obs[self.CHICKEN_Y_IDX])
        collision_state = 1.0 if obs[self.COLLISION_STATE_IDX] > 0 else 0.0
        current_cars_x = obs[self.CARS_X_START:self.CARS_X_START + self.N_CARS].astype(np.float32)

        # 2. Mirroring
        if self.mirror_last_5:
            current_cars_x[5:] = self.MAX_CAR_X - current_cars_x[5:]
            np.clip(current_cars_x, 0.0, self.MAX_CAR_X, out=current_cars_x)

        # 3. Velocity calculation
        if self.prev_cars_x is None:
            velocities = np.zeros(self.N_CARS, dtype=np.float32)
        else:
            velocities = current_cars_x - self.prev_cars_x
            velocities[np.abs(velocities) > 20] = 0.0 
        
        self.prev_cars_x = current_cars_x.copy()

        # 4. Assembly
        feats = np.zeros((self.num_features,), dtype=np.float32)
        feats[0] = chicken_y
        feats[1] = collision_state
        feats[2:12] = current_cars_x
        feats[12:22] = velocities

        # 5. Normalization
        if self.normalize:
            # Inverted Norm: 0.0 = bottom, 1.0 = goal
            feats[0] = (self.MAX_CHICKEN_Y - chicken_y) / (self.MAX_CHICKEN_Y - self.MIN_CHICKEN_Y)
            feats[2:12] /= self.MAX_CAR_X
            feats[12:22] = (feats[12:22] / 2.0)
            
            np.clip(feats, -1.0, 1.0, out=feats)

        return feats


class FreewayEvoWrapper(FreewaySpeedWrapper):
    """
    Extended Wrapper for Evolution Loop.
    Adds:
    1. Reward Shaping (bonus for moving up, penalty for collision).
    2. Anti-Camping (timeout if chicken stays still).
    """
    def __init__(self, env):
        super().__init__(env, normalize=True, mirror_last_5=True)
        self.prev_y = 0.0
        self.stuck_counter = 0
        self.max_stuck_steps = 150  # Max frames static before timeout

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.prev_y = obs[0]
        self.stuck_counter = 0
        return obs, info

    def step(self, action):
        # ALE Action Mapping: 0: NOOP, 1: UP, 2: DOWN (Standard assumption)
        
        obs, native_reward, terminated, truncated, info = self.env.step(action)
        
        # Process observation (extract the 22 features)
        processed_obs = self.observation(obs)
        
        # --- REWARD SHAPING ---
        current_y = processed_obs[0] # obs[0] is normalized Y
        
        custom_reward = 0.0
        
        # 1. Native Reward (Full point on crossing)
        if native_reward > 0:
            custom_reward += 100.0 # Big bonus for success
            self.stuck_counter = 0

        # 2. Progressive Shaping (Incentivize moving up)
        delta_y = (current_y - self.prev_y)
        if delta_y > 0:
            custom_reward += (delta_y * 10.0) 
        
        # 3. Collision Penalty (Detected via sudden drop in Y)
        if delta_y < -0.05: 
            custom_reward -= 1.0

        # 4. Anti-Camping Logic
        if abs(delta_y) < 0.001:
            self.stuck_counter += 1
            custom_reward -= 0.01 # Slight penalty for idleness
        else:
            self.stuck_counter = 0
        
        if self.stuck_counter > self.max_stuck_steps:
            truncated = True # Kill episode if camping
            custom_reward -= 10.0
            
        self.prev_y = current_y
        
        return processed_obs, custom_reward, terminated, truncated, info