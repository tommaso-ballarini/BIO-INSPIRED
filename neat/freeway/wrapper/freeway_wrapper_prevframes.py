import numpy as np
import gymnasium as gym

class FreewayRAM11Wrapper(gym.ObservationWrapper):
    """
    11-feature wrapper optimized for frame stacking:
    - 1: chicken Y (normalized 0.0 bottom -> 1.0 finish)
    - 10: car X (normalized 0.0 entry -> 1.0 exit)
    """

    CHICKEN_Y_IDX = 14
    CARS_X_START = 108
    N_CARS = 10
    
    MIN_CHICKEN_Y = 14.0
    MAX_CHICKEN_Y = 177.0
    MAX_CAR_X = 159.0

    def __init__(self, env, normalize: bool = True, mirror_last_5: bool = True):
        super().__init__(env)
        self.normalize = normalize
        self.mirror_last_5 = mirror_last_5

        low = np.zeros((11,), dtype=np.float32)
        high = np.ones((11,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        obs = np.asarray(obs)
        
        chicken_y_raw = float(obs[self.CHICKEN_Y_IDX])
        
        cars_x = obs[self.CARS_X_START:self.CARS_X_START + self.N_CARS].astype(np.float32)

        # Mirror lanes to make car directions consistent (0=entry, 159=exit)
        if self.mirror_last_5:
            # Lanes 5-9 run right-to-left in the original
            cars_x[5:] = self.MAX_CAR_X - cars_x[5:]
            np.clip(cars_x, 0.0, self.MAX_CAR_X, out=cars_x)

        feats = np.empty((11,), dtype=np.float32)
        
        feats[0] = chicken_y_raw
        feats[1:] = cars_x

        if self.normalize:
            # Inverted Y normalization: (bottom - current) / (bottom - finish)
            feats[0] = (self.MAX_CHICKEN_Y - feats[0]) / (self.MAX_CHICKEN_Y - self.MIN_CHICKEN_Y)
            
            feats[1:] = feats[1:] / self.MAX_CAR_X
            
            np.clip(feats, 0.0, 1.0, out=feats)

        return feats
