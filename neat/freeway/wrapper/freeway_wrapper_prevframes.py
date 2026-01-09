import numpy as np
import gymnasium as gym

class FreewayRAM11Wrapper(gym.ObservationWrapper):
    """
    Wrapper a 11 feature ottimizzato per Frame Stacking:
    - 1: Chicken Y (Normalizzata 0.0 fondo -> 1.0 traguardo)
    - 10: Auto X (Normalizzate 0.0 entrata -> 1.0 uscita)
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

        # Output: 11 float32
        low = np.zeros((11,), dtype=np.float32)
        high = np.ones((11,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        obs = np.asarray(obs)
        
        # 1. Posizione Pollo
        chicken_y_raw = float(obs[self.CHICKEN_Y_IDX])
        
        # 2. Posizioni Auto
        cars_x = obs[self.CARS_X_START:self.CARS_X_START + self.N_CARS].astype(np.float32)

        # Mirroring: uniforma la direzione di tutte le auto (0=entrata, 159=uscita)
        if self.mirror_last_5:
            # Le corsie 5-9 nell'originale vanno da destra a sinistra
            cars_x[5:] = self.MAX_CAR_X - cars_x[5:]
            np.clip(cars_x, 0.0, self.MAX_CAR_X, out=cars_x)

        # Creazione array feature
        feats = np.empty((11,), dtype=np.float32)
        
        # Inversione Y: vogliamo che 0 sia l'inizio (177) e 1 sia il traguardo (14)
        feats[0] = chicken_y_raw
        feats[1:] = cars_x

        if self.normalize:
            # Normalizzazione Y: (Fondo - Corrente) / (Fondo - Traguardo)
            # Se siamo a 177 -> (177-177)/163 = 0.0
            # Se siamo a 14  -> (177-14)/163  = 1.0
            feats[0] = (self.MAX_CHICKEN_Y - feats[0]) / (self.MAX_CHICKEN_Y - self.MIN_CHICKEN_Y)
            
            # Normalizzazione X
            feats[1:] = feats[1:] / self.MAX_CAR_X
            
            np.clip(feats, 0.0, 1.0, out=feats)

        return feats