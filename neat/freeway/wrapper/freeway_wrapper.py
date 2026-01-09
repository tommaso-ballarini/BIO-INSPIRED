import numpy as np
import gymnasium as gym

class FreewaySpeedWrapper(gym.ObservationWrapper):
    """
    Freeway RAM Wrapper con Velocità - 22 features:
      - 1: chicken Y position
      - 1: collision state
      - 10: car X positions
      - 10: car X velocities (delta X)
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
        
        # Buffer per calcolare la velocità
        self.prev_cars_x = None

        # 1 (Y) + 1 (Coll) + 10 (Pos X) + 10 (Vel X) = 22
        self.num_features = 22
        
        low = np.zeros((self.num_features,), dtype=np.float32)
        # Per la velocità, il range può essere negativo se non specchiato, 
        # ma con il mirror e la normalizzazione lo terremo in [0, 1] o [-1, 1]
        high = np.ones((self.num_features,), dtype=np.float32)
        
        if not normalize:
            high = np.array([self.MAX_CHICKEN_Y, 1.0] + [self.MAX_CAR_X]*10 + [10.0]*10, dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        # Reset del buffer velocità quando ricomincia il gioco
        obs, info = self.env.reset(**kwargs)
        self.prev_cars_x = None 
        return self.observation(obs), info

    def observation(self, obs):
        obs = np.asarray(obs)
        
        # 1. Estrazione dati base
        chicken_y = float(obs[self.CHICKEN_Y_IDX])
        collision_state = 1.0 if obs[self.COLLISION_STATE_IDX] > 0 else 0.0
        current_cars_x = obs[self.CARS_X_START:self.CARS_X_START + self.N_CARS].astype(np.float32)

        # 2. Mirroring (rende la direzione consistente)
        if self.mirror_last_5:
            current_cars_x[5:] = self.MAX_CAR_X - current_cars_x[5:]
            np.clip(current_cars_x, 0.0, self.MAX_CAR_X, out=current_cars_x)

        # 3. Calcolo Velocità
        if self.prev_cars_x is None:
            velocities = np.zeros(self.N_CARS, dtype=np.float32)
        else:
            # Calcoliamo lo spostamento. 
            # Poiché Atari usa un sistema ciclico (0->159, poi torna a 0),
            # usiamo il modulo o ignoriamo i salti enormi per evitare picchi di velocità.
            velocities = current_cars_x - self.prev_cars_x
            # Se l'auto è "respawnata", la velocità sembrerà enorme/negativa. La resettiamo a 0 o a una media.
            velocities[np.abs(velocities) > 20] = 0.0 
        
        self.prev_cars_x = current_cars_x.copy()

        # 4. Assemblaggio
        feats = np.zeros((self.num_features,), dtype=np.float32)
        feats[0] = chicken_y
        feats[1] = collision_state
        feats[2:12] = current_cars_x
        feats[12:22] = velocities

        # 5. Normalizzazione
        if self.normalize:
            # Normalizzazione invertita: 0.0 = fondo, 1.0 = traguardo
            feats[0] = (self.MAX_CHICKEN_Y - chicken_y) / (self.MAX_CHICKEN_Y - self.MIN_CHICKEN_Y)
            # Collisione (già 0-1)
            # Posizioni X (0-1)
            feats[2:12] /= self.MAX_CAR_X
            # Velocità: le auto in Freeway hanno velocità fisse tra 0.2 e 1.5 circa.
            # Normalizziamo assumendo una velocità massima di 2.0 pixel/frame
            feats[12:22] = (feats[12:22] / 2.0)
            
            np.clip(feats, -1.0, 1.0, out=feats) # Permettiamo velocità negative se necessario

        return feats