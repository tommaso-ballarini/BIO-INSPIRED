import numpy as np
import gymnasium as gym

class FreewaySpeedWrapper(gym.ObservationWrapper):
    """
    Freeway RAM Wrapper con Velocità - 22 features.
    CODICE ORIGINALE (Non modificato nella logica di estrazione).
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
        
        # 1. Estrazione dati base
        chicken_y = float(obs[self.CHICKEN_Y_IDX])
        collision_state = 1.0 if obs[self.COLLISION_STATE_IDX] > 0 else 0.0
        current_cars_x = obs[self.CARS_X_START:self.CARS_X_START + self.N_CARS].astype(np.float32)

        # 2. Mirroring
        if self.mirror_last_5:
            current_cars_x[5:] = self.MAX_CAR_X - current_cars_x[5:]
            np.clip(current_cars_x, 0.0, self.MAX_CAR_X, out=current_cars_x)

        # 3. Calcolo Velocità
        if self.prev_cars_x is None:
            velocities = np.zeros(self.N_CARS, dtype=np.float32)
        else:
            velocities = current_cars_x - self.prev_cars_x
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
            feats[2:12] /= self.MAX_CAR_X
            feats[12:22] = (feats[12:22] / 2.0)
            
            np.clip(feats, -1.0, 1.0, out=feats)

        return feats


class FreewayEvoWrapper(FreewaySpeedWrapper):
    """
    Wrapper esteso per l'Evolution Loop.
    Aggiunge:
    1. Reward Shaping (piccolo bonus per salire, penalità per collisione).
    2. Anti-Camping (timeout se il pollo sta fermo).
    """
    def __init__(self, env):
        super().__init__(env, normalize=True, mirror_last_5=True)
        self.prev_y = 0.0
        self.stuck_counter = 0
        self.max_stuck_steps = 150  # Se stai fermo per 150 frame, muori

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.prev_y = obs[0]
        self.stuck_counter = 0
        return obs, info

    def step(self, action):
        # Mappatura azioni per l'agente (0: NOOP, 1: UP, 2: DOWN)
        # Freeway ALE accetta: 0=NOOP, 1=FIRE(acceleration), 2=UP, 3=RIGHT... 
        # Mappatura standard ALE/Freeway-v5 (visto che usiamo 'ram'):
        # 0: NOOP, 1: UP, 2: DOWN (Usando get_action_meanings() o la logica NEAT fornita: UP=1, DOWN=2 nel codice NEAT)
        
        # Mapping esplicito per sicurezza:
        real_action = 0
        if action == 1: real_action = 1 # UP (ale_py default usually: 0 NOOP, 1 UP, 2 RIGHT...) 
        # Nota: In ALE "Up" è spesso indice 2 o 1 dipendentemente dalla versione. 
        # Nello script NEAT usavano: meanings.index("UP").
        # Per sicurezza usiamo un mapping standard per Freeway-v5 se non specificato altrimenti:
        # NOOP=0, UP=1, DOWN=2. (Assumendo l'uso del wrapper standard)
        
        obs, native_reward, terminated, truncated, info = self.env.step(action)
        # Nota: Poiché ereditiamo da ObservationWrapper, 'step' non è automaticamente wrappato per 
        # restituire l'osservazione modificata se non chiamiamo observation().
        # Gymnasium ObservationWrapper sovrascrive step per applicare observation() automaticamente? 
        # No, ObservationWrapper modifica solo reset e step(return values).
        
        # Recuperiamo l'osservazione processata (le 22 features)
        processed_obs = self.observation(obs) # obs qui è la RAM raw ritornata da env.step
        
        # --- REWARD SHAPING ---
        current_y = processed_obs[0] # obs[0] è la Y normalizzata (0=start, 1=goal)
        
        custom_reward = 0.0
        
        # 1. Native Reward (Punto pieno quando attraversa)
        if native_reward > 0:
            custom_reward += 100.0 # Grande bonus per il successo reale
            self.stuck_counter = 0

        # 2. Shaping Progressivo (Incoraggia a salire)
        delta_y = (current_y - self.prev_y)
        if delta_y > 0:
            custom_reward += (delta_y * 10.0) # Piccolo incentivo per ogni passo avanti
        
        # 3. Collision Penalty (Implicita perché ti spinge giù, ma aggiungiamo un feedback)
        # Se obs[1] (collision) è attivo e prima non lo era (opzionale), o semplicemente se siamo scesi molto
        if delta_y < -0.05: # Sei stato spinto indietro significativamente
            custom_reward -= 1.0

        # 4. Anti-Camping Logic
        if abs(delta_y) < 0.001:
            self.stuck_counter += 1
            custom_reward -= 0.01 # Leggera penalità per l'ozio
        else:
            self.stuck_counter = 0
        
        if self.stuck_counter > self.max_stuck_steps:
            truncated = True # Uccidi l'episodio se campeggia
            custom_reward -= 10.0
            
        self.prev_y = current_y
        
        return processed_obs, custom_reward, terminated, truncated, info