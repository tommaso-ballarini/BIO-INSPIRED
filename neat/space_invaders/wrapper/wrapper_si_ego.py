import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

class SpaceInvadersEgocentricWrapper(gym.ObservationWrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
        self.W = 160.0
        self.H = 210.0
        
        # --- DEFINIZIONE FEATURE (Totale: 19) ---
        # [0] Player X Norm (Ricalibrato 0.0-1.0 su area giocabile)
        # [1-5] Sensori Prossimità (S1..S5)
        # [6-10] Delta Sensori (dS1..dS5)
        # [11] Nearest Alien Relative X (Targeting)
        # [12-15] Densità Alieni (4 Quadranti Relativi)
        # [16] Frazione Totale Alieni (Game Progression) <--- NUOVO
        # [17] UFO Relative X
        # [18] UFO Active
        
        self.n_features = 19
        
        self.observation_space = Box(
            low=-1.0, high=1.0, 
            shape=(self.n_features,), 
            dtype=np.float32
        )
        
        self.action_map = {0: 0, 1: 1, 2: 2, 3: 3}
        self.prev_sensors = np.zeros(5, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_sensors = np.zeros(5, dtype=np.float32)
        return self._generate_features(), info
        
    def step(self, action_idx):
        real_action = self.action_map.get(action_idx, 0)
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        for _ in range(self.skip):
            obs, reward, term, trunc, info = self.env.step(real_action)
            total_reward += reward
            terminated = term or terminated
            truncated = trunc or truncated
            if terminated or truncated: break

        return self._generate_features(), total_reward, terminated, truncated, info

    def observation(self, obs):
        return self._generate_features()

    def _generate_features(self):
        objects = getattr(self.env, "objects", getattr(self.env.unwrapped, "objects", []))
        
        player_x = self.W / 2.0
        projectiles = []
        aliens = []
        ufo = None
        
        for obj in objects:
            cat = obj.category.lower()
            if "player" in cat and "score" not in cat:
                player_x = obj.x
            elif "alien" in cat:
                aliens.append(obj)
            elif "satellite" in cat or "ufo" in cat:
                ufo = obj
            elif "bullet" in cat or "missile" in cat or "bomb" in cat:
                projectiles.append(obj)
        
        # --- 1. PLAYER X (Ricalibrata) ---
        # Range giocabile stimato: 30px - 130px (Ampiezza 100px)
        # Mappiamo questo range su 0.0 - 1.0
        norm_x = (player_x - 30.0) / 100.0
        norm_x = np.clip(norm_x, 0.0, 1.0)

        # --- 2. SENSORI (Invariati) ---
        sensor_ranges = [(-50, -30), (-30, -10), (-10, 10), (10, 30), (30, 50)]
        current_sensors = np.zeros(5, dtype=np.float32)
        
        for i, (min_off, max_off) in enumerate(sensor_ranges):
            x_min = player_x + min_off
            x_max = player_x + max_off
            for p in projectiles:
                px_center = p.x + (p.w / 2.0)
                if x_min <= px_center <= x_max:
                    proximity = (p.y / self.H) 
                    if proximity > current_sensors[i]:
                        current_sensors[i] = proximity

        delta_sensors = current_sensors - self.prev_sensors
        self.prev_sensors = current_sensors.copy()
        
        # --- 3. TARGETING (Invariato) ---
        target_alien_rel_x = 0.0
        if aliens:
            lowest_alien = max(aliens, key=lambda a: a.y)
            target_alien_rel_x = (lowest_alien.x - player_x) / (self.W / 2.0)
            target_alien_rel_x = np.clip(target_alien_rel_x, -1.0, 1.0)
        
        # --- 4. DENSITÀ (Invariato) ---
        q_counts = [0, 0, 0, 0]
        total_aliens = len(aliens)
        if total_aliens > 0:
            for a in aliens:
                is_near = a.y > 100
                is_left = a.x < player_x
                if is_near and is_left: q_counts[0] += 1
                elif is_near and not is_left: q_counts[1] += 1
                elif not is_near and is_left: q_counts[2] += 1
                elif not is_near and not is_left: q_counts[3] += 1
            q_densities = np.array(q_counts, dtype=np.float32) / total_aliens
        else:
            q_densities = np.zeros(4, dtype=np.float32)

        # --- 5. FRAZIONE ALIENI (Nuovo) ---
        # Sostituisce il Delta inutile.
        # Indica "A che punto siamo del livello". Più è basso, più sono veloci.
        alien_fraction = total_aliens / 36.0

        # --- 6. UFO (Invariato) ---
        ufo_rel_x = 0.0
        ufo_active = 0.0
        if ufo:
            ufo_active = 1.0
            ufo_rel_x = (ufo.x - player_x) / (self.W / 2.0)
            ufo_rel_x = np.clip(ufo_rel_x, -1.0, 1.0)

        # --- ASSEMBLAGGIO (19 Features) ---
        features = np.concatenate([
            [norm_x],               # [0]
            current_sensors,        # [1-5]
            delta_sensors,          # [6-10]
            [target_alien_rel_x],   # [11]
            q_densities,            # [12-15]
            [alien_fraction],       # [16] EX Delta Density
            [ufo_rel_x],            # [17] EX UFO X
            [ufo_active]            # [18] EX UFO Active
        ])
        
        return features.astype(np.float32)