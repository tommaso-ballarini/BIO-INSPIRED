"""
Wrapper per Atari Skiing - Feature Extraction

Estrae feature interpretabili dalla RAM per controllare lo sciatore:
- Posizione sciatore (X, Y)
- Velocità/movimento
- Ostacoli vicini (alberi, moguls, gates)
- Direzione e distanze
- Timer del gioco
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SkiingFeatureWrapper(gym.Wrapper):
    """
    Wrapper che converte lo stato RAM di Skiing in feature interpretabili.
    
    Feature estratte:
    1. Posizione sciatore (X, Y normalizzate)
    2. Velocità di scroll (quanto veloce si muove)
    3. Ostacoli vicini (distanza e direzione)
    4. Gate/flag positions (per slalom mode)
    5. Collision state (se ha appena colpito qualcosa)
    6. Timer/score corrente
    7. Visual grid semplificata (opzionale)
    """
    
    # Indirizzi RAM chiave per Skiing (da reverse engineering community)
    # Questi sono approssimazioni basate su pattern comuni Atari
    RAM_SKIER_X = 25          # Posizione X dello sciatore (0-255)
    RAM_SKIER_Y = 26          # Offset Y (relativo)
    RAM_SCROLL_SPEED = 12     # Velocità scroll
    RAM_COLLISION = 14        # Flag collisione
    RAM_TIMER_HIGH = 105      # Timer alto byte
    RAM_TIMER_MID = 106       # Timer medio byte
    RAM_TIMER_LOW = 107       # Timer basso byte
    
    # Range per oggetti ostacolo nella RAM
    RAM_OBSTACLES_START = 30
    RAM_OBSTACLES_END = 50
    
    def __init__(self, env, use_visual_grid=True, grid_size=10):
        """
        Args:
            env: Ambiente Skiing (OCAtari o Gymnasium)
            use_visual_grid: Se includere una griglia visiva semplificata
            grid_size: Dimensione della griglia (NxN)
        """
        super().__init__(env)
        
        self.use_visual_grid = use_visual_grid
        self.grid_size = grid_size
        
        # Calcola dimensione observation space
        self.feature_dim = self._calculate_feature_dim()
        
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.feature_dim,),
            dtype=np.float32
        )
        
        # Cache per smooth feature
        self.prev_skier_x = 0
        self.prev_scroll_speed = 0
        
    def _calculate_feature_dim(self):
        """Calcola dimensione totale delle feature."""
        base_features = 0
        
        # 1. Posizione sciatore (2)
        base_features += 2
        
        # 2. Velocità e movimento (3)
        base_features += 3
        
        # 3. Ostacoli vicini - 8 direzioni (8)
        base_features += 8
        
        # 4. Gate/flag detection - più vicino sx e dx (4)
        base_features += 4
        
        # 5. Stato collision e recovery (2)
        base_features += 2
        
        # 6. Timer info (3)
        base_features += 3
        
        # 7. Direzione preferita (2)
        base_features += 2
        
        # TOTALE BASE: 24 feature
        
        # 8. Visual grid (opzionale)
        if self.use_visual_grid:
            base_features += self.grid_size * self.grid_size
        
        return base_features
    
    def reset(self, **kwargs):
        """Reset environment e feature cache."""
        observation, info = self.env.reset(**kwargs)
        self.prev_skier_x = 0
        self.prev_scroll_speed = 0
        
        features = self._extract_features(observation)
        return features, info
    
    def step(self, action):
        """Step con feature extraction."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        features = self._extract_features(observation)
        return features, reward, terminated, truncated, info
    
    def _get_ram_state(self):
        """Ottiene RAM state dall'ambiente."""
        try:
            # OCAtari
            if hasattr(self.env, 'ale'):
                return self.env.ale.getRAM()
            elif hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'ale'):
                return self.env.unwrapped.ale.getRAM()
            # Gymnasium con obs_type="ram"
            elif hasattr(self.env, '_get_ram'):
                return self.env._get_ram()
            else:
                raise AttributeError("Cannot access RAM")
        except Exception as e:
            print(f"⚠️ Error accessing RAM: {e}")
            return np.zeros(128, dtype=np.uint8)
    
    def _extract_features(self, observation):
        """
        Estrae feature interpretabili dalla RAM.
        
        Returns:
            numpy array con tutte le feature normalizzate
        """
        ram = self._get_ram_state()
        features = []
        
        # ============================================
        # 1. POSIZIONE SCIATORE (2 feature)
        # ============================================
        skier_x = ram[self.RAM_SKIER_X]
        skier_y_offset = ram[self.RAM_SKIER_Y]
        
        # Normalizza in [-1, 1]
        norm_x = (skier_x / 127.5) - 1.0
        norm_y = (skier_y_offset / 127.5) - 1.0
        
        features.extend([norm_x, norm_y])
        
        # ============================================
        # 2. VELOCITÀ E MOVIMENTO (3 feature)
        # ============================================
        scroll_speed = ram[self.RAM_SCROLL_SPEED]
        
        # Delta X (movimento orizzontale)
        delta_x = skier_x - self.prev_skier_x
        norm_delta_x = np.clip(delta_x / 10.0, -1.0, 1.0)
        
        # Delta velocità
        delta_speed = scroll_speed - self.prev_scroll_speed
        norm_delta_speed = np.clip(delta_speed / 10.0, -1.0, 1.0)
        
        # Velocità normalizzata
        norm_speed = (scroll_speed / 127.5) - 1.0
        
        features.extend([norm_speed, norm_delta_x, norm_delta_speed])
        
        # Update cache
        self.prev_skier_x = skier_x
        self.prev_scroll_speed = scroll_speed
        
        # ============================================
        # 3. OSTACOLI VICINI - 8 DIREZIONI (8 feature)
        # ============================================
        # Analizza RAM per trovare ostacoli vicini
        obstacle_sensors = self._detect_obstacles_8dir(ram, skier_x, skier_y_offset)
        features.extend(obstacle_sensors)
        
        # ============================================
        # 4. GATE/FLAG DETECTION (4 feature)
        # ============================================
        # Trova gate più vicini a sinistra e destra
        left_gate_dist, left_gate_y = self._find_nearest_gate(ram, skier_x, side='left')
        right_gate_dist, right_gate_y = self._find_nearest_gate(ram, skier_x, side='right')
        
        features.extend([left_gate_dist, left_gate_y, right_gate_dist, right_gate_y])
        
        # ============================================
        # 5. COLLISION STATE (2 feature)
        # ============================================
        collision_flag = ram[self.RAM_COLLISION]
        is_collision = 1.0 if collision_flag > 0 else -1.0
        collision_intensity = (collision_flag / 127.5) - 1.0
        
        features.extend([is_collision, collision_intensity])
        
        # ============================================
        # 6. TIMER INFO (3 feature)
        # ============================================
        timer_high = ram[self.RAM_TIMER_HIGH]
        timer_mid = ram[self.RAM_TIMER_MID]
        timer_low = ram[self.RAM_TIMER_LOW]
        
        # Normalizza timer bytes
        norm_timer_high = (timer_high / 127.5) - 1.0
        norm_timer_mid = (timer_mid / 127.5) - 1.0
        norm_timer_low = (timer_low / 127.5) - 1.0
        
        features.extend([norm_timer_high, norm_timer_mid, norm_timer_low])
        
        # ============================================
        # 7. DIREZIONE PREFERITA (2 feature)
        # ============================================
        # Calcola dove c'è più spazio libero
        left_space = self._calculate_free_space(obstacle_sensors[:4])
        right_space = self._calculate_free_space(obstacle_sensors[4:])
        
        features.extend([left_space, right_space])
        
        # ============================================
        # 8. VISUAL GRID (opzionale)
        # ============================================
        if self.use_visual_grid:
            grid = self._create_visual_grid(ram, skier_x)
            features.extend(grid.flatten())
        
        return np.array(features, dtype=np.float32)
    
    def _detect_obstacles_8dir(self, ram, skier_x, skier_y):
        """
        Rileva ostacoli in 8 direzioni attorno allo sciatore.
        
        Direzioni: N, NE, E, SE, S, SW, W, NW
        
        Returns:
            Lista di 8 valori normalizzati (distanza/intensità ostacolo)
        """
        sensors = []
        
        # Angoli in radianti per 8 direzioni
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        
        for angle in angles:
            # Analizza RAM in quella direzione
            obstacle_strength = self._scan_direction(ram, skier_x, skier_y, angle)
            sensors.append(obstacle_strength)
        
        return sensors
    
    def _scan_direction(self, ram, x, y, angle):
        """
        Scansiona una direzione specifica per ostacoli.
        
        Returns:
            Valore normalizzato [-1, 1]: -1 = libero, +1 = ostacolo vicino
        """
        # Calcola offset direzione
        rad = np.radians(angle)
        dx = int(np.cos(rad) * 20)  # Range scan
        dy = int(np.sin(rad) * 20)
        
        # Cerca nella RAM per oggetti in quella posizione
        # Usa range ostacoli
        obstacle_detected = 0
        
        for addr in range(self.RAM_OBSTACLES_START, self.RAM_OBSTACLES_END):
            obj_value = ram[addr]
            if obj_value > 0:
                # Semplice euristica: se valore RAM alto in quella zona
                # potrebbe indicare presenza ostacolo
                obj_x_approx = addr - self.RAM_OBSTACLES_START
                
                # Distanza approssimativa
                distance = abs(obj_x_approx - (x + dx))
                if distance < 15:  # Soglia vicinanza
                    obstacle_detected = max(obstacle_detected, obj_value)
        
        # Normalizza
        return (obstacle_detected / 127.5) - 1.0 if obstacle_detected > 0 else -1.0
    
    def _find_nearest_gate(self, ram, skier_x, side='left'):
        """
        Trova gate più vicino su un lato.
        
        Returns:
            (distance, y_position) normalizzati
        """
        # Cerca pattern gate nella RAM
        # Gates sono tipicamente coppie di flag
        
        gate_dist = 1.0  # Max distance (nessun gate)
        gate_y = 0.0
        
        # Scansiona RAM per pattern gate
        # (Implementazione semplificata)
        for i in range(40, 60):
            val = ram[i]
            if val > 100:  # Soglia per gate
                # Calcola posizione relativa
                if side == 'left' and i < 50:
                    gate_dist = min(gate_dist, (50 - i) / 50.0)
                    gate_y = (val / 127.5) - 1.0
                elif side == 'right' and i >= 50:
                    gate_dist = min(gate_dist, (i - 50) / 50.0)
                    gate_y = (val / 127.5) - 1.0
        
        return gate_dist, gate_y
    
    def _calculate_free_space(self, sensors):
        """
        Calcola quanto spazio libero c'è in una direzione.
        
        Args:
            sensors: Lista di valori sensori
        
        Returns:
            Valore normalizzato: -1 = tutto bloccato, +1 = tutto libero
        """
        # Media inversa dei sensori
        avg = np.mean(sensors)
        return -avg  # Inverte perché -1 era libero
    
    def _create_visual_grid(self, ram, skier_x):
        """
        Crea una griglia visiva semplificata attorno allo sciatore.
        
        Returns:
            Array NxN con valori normalizzati
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Centro griglia = sciatore
        center = self.grid_size // 2
        
        # Mappa ostacoli sulla griglia
        for addr in range(self.RAM_OBSTACLES_START, self.RAM_OBSTACLES_END):
            obj_val = ram[addr]
            if obj_val > 50:  # Soglia presenza
                # Calcola posizione relativa
                rel_x = (addr - self.RAM_OBSTACLES_START) - skier_x // 5
                
                grid_x = center + (rel_x // 5)
                grid_y = (obj_val % self.grid_size)
                
                # Clamp in bounds
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    grid[grid_y, grid_x] = (obj_val / 127.5) - 1.0
        
        # Sciatore al centro
        grid[center, center] = 1.0
        
        return grid


# ============================================
# UTILITY FUNCTIONS
# ============================================

def print_skiing_features(env, num_steps=50):
    """
    Utility per visualizzare le feature estratte durante il gioco.
    Utile per debugging e comprensione del wrapper.
    """
    print("=" * 80)
    print("SKIING FEATURE EXTRACTION TEST")
    print("=" * 80)
    
    obs, info = env.reset()
    print(f"\nObservation shape: {obs.shape}")
    print(f"Feature names and ranges:")
    print("  [0-1]:   Skier position (x, y)")
    print("  [2-4]:   Speed and movement")
    print("  [5-12]:  Obstacle sensors (8 directions)")
    print("  [13-16]: Gate detection (left/right)")
    print("  [17-18]: Collision state")
    print("  [19-21]: Timer info")
    print("  [22-23]: Free space (left/right)")
    if hasattr(env, 'use_visual_grid') and env.use_visual_grid:
        grid_start = 24
        grid_size = env.grid_size * env.grid_size
        print(f"  [{grid_start}-{grid_start + grid_size - 1}]: Visual grid {env.grid_size}x{env.grid_size}")
    
    print("\n" + "=" * 80)
    print("Running simulation...")
    print("=" * 80)
    
    for step in range(num_steps):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 10 == 0:
            print(f"\nStep {step}:")
            print(f"  Skier X: {obs[0]:.3f}, Y: {obs[1]:.3f}")
            print(f"  Speed: {obs[2]:.3f}")
            print(f"  Collision: {obs[17]:.3f}")
            print(f"  Reward: {reward}")
        
        if terminated or truncated:
            print("\n Episode finished!")
            break
    
    env.close()
    print("\n" + "=" * 80)


if __name__ == "__main__":
    """Test del wrapper"""
    from ocatari.core import OCAtari
    
    print("Testing SkiingFeatureWrapper...")
    
    # Crea ambiente
    base_env = OCAtari("Skiing", mode="ram", render_mode=None, hud=False)
    
    # Applica wrapper
    wrapped_env = SkiingFeatureWrapper(base_env, use_visual_grid=True, grid_size=10)
    
    # Test
    print_skiing_features(wrapped_env, num_steps=100)