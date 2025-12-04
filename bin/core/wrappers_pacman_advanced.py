# core/wrappers_pacman_advanced.py
"""
Wrapper Pac-Man con Feature Engineering Completa secondo il Report.
Implementa: Ray-Casting (Muri), Pie-Slice Radar (Fantasmi/Cibo), State Encoding.
"""

import gymnasium as gym
import numpy as np
import math
from gymnasium.spaces import Box

class PacmanAdvancedWrapper(gym.ObservationWrapper):
    """
    Wrapper che implementa la strategia IBRIDA del documento:
    - Ray-Casting per muri (8 direzioni)
    - Pie-Slice Radar per fantasmi e pellet (8 settori)
    - State encoding (timer vulnerabilità, distanze, ecc.)
    """
    
    def __init__(self, env, num_rays=8, num_sectors=8, max_vision=160.0):
        super().__init__(env)
        
        self.num_rays = num_rays
        self.num_sectors = num_sectors
        self.max_vision = max_vision  # Distanza max normalizzazione
        
        # --- CALCOLO DIMENSIONI INPUT ---
        # 1. Player State (2): x, y normalizzati
        # 2. Ray-Cast Muri (8): distanze normalizzate
        # 3. Pie-Slice Fantasmi (8): densità pericolo per settore
        # 4. Pie-Slice Pellet (8): densità cibo per settore
        # 5. Pie-Slice PowerPills (8): densità power-up per settore
        # 6. Ghost Edibility (4): timer normalizzato per ogni fantasma
        # 7. Nearest Targets (4): dx, dy powerpill + dx, dy pellet più vicini
        # TOTALE: 2 + 8 + 8 + 8 + 8 + 4 + 4 = 42 input
        
        self.n_inputs = 42
        
        self.observation_space = Box(
            low=0.0, high=1.0, 
            shape=(self.n_inputs,), 
            dtype=np.float32
        )
        
        # --- MAPPA LABIRINTO (Hardcoded Pacman Layout) ---
        # In un'implementazione reale, caricheresti questo da un file
        # Per ora uso una griglia semplificata 28x31 (dimensioni standard Pacman)
        self.maze_width = 28
        self.maze_height = 31
        self.wall_map = self._create_wall_map()
        
        # --- MEMORIA PER VELOCITÀ FANTASMI ---
        self.prev_ghost_positions = {}
        
    def _create_wall_map(self):
        """
        Crea una mappa binaria dei muri del labirinto Pacman.
        1 = Muro, 0 = Passaggio.
        
        NOTA: Questo è un PLACEHOLDER. Per Pacman reale dovresti:
        - Estrarre i muri da OCAtari al primo frame
        - Oppure caricare una mappa pre-definita del layout
        """
        # Mappa semplificata (bordi + alcuni muri interni casuali)
        walls = np.zeros((self.maze_height, self.maze_width), dtype=np.uint8)
        
        # Bordi esterni
        walls[0, :] = 1
        walls[-1, :] = 1
        walls[:, 0] = 1
        walls[:, -1] = 1
        
        # Muri interni (esempio semplificato - NON accurato per Pacman reale)
        # In produzione: usa OCAtari o carica un file .txt del layout
        walls[5:10, 10:15] = 1
        walls[15:20, 5:8] = 1
        walls[15:20, 20:23] = 1
        
        return walls
    
    def _pixel_to_grid(self, x, y):
        """Converte coordinate pixel (0-160, 0-210) in coordinate griglia (0-28, 0-31)."""
        grid_x = int((x / 160.0) * self.maze_width)
        grid_y = int((y / 210.0) * self.maze_height)
        return np.clip(grid_x, 0, self.maze_width - 1), np.clip(grid_y, 0, self.maze_height - 1)
    
    def _ray_cast_walls(self, p_x, p_y):
        """
        Esegue ray-casting in 8 direzioni dal player per rilevare muri.
        Restituisce un array [8] di distanze normalizzate (0.0=lontano, 1.0=vicino).
        """
        sensors = np.zeros(self.num_rays, dtype=np.float32)
        
        # Converti posizione pixel in griglia
        grid_x, grid_y = self._pixel_to_grid(p_x, p_y)
        
        # 8 direzioni: N, NE, E, SE, S, SW, W, NW
        angles = np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False)
        
        for i, angle in enumerate(angles):
            dx = math.cos(angle)
            dy = math.sin(angle)
            
            # Ray-marching: cammina lungo il raggio finché non trovi un muro
            distance = 0.0
            max_steps = 50  # Limite per evitare loop infiniti
            
            for step in range(1, max_steps):
                # Posizione corrente del raggio
                check_x = grid_x + dx * step
                check_y = grid_y + dy * step
                
                # Converti in indici griglia
                ix = int(check_x)
                iy = int(check_y)
                
                # Controlla bounds
                if ix < 0 or ix >= self.maze_width or iy < 0 or iy >= self.maze_height:
                    distance = max_steps  # Bordo schermo
                    break
                
                # Controlla muro
                if self.wall_map[iy, ix] == 1:
                    distance = step
                    break
            
            # Normalizzazione inversa: vicino=1.0, lontano=0.0
            sensors[i] = 1.0 - min(distance / max_steps, 1.0)
        
        return sensors
    
    def _pie_slice_radar(self, p_x, p_y, targets, target_type="ghost"):
        """
        Calcola sensori "fetta di torta" per una categoria di oggetti.
        
        Args:
            p_x, p_y: Posizione player (pixel)
            targets: Lista di oggetti OCAtari
            target_type: "ghost", "pellet", o "powerpill"
        
        Returns:
            Array [8] con densità/prossimità per ogni settore
        """
        sectors = np.zeros(self.num_sectors, dtype=np.float32)
        sector_angles = np.linspace(0, 2 * np.pi, self.num_sectors, endpoint=False)
        
        for obj in targets:
            # Vettore dal player all'oggetto
            dx = obj.x - p_x
            dy = obj.y - p_y
            dist = math.sqrt(dx**2 + dy**2)
            
            if dist < 1.0:  # Evita divisione per zero
                continue
            
            # Calcola angolo (atan2 restituisce [-pi, pi])
            angle = math.atan2(dy, dx)
            if angle < 0:
                angle += 2 * np.pi
            
            # Determina il settore
            sector_idx = int((angle / (2 * np.pi)) * self.num_sectors) % self.num_sectors
            
            # Contributo inversamente proporzionale alla distanza
            # Per fantasmi: pericolo alto se vicini
            # Per cibo: attrazione alta se vicini
            contribution = 1.0 / (1.0 + dist / 10.0)  # Normalizzazione soft
            
            # Per fantasmi, considera anche lo stato (vulnerabile vs pericoloso)
            if target_type == "ghost":
                # OCAtari può avere obj.rgb che cambia se il fantasma è blu
                # Fantasmi blu (vulnerabili) hanno tipicamente colori diversi
                # Per semplicità, assumiamo che se obj ha attributo "edible" o colore blu
                is_blue = (obj.rgb[2] > 150 and obj.rgb[0] < 100)  # Blu dominante
                if is_blue:
                    contribution *= -0.5  # Attrazione verso fantasmi vulnerabili
            
            sectors[sector_idx] += contribution
        
        # Normalizzazione finale (clamp a [0, 1])
        return np.clip(sectors, 0.0, 1.0)
    
    def _get_ghost_edibility(self, ghosts):
        """
        Restituisce timer di vulnerabilità per ogni fantasma (max 4).
        Se un fantasma è blu, il valore è >0 (proporzionale al tempo rimanente).
        """
        edibility = np.zeros(4, dtype=np.float32)
        
        for i, ghost in enumerate(ghosts[:4]):
            # OCAtari potrebbe avere un campo 'edible' o possiamo inferire dal colore
            is_blue = (ghost.rgb[2] > 150 and ghost.rgb[0] < 100)
            
            if is_blue:
                # In assenza di timer esplicito, usiamo un valore fisso
                # In un'implementazione reale, dovresti tracciare i frame da quando è diventato blu
                edibility[i] = 0.8  # Placeholder: "è vulnerabile"
        
        return edibility
    
    def _get_nearest_targets(self, p_x, p_y, targets):
        """
        Trova l'oggetto più vicino e restituisce (dx, dy) normalizzato.
        """
        if not targets:
            return np.array([0.0, 0.0], dtype=np.float32)
        
        nearest = min(targets, key=lambda o: (o.x - p_x)**2 + (o.y - p_y)**2)
        dx = (nearest.x - p_x) / self.max_vision
        dy = (nearest.y - p_y) / self.max_vision
        
        return np.array([dx, dy], dtype=np.float32)
    
    def observation(self, obs):
        """
        Trasforma l'osservazione OCAtari in un vettore di feature avanzato.
        """
        # Estrai oggetti da OCAtari
        objects = []
        if hasattr(self.env, "objects"):
            objects = [o for o in self.env.objects if o is not None]
        elif hasattr(self.env.unwrapped, "objects"):
            objects = [o for o in self.env.unwrapped.objects if o is not None]
        
        # Inizializza vettore output
        features = np.zeros(self.n_inputs, dtype=np.float32)
        idx = 0
        
        # --- 1. PLAYER STATE (2) ---
        player = next((obj for obj in objects if "Player" in obj.category or "Pacman" in obj.category), None)
        p_x, p_y = 80.0, 105.0  # Centro schermo di default
        
        if player:
            p_x, p_y = player.x, player.y
            features[idx] = p_x / 160.0
            features[idx + 1] = p_y / 210.0
        idx += 2
        
        # --- 2. RAY-CAST MURI (8) ---
        wall_sensors = self._ray_cast_walls(p_x, p_y)
        features[idx:idx + 8] = wall_sensors
        idx += 8
        
        # --- 3-5. PIE-SLICE RADAR (3 x 8 = 24) ---
        ghosts = [obj for obj in objects if "Enemy" in obj.category or "Ghost" in obj.category]
        pellets = [obj for obj in objects if "Pellet" in obj.category or "Small" in obj.category]
        powerpills = [obj for obj in objects if "Power" in obj.category or "Ball" in obj.category]
        
        ghost_radar = self._pie_slice_radar(p_x, p_y, ghosts, "ghost")
        pellet_radar = self._pie_slice_radar(p_x, p_y, pellets, "pellet")
        power_radar = self._pie_slice_radar(p_x, p_y, powerpills, "powerpill")
        
        features[idx:idx + 8] = ghost_radar
        idx += 8
        features[idx:idx + 8] = pellet_radar
        idx += 8
        features[idx:idx + 8] = power_radar
        idx += 8
        
        # --- 6. GHOST EDIBILITY (4) ---
        edibility = self._get_ghost_edibility(ghosts)
        features[idx:idx + 4] = edibility
        idx += 4
        
        # --- 7. NEAREST TARGETS (4) ---
        nearest_power = self._get_nearest_targets(p_x, p_y, powerpills)
        nearest_pellet = self._get_nearest_targets(p_x, p_y, pellets)
        
        features[idx:idx + 2] = nearest_power
        idx += 2
        features[idx:idx + 2] = nearest_pellet
        idx += 2
        
        assert idx == self.n_inputs, f"Feature mismatch: {idx} != {self.n_inputs}"
        
        return features
    
    def reset(self, **kwargs):
        self.prev_ghost_positions.clear()
        return super().reset(**kwargs)