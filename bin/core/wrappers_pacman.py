# FILE: core/wrappers_pacman.py
import gymnasium as gym
import numpy as np
import cv2
from gymnasium.spaces import Box

try:
    from ocatari.core import OCAtari
    OCATARI_AVAILABLE = True
except ImportError:
    OCATARI_AVAILABLE = False
    print("⚠️ OCAtari non disponibile. Installare con: pip install ocatari")


class PacmanFeatureWrapper(gym.ObservationWrapper):
    """
    Wrapper ottimizzato per Pacman seguendo le linee guida del documento tecnico.
    
    Estrae feature semantiche dalla RAM tramite OCAtari (REM) e le combina
    con una griglia visiva ridotta per fornire un vettore di input fisso a NEAT.
    
    Feature Vector Structure (totale: 46 + 100 = 146 dimensioni):
    -----------------------------------------------------------
    [0-1]:   Player (x, y) normalizzate [0,1]
    [2-9]:   4 Ghosts relative position (dx, dy) rispetto al player
    [10-17]: 4 Ghosts velocity (vx, vy) - delta posizione tra frame
    [18-19]: Nearest PowerPill (dx, dy) relativo
    [20-21]: Nearest Pellet (dx, dy) relativo  
    [22-25]: Wall Sensors (UP, RIGHT, DOWN, LEFT) binari [0,1]
    [26-29]: 4 Ghosts edibility status [0,1] - CRITICO per invertire policy
    [30]:    Distanza minima da ghost non-edible (normalizzata)
    [31]:    Distanza minima da ghost edible (normalizzata)
    [32]:    Numero pellet rimanenti (normalizzato su 244 max)
    [33]:    Numero vite (normalizzato su 3 max)
    [34]:    Player direction (0-3: UP, RIGHT, DOWN, LEFT)
    [35-42]: Ray sensors 8 direzioni (distanza a primo ostacolo)
    [43]:    Tempo dall'ultima power pill (normalizzato)
    [44]:    Corners cleared flag (0/1)
    [45]:    Stuck detection (movimento minimo negli ultimi N frame)
    [46-145]: Visual grid 10x10 (maze structure)
    """
    
    def __init__(self, env, grid_rows=10, grid_cols=10, max_stuck_frames=50):
        super().__init__(env)
        
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.max_stuck_frames = max_stuck_frames
        
        # Dimensioni feature vector
        self.n_vector_features = 46
        self.n_grid_features = grid_rows * grid_cols
        self.total_inputs = self.n_vector_features + self.n_grid_features
        
        # Observation space per NEAT (Box 1D)
        self.observation_space = Box(
            low=-1.0, high=1.0,
            shape=(self.total_inputs,),
            dtype=np.float32
        )
        
        # Memoria per calcolo velocità fantasmi (4 fantasmi, coordinata x e y)
        self.prev_ghosts = np.zeros((4, 2), dtype=np.float32)
        
        # Memoria per stuck detection
        self.prev_player_pos = np.zeros(2, dtype=np.float32)
        self.stuck_counter = 0
        
        # Timer power pill
        self.frames_since_powerpill = 0
        self.max_powerpill_duration = 200  # Frame circa di durata power pill
        
        # Cache per la griglia visiva (aggiornata meno frequentemente)
        self.visual_grid_cache = np.zeros(self.n_grid_features, dtype=np.float32)
        self.frames_since_grid_update = 0
        self.grid_update_interval = 4  # Aggiorna ogni 4 frame per efficienza
        
        print(f"✅ PacmanFeatureWrapper inizializzato: {self.total_inputs} input totali")
        print(f"   - Vector features: {self.n_vector_features}")
        print(f"   - Visual grid: {self.n_grid_features} ({grid_rows}x{grid_cols})")

    def reset(self, **kwargs):
        """Reset dello stato interno del wrapper"""
        self.prev_ghosts.fill(0.0)
        self.prev_player_pos.fill(0.0)
        self.stuck_counter = 0
        self.frames_since_powerpill = 0
        self.visual_grid_cache.fill(0.0)
        self.frames_since_grid_update = 0
        return super().reset(**kwargs)

    def _extract_objects(self):
        """Estrae gli oggetti da OCAtari in modo sicuro"""
        objects = []
        
        # Prova diversi percorsi per accedere agli oggetti
        if hasattr(self.env, 'objects'):
            objects = [o for o in self.env.objects if o is not None]
        elif hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'objects'):
            objects = [o for o in self.env.unwrapped.objects if o is not None]
        
        return objects

    def _get_wall_sensors(self, rgb_image, p_x, p_y):
        """
        Rileva muri nelle 4 direzioni cardinali usando color thresholding.
        I muri in Pacman sono tipicamente blu (R≈33, G≈33, B>100).
        
        Returns: array [UP, RIGHT, DOWN, LEFT] con valori [0,1]
        """
        sensors = np.zeros(4, dtype=np.float32)
        
        if rgb_image is None or rgb_image.size == 0:
            return sensors
        
        h, w = rgb_image.shape[:2]
        
        # Centro approssimativo dello sprite di Pacman (8x14 pixel circa)
        cx = int(p_x + 4)
        cy = int(p_y + 7)
        
        # Distanza di probing (circa 1.5x dimensione sprite)
        probe_dist = 12
        
        # Coordinate da controllare: [UP, RIGHT, DOWN, LEFT]
        check_coords = [
            (cx, max(0, cy - probe_dist)),          # UP
            (min(w-1, cx + probe_dist), cy),        # RIGHT  
            (cx, min(h-1, cy + probe_dist)),        # DOWN
            (max(0, cx - probe_dist), cy)           # LEFT
        ]
        
        for i, (px, py) in enumerate(check_coords):
            px = int(np.clip(px, 0, w-1))
            py = int(np.clip(py, 0, h-1))
            
            pixel = rgb_image[py, px]
            
            # Rilevamento muro blu: B > 100 e R < 100
            if pixel[2] > 100 and pixel[0] < 100:
                sensors[i] = 1.0
        
        return sensors

    def _get_ray_sensors(self, objects, player_x, player_y, screen_w=160, screen_h=210):
        """
        Ray casting in 8 direzioni per rilevare la distanza al primo ostacolo (muro/ghost).
        Utile per navigazione e evitamento ostacoli.
        
        Direzioni: N, NE, E, SE, S, SW, W, NW
        Returns: array di 8 distanze normalizzate [0,1]
        """
        sensors = np.ones(8, dtype=np.float32)  # Default: nessun ostacolo vicino
        
        # Ottieni fantasmi per collision detection
        ghosts = [o for o in objects if "Enemy" in o.category or "Ghost" in o.category]
        
        # Angoli delle 8 direzioni (in radianti)
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
        max_ray_dist = 50.0  # Distanza massima di rilevamento
        
        for i, angle in enumerate(angles):
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            # Ray marching semplificato
            for dist in range(5, int(max_ray_dist), 5):
                ray_x = player_x + dx * dist
                ray_y = player_y + dy * dist
                
                # Controlla collisione con fantasmi
                for ghost in ghosts:
                    g_dist = np.sqrt((ray_x - ghost.x)**2 + (ray_y - ghost.y)**2)
                    if g_dist < 8:  # Collision radius
                        sensors[i] = dist / max_ray_dist
                        break
                
                # Controlla limiti schermo (muri perimetrali)
                if ray_x < 0 or ray_x > screen_w or ray_y < 0 or ray_y > screen_h:
                    sensors[i] = dist / max_ray_dist
                    break
        
        return sensors

    def _update_visual_grid(self, rgb_image):
        """
        Estrae una griglia visiva ridotta del labirinto.
        Aggiornata meno frequentemente per efficienza (come suggerito dal documento).
        """
        if rgb_image is None or rgb_image.size == 0:
            return self.visual_grid_cache
        
        try:
            # Converti in grayscale
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            
            # Crop dell'area di gioco (rimuove HUD)
            # Per Pacman standard: area gioco circa y=[20:190]
            play_area = gray[20:190, :]
            
            # Resize alla griglia target
            small_grid = cv2.resize(
                play_area,
                (self.grid_cols, self.grid_rows),
                interpolation=cv2.INTER_AREA
            )
            
            # Normalizza e appiattisci
            grid_flat = small_grid.flatten().astype(np.float32) / 255.0
            
            # Edge enhancement opzionale per evidenziare i muri
            # grid_flat = np.where(grid_flat < 0.3, 0.0, grid_flat)  # Threshold muri
            
            self.visual_grid_cache = grid_flat
            
        except Exception as e:
            print(f"⚠️ Errore visual grid: {e}")
        
        return self.visual_grid_cache

    def observation(self, obs):
        """
        Trasforma l'osservazione OCAtari in un vettore di feature strutturato
        seguendo le best practice del documento tecnico.
        """
        # Estrai oggetti da OCAtari
        objects = self._extract_objects()
        
        # Inizializza vettore feature
        vector_features = np.zeros(self.n_vector_features, dtype=np.float32)
        
        # --- ESTRAZIONE PLAYER ---
        player = next((o for o in objects if "Player" in o.category or "Pacman" in o.category), None)
        
        p_x_norm, p_y_norm = 0.0, 0.0
        p_x_pixel, p_y_pixel = 0, 0
        
        if player:
            # p_x_pixel = player.x
            # p_y_pixel = player.y
            p_x_pixel = int(player.x) if hasattr(player, 'x') else 0
            p_y_pixel = int(player.y) if hasattr(player, 'y') else 0
            p_x_norm = p_x_pixel / 160.0
            p_y_norm = p_y_pixel / 210.0
            
            vector_features[0] = p_x_norm
            vector_features[1] = p_y_norm
            
            # Player direction (se disponibile in OCAtari)
            # if hasattr(player, 'orientation'):
            #     # Converti orientamento in indice 0-3
            #     vector_features[34] = float(player.orientation) / 3.0

            # if hasattr(player, 'orientation') and player.orientation is not None:
            #     # Converti orientamento in valore normalizzato 0-1
            #     # OCAtari usa: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
            #     vector_features[34] = float(player.orientation) / 3.0
            # else:
            #     # Valore di default se orientamento non disponibile
            #     vector_features[34] = 0.0

            if hasattr(player, 'orientation') and player.orientation is not None:
                try:
                    vector_features[34] = float(player.orientation) / 3.0
                except (ValueError, TypeError):
                    pass
        


        # --- ESTRAZIONE GHOSTS (Posizione, Velocità, Edibility) ---
        ghosts = [o for o in objects if "Enemy" in o.category or "Ghost" in o.category]
        ghosts.sort(key=lambda x: x.category)  # Ordine stabile per velocità
        
        min_dist_dangerous = 1.0  # Distanza minima da ghost pericoloso
        min_dist_edible = 1.0     # Distanza minima da ghost commestibile
        any_ghost_edible = False
        
        for i in range(min(len(ghosts), 4)):
            ghost = ghosts[i]
            
            if not (hasattr(ghost, 'x') and hasattr(ghost, 'y')):
                continue
            # Coordinate normalizzate
            g_x_norm = ghost.x / 160.0
            g_y_norm = ghost.y / 210.0
            
            # Indici nel vettore
            pos_idx = 2 + (i * 2)      # Slot [2-9]: posizioni relative
            vel_idx = 10 + (i * 2)     # Slot [10-17]: velocità
            edible_idx = 26 + i        # Slot [26-29]: stato commestibile
            
            # A. Posizione Relativa (come da documento: feature fondamentale)
            dx = g_x_norm - p_x_norm
            dy = g_y_norm - p_y_norm
            vector_features[pos_idx] = dx
            vector_features[pos_idx + 1] = dy
            
            # B. Velocità (Delta tra frame)
            vx = g_x_norm - self.prev_ghosts[i][0]
            vy = g_y_norm - self.prev_ghosts[i][1]
            vector_features[vel_idx] = vx * 10.0      # Scala per visibilità
            vector_features[vel_idx + 1] = vy * 10.0
            
            # Aggiorna memoria
            self.prev_ghosts[i][0] = g_x_norm
            self.prev_ghosts[i][1] = g_y_norm
            
            # C. Stato Edibility (CRITICO - vedi documento sezione 4.1)
            is_edible = 0.0

            # if hasattr(ghost, 'rgb'):
            #     # I fantasmi edibili hanno colore blu (approssimativo)
            #     if ghost.rgb[2] > 150:  # Componente blu alta
            #         is_edible = 1.0
            #         any_ghost_edible = True
            #         self.frames_since_powerpill = 0

            if hasattr(ghost, 'rgb') and ghost.rgb is not None:
                try:
                    # I fantasmi edibili sono blu/bianchi
                    if len(ghost.rgb) >= 3 and ghost.rgb[2] > 150:
                        is_edible = 1.0
                        any_ghost_edible = True
                        self.frames_since_powerpill = 0
                except (TypeError, IndexError):
                    pass
            
            vector_features[edible_idx] = is_edible
            
            # D. Tracking distanze minime
            dist = np.sqrt(dx**2 + dy**2)
            if is_edible > 0.5:
                min_dist_edible = min(min_dist_edible, dist)
            else:
                min_dist_dangerous = min(min_dist_dangerous, dist)
        
        vector_features[30] = min_dist_dangerous
        vector_features[31] = min_dist_edible

        # --- NEAREST POWER PILL ---
        power_pills = [o for o in objects if "PowerPill" in o.category or "Ball" in o.category]
        # if player and power_pills:
        #     nearest_pp = min(power_pills, key=lambda o: (o.x - player.x)**2 + (o.y - player.y)**2)
        #     vector_features[18] = (nearest_pp.x / 160.0) - p_x_norm
        #     vector_features[19] = (nearest_pp.y / 210.0) - p_y_norm

        if player and power_pills:
            try:
                nearest_pp = min(power_pills, 
                               key=lambda o: (o.x - player.x)**2 + (o.y - player.y)**2)
                vector_features[18] = (nearest_pp.x / 160.0) - p_x_norm
                vector_features[19] = (nearest_pp.y / 210.0) - p_y_norm
            except (AttributeError, TypeError):
                pass

        # --- NEAREST PELLET ---
        pellets = [o for o in objects if "Pellet" in o.category or "Dot" in o.category]
        # if player and pellets:
        #     nearest_pellet = min(pellets, key=lambda o: (o.x - player.x)**2 + (o.y - player.y)**2)
        #     vector_features[20] = (nearest_pellet.x / 160.0) - p_x_norm
        #     vector_features[21] = (nearest_pellet.y / 210.0) - p_y_norm

        if player and pellets:
            try:
                nearest_pellet = min(pellets,
                                    key=lambda o: (o.x - player.x)**2 + (o.y - player.y)**2)
                vector_features[20] = (nearest_pellet.x / 160.0) - p_x_norm
                vector_features[21] = (nearest_pellet.y / 210.0) - p_y_norm
            except (AttributeError, TypeError):
                pass
        
        # Conteggio pellet rimanenti (normalizzato su max ~244)
        #vector_features[32] = len(pellets) / 244.0
        vector_features[32] = min(len(pellets) / 244.0, 1.0)


        # --- WALL SENSORS (4 direzioni) ---
        rgb_screen = None
        try:
            rgb_screen = self.env.render()
            if isinstance(rgb_screen, list):
                rgb_screen = rgb_screen[0]
        except:
            pass
        
        # if player and rgb_screen is not None:
        #     wall_sensors = self._get_wall_sensors(rgb_screen, p_x_pixel, p_y_pixel)
        #     vector_features[22:26] = wall_sensors

        if player and rgb_screen is not None and rgb_screen.size > 0:
            try:
                wall_sensors = self._get_wall_sensors(rgb_screen, p_x_pixel, p_y_pixel)
                vector_features[22:26] = wall_sensors
            except Exception:
                pass

        # --- RAY SENSORS (8 direzioni) ---
        # ray_sensors = self._get_ray_sensors(objects, p_x_pixel, p_y_pixel)
        # vector_features[35:43] = ray_sensors
        try:
            ray_sensors = self._get_ray_sensors(objects, p_x_pixel, p_y_pixel)
            vector_features[35:43] = ray_sensors
        except Exception:
            pass

        # --- GAME STATE FEATURES ---
        # Vite rimanenti (normalizzato su 3 max)
        # if hasattr(self.env.unwrapped, 'ale'):
        #     lives = self.env.unwrapped.ale.lives()
        #     vector_features[33] = lives / 3.0
        try:
            if hasattr(self.env.unwrapped, 'ale'):
                lives = self.env.unwrapped.ale.lives()
                vector_features[33] = np.clip(lives / 3.0, 0.0, 1.0)
        except Exception:
            vector_features[33] = 1.0

        # Timer power pill (normalizzato)
        self.frames_since_powerpill += 1
        if any_ghost_edible:
            self.frames_since_powerpill = 0
        vector_features[43] = min(self.frames_since_powerpill / self.max_powerpill_duration, 1.0)

        # --- STUCK DETECTION ---
        if player:
            current_pos = np.array([p_x_norm, p_y_norm])
            pos_delta = np.linalg.norm(current_pos - self.prev_player_pos)
            
            if pos_delta < 0.01:  # Movimento minimo
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
            
            vector_features[45] = float(self.stuck_counter > self.max_stuck_frames)
            self.prev_player_pos = current_pos

        # --- VISUAL GRID (aggiornata periodicamente) ---
        # if rgb_screen is not None:
        #     self.frames_since_grid_update += 1
            
        #     if self.frames_since_grid_update >= self.grid_update_interval:
        #         self.visual_grid_cache = self._update_visual_grid(rgb_screen)
        #         self.frames_since_grid_update = 0

        if rgb_screen is not None and rgb_screen.size > 0:
            self.frames_since_grid_update += 1
            
            if self.frames_since_grid_update >= self.grid_update_interval:
                try:
                    self.visual_grid_cache = self._update_visual_grid(rgb_screen)
                    self.frames_since_grid_update = 0
                except Exception:
                    pass
        
        # --- CONCATENAZIONE FINALE ---
        # return np.concatenate((vector_features, self.visual_grid_cache)).astype(np.float32)
        result = np.concatenate((vector_features, self.visual_grid_cache)).astype(np.float32)
        
        # Sanity check: assicurati che non ci siano NaN o Inf
        result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return result