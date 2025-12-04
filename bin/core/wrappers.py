import gymnasium as gym
import numpy as np
import cv2
from gymnasium.spaces import Box

try:
    from ocatari.core import OCAtari
    OCATARI_AVAILABLE = True
except ImportError:
    OCATARI_AVAILABLE = False

# =========================================
# 1. PACMAN WRAPPER
# =========================================
class PacmanHybridWrapper(gym.ObservationWrapper):
    def __init__(self, env, grid_rows=10, grid_cols=10):
        super().__init__(env)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        # 26 Vector Features + Grid Features
        self.n_vector_features = 26 
        self.n_grid_features = grid_rows * grid_cols
        self.total_inputs = self.n_vector_features + self.n_grid_features
        
        self.observation_space = Box(
            low=-1.0, high=1.0, 
            shape=(self.total_inputs,), 
            dtype=np.float32
        )
        self.prev_ghosts = np.zeros((4, 2), dtype=np.float32)

    def reset(self, **kwargs):
        self.prev_ghosts.fill(0.0)
        return super().reset(**kwargs)

    def _get_wall_sensors(self, rgb_image, p_x, p_y):
        sensors = np.zeros(4, dtype=np.float32)
        if rgb_image is None: return sensors
        h, w, _ = rgb_image.shape
        cx, cy = p_x + 4, p_y + 7
        dist = 12 
        check_points = [(cx, cy - dist), (cx + dist, cy), (cx, cy + dist), (cx - dist, cy)]
        
        for i, (check_x, check_y) in enumerate(check_points):
            px = int(max(0, min(check_x, w-1)))
            py = int(max(0, min(check_y, h-1)))
            pixel = rgb_image[py, px]
            if pixel[2] > 100 and pixel[0] < 100:
                sensors[i] = 1.0
        return sensors

    def observation(self, obs):
        objects = getattr(self.env, "objects", getattr(self.env.unwrapped, "objects", []))
        vector_input = np.zeros(self.n_vector_features, dtype=np.float32)
        
        player = next((obj for obj in objects if "Player" in obj.category or "Pacman" in obj.category), None)
        p_x, p_y = 0.0, 0.0
        p_x_pix, p_y_pix = 0, 0
        
        if player:
            p_x_pix, p_y_pix = player.x, player.y
            p_x, p_y = p_x_pix / 160.0, p_y_pix / 210.0
            vector_input[0], vector_input[1] = p_x, p_y
            
        ghosts = [obj for obj in objects if "Enemy" in obj.category or "Ghost" in obj.category]
        ghosts.sort(key=lambda x: x.category)
        
        for i in range(min(len(ghosts), 4)):
            g = ghosts[i]
            g_x_norm, g_y_norm = g.x / 160.0, g.y / 210.0
            pos_idx = 2 + (i * 2)
            vel_idx = 18 + (i * 2)
            
            vector_input[pos_idx]   = g_x_norm - p_x
            vector_input[pos_idx+1] = g_y_norm - p_y
            
            vx = g_x_norm - self.prev_ghosts[i][0]
            vy = g_y_norm - self.prev_ghosts[i][1]
            vector_input[vel_idx]   = vx * 10.0
            vector_input[vel_idx+1] = vy * 10.0
            
            self.prev_ghosts[i][0] = g_x_norm
            self.prev_ghosts[i][1] = g_y_norm

        power_pills = [obj for obj in objects if "Power" in obj.category or "Ball" in obj.category]
        if player and power_pills:
            nearest = min(power_pills, key=lambda o: (o.x - player.x)**2 + (o.y - player.y)**2)
            vector_input[10] = (nearest.x / 160.0) - p_x
            vector_input[11] = (nearest.y / 210.0) - p_y

        pellets = [obj for obj in objects if "Pellet" in obj.category or "Small" in obj.category]
        if player and pellets:
            nearest = min(pellets, key=lambda o: (o.x - player.x)**2 + (o.y - player.y)**2)
            vector_input[12] = (nearest.x / 160.0) - p_x
            vector_input[13] = (nearest.y / 210.0) - p_y
        
        rgb_screen = None
        try:
            rgb_screen = self.env.render()
            if isinstance(rgb_screen, list): rgb_screen = rgb_screen[0]
        except Exception: pass
            
        if player and rgb_screen is not None:
            sensors = self._get_wall_sensors(rgb_screen, p_x_pix, p_y_pix)
            vector_input[14:18] = sensors
        
        try:
            if rgb_screen is not None:
                gray = cv2.cvtColor(rgb_screen, cv2.COLOR_RGB2GRAY)
                play_area = gray[20:190, :] 
                small_grid = cv2.resize(play_area, (self.grid_cols, self.grid_rows), interpolation=cv2.INTER_AREA)
                grid_input = small_grid.flatten() / 255.0
            else:
                grid_input = np.zeros(self.n_grid_features, dtype=np.float32)
        except Exception:
            grid_input = np.zeros(self.n_grid_features, dtype=np.float32)
        
        return np.concatenate((vector_input, grid_input))


# =========================================
# 2. FREEWAY WRAPPER
# =========================================
class FreewayOCAtariWrapper(gym.ObservationWrapper):
    def __init__(self, env_name="Freeway-v4"):
        if not OCATARI_AVAILABLE:
            raise ImportError("OCAtari non installato")
        
        # Creiamo l'ambiente OCAtari internamente se passato come stringa, 
        # ma qui solitamente riceviamo già l'env. Gestiamo entrambi i casi.
        if isinstance(env_name, str):
             self.ocatari_env = OCAtari(env_name, mode="ram", hud=False, render_mode="rgb_array")
             self.ocatari_env.reset()
             super().__init__(self.ocatari_env)
        else:
             super().__init__(env_name)

        self.num_lanes = 10
        self.observation_space = Box(low=0.0, high=1.0, shape=(1 + self.num_lanes,), dtype=np.float32)
        self.SCREEN_H = 210.0
        self.SCREEN_W = 160.0

    def observation(self, obs):
        objects = getattr(self.env, "objects", getattr(self.env.unwrapped, "objects", []))
        state_vector = np.zeros(1 + self.num_lanes, dtype=np.float32)
        
        chicken = None
        cars = []

        for obj in objects:
            name = obj.category.lower()
            if "chicken" in name: chicken = obj
            elif "car" in name or "enemy" in name: cars.append(obj)
        
        if chicken:
            y_norm = (175.0 - chicken.y) / (175.0 - 15.0)
            state_vector[0] = np.clip(y_norm, 0.0, 1.0)
        
        cars.sort(key=lambda c: c.y, reverse=True)
        for i in range(min(len(cars), self.num_lanes)):
            state_vector[1 + i] = cars[i].x / self.SCREEN_W
            
        return state_vector

    def step(self, action):
        obs, reward, truncated, terminated, info = self.env.step(action)
        return self.observation(obs), reward, truncated, terminated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


# =========================================
# 3. SPACE INVADERS WRAPPER (NUOVO)
# =========================================
class SpaceInvadersOCAtariWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_features = 8
        self.observation_space = Box(
            low=-1.0, high=1.0, 
            shape=(self.n_features,), 
            dtype=np.float32
        )
        
        self.W = 160.0
        self.H = 210.0
        
        # Gym Actions: 0:NOOP, 1:FIRE, 2:RIGHT, 3:LEFT, 4:RIGHTFIRE, 5:LEFTFIRE
        self.action_map = {
            0: 0, # NOOP
            1: 3, # LEFT
            2: 2, # RIGHT
            3: 1  # FIRE
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._get_obs(), info
        
    def step(self, action_idx):
        real_action = self.action_map.get(action_idx, 0)
        obs, reward, truncated, terminated, info = self.env.step(real_action)
        return self._get_obs(), reward, truncated, terminated, info

    def observation(self, obs):
        return self._get_obs()

    def _get_obs(self):
        objects = getattr(self.env, "objects", getattr(self.env.unwrapped, "objects", []))
        player = None
        aliens = []
        bullets = []
        shields = []
        
        for obj in objects:
            cat = obj.category.lower()
            if "player" in cat and "score" not in cat: player = obj
            elif "alien" in cat or "enemy" in cat: aliens.append(obj)
            elif "bullet" in cat: bullets.append(obj)
            elif "shield" in cat: shields.append(obj)
            
        vec = np.zeros(self.n_features, dtype=np.float32)
        
        # 1. Player
        px, py = 0.5, 0.9
        if player:
            px = player.x / self.W
            py = player.y / self.H
            vec[0] = px
            
        # 2. Nearest Alien
        if aliens:
            aliens.sort(key=lambda a: (a.x/self.W - px)**2 + (a.y/self.H - py)**2)
            nearest = aliens[0]
            vec[1] = (nearest.x / self.W) - px
            vec[2] = (nearest.y / self.H) - py
            if abs((nearest.x / self.W) - px) < 0.05:
                vec[7] = 1.0
            vec[5] = len(aliens) / 36.0
        else:
            vec[2] = -1.0
            
        # 3. Incoming Bullets
        incoming_bullets = [b for b in bullets if b.y < (py * self.H)]
        if incoming_bullets:
            incoming_bullets.sort(key=lambda b: (b.x/self.W - px)**2 + (b.y/self.H - py)**2)
            nb = incoming_bullets[0]
            vec[3] = (nb.x / self.W) - px
            vec[4] = (nb.y / self.H) - py
        else:
            vec[3] = 1.0
            vec[4] = -1.0
            
        # 4. Shield
        is_covered = False
        for s in shields:
            if abs((s.x / self.W) - px) < 0.05:
                is_covered = True
                break
        vec[6] = 1.0 if is_covered else 0.0
        
        return vec
    

class NeatMsPacmanWrapper(gym.ObservationWrapper):
    """
    Wrapper per Ms. Pac-Man che implementa l'ingegneria delle feature
    descritta nel PDF tecnico (Sezione 4.1).
    
    Estrae feature semantiche da OCAtari (REM) e le trasforma in un
    vettore fisso a bassa dimensionalità per NEAT.
    
    Feature Vector (20 dimensioni):
    - [0-1]: Player position (x, y) normalizzate
    - [2]: Player direction (0-3 mappato a 0.0-1.0)
    - [3-10]: 4 Ghosts relative position (dx, dy) per ghost
    - [11-14]: 4 Ghosts edibility status (0.0 o 1.0)
    - [15-16]: Nearest Power Pill relative position (dx, dy)
    - [17-18]: Nearest Fruit relative position (dx, dy)
    - [19]: Dots eaten count (normalizzato)
    """
    
    def _init_(self, env, max_objects=None):
        super()._init_(env)
        
        # Dimensione fissa del vettore di feature
        self.num_features = 20
        
        # Spazio di osservazione: vettore 1D di 20 float normalizzati in [0, 1]
        self.observation_space = Box(
            low=0.0, 
            high=1.0, 
            shape=(self.num_features,), 
            dtype=np.float32
        )
        
        # Costanti per normalizzazione (dimensioni schermo Atari standard)
        self.SCREEN_WIDTH = 160.0
        self.SCREEN_HEIGHT = 210.0
        
        # Massimo dots nel gioco (approssimativo per normalizzazione)
        self.MAX_DOTS = 240.0
        
    def observation(self, obs):
        """
        Trasforma l'osservazione OCAtari in un vettore di feature.
        
        Args:
            obs: Output di OCAtari (può essere ignorato se usiamo self.env.objects)
            
        Returns:
            np.array: Vettore di 20 feature normalizzate
        """
        # Inizializza vettore di feature a zero
        features = np.zeros(self.num_features, dtype=np.float32)
        
        # Estrai oggetti da OCAtari
        # OCAtari popola self.env.objects con la lista di oggetti rilevati
        objects = []
        if hasattr(self.env, 'objects'):
            objects = [o for o in self.env.objects if o is not None]
        elif hasattr(self.env.unwrapped, 'objects'):
            objects = [o for o in self.env.unwrapped.objects if o is not None]
        
        # --- 1. ESTRAZIONE PLAYER ---
        player = self._find_player(objects)
        player_x, player_y = 0.0, 0.0
        
        if player:
            # Normalizza coordinate in [0, 1]
            player_x = player.x / self.SCREEN_WIDTH
            player_y = player.y / self.SCREEN_HEIGHT
            features[0] = player_x
            features[1] = player_y
            
            # Direzione (OCAtari fornisce orientation per alcuni giochi)
            # Mappatura: 0=Right, 1=Left, 2=Up, 3=Down -> normalizzato a [0, 1]
            if hasattr(player, 'orientation'):
                features[2] = player.orientation / 3.0
        
        # --- 2. ESTRAZIONE GHOSTS ---
        ghosts = self._find_ghosts(objects)
        
        # Ordina i ghosts per garantire input consistente
        # (ordine alfabetico per categoria: Blinky, Inky, Pinky, Sue)
        ghosts.sort(key=lambda g: g.category if hasattr(g, 'category') else '')
        
        for i in range(min(len(ghosts), 4)):
            ghost = ghosts[i]
            
            # Posizione relativa (delta x, delta y)
            # Nota: range può essere negativo, quindi spostiamo in [0, 1]
            dx = (ghost.x / self.SCREEN_WIDTH) - player_x
            dy = (ghost.y / self.SCREEN_HEIGHT) - player_y
            
            # Normalizza delta in [0, 1] con offset 0.5
            # (dx può andare da -1 a +1, quindi (dx + 1) / 2 lo porta in [0, 1])
            features[3 + i*2] = (dx + 1.0) / 2.0
            features[4 + i*2] = (dy + 1.0) / 2.0
            
            # Stato di commestibilità (CRITICO per la strategia)
            # OCAtari marca i ghosts come "edible" quando sono blu
            edible = 0.0
            if hasattr(ghost, 'rgb'):
                # I ghosts commestibili sono blu in Ms. Pac-Man
                # RGB tipico: (33, 33, 255) o simili
                if ghost.rgb[2] > 200 and ghost.rgb[0] < 100:
                    edible = 1.0
            
            features[11 + i] = edible
        
        # --- 3. NEAREST POWER PILL ---
        power_pills = self._find_power_pills(objects)
        
        if player and power_pills:
            nearest_pill = min(
                power_pills, 
                key=lambda p: (p.x - player.x)*2 + (p.y - player.y)*2
            )
            
            dx = (nearest_pill.x / self.SCREEN_WIDTH) - player_x
            dy = (nearest_pill.y / self.SCREEN_HEIGHT) - player_y
            
            features[15] = (dx + 1.0) / 2.0
            features[16] = (dy + 1.0) / 2.0
        else:
            # Se non ci sono power pills, centra il valore (nessuna informazione)
            features[15] = 0.5
            features[16] = 0.5
        
        # --- 4. NEAREST FRUIT ---
        fruits = self._find_fruits(objects)
        
        if player and fruits:
            nearest_fruit = min(
                fruits,
                key=lambda f: (f.x - player.x)*2 + (f.y - player.y)*2
            )
            
            dx = (nearest_fruit.x / self.SCREEN_WIDTH) - player_x
            dy = (nearest_fruit.y / self.SCREEN_HEIGHT) - player_y
            
            features[17] = (dx + 1.0) / 2.0
            features[18] = (dy + 1.0) / 2.0
        else:
            features[17] = 0.5
            features[18] = 0.5
        
        # --- 5. DOTS EATEN COUNT ---
        # Questa informazione può essere estratta da info o dalla RAM
        # Per ora usiamo un placeholder (OCAtari potrebbe non esporla direttamente)
        # In alternativa, possiamo contare i pellet rimanenti
        dots_eaten = 0.0
        
        # Se OCAtari espone dots_eaten_count in info, usalo
        if hasattr(self.env, '_get_ram'):
            # AtariARI mapping per Ms. Pac-Man: dots_eaten_count è in RAM
            # Questo richiede conoscenza specifica della mappa RAM
            # Per semplicità, lasciamo a 0 o implementiamo conteggio pellet
            pass
        
        features[19] = dots_eaten / self.MAX_DOTS
        
        return features
    
    def _find_player(self, objects):
        """Trova l'oggetto Player nella lista OCAtari"""
        for obj in objects:
            if hasattr(obj, 'category'):
                cat = obj.category.lower()
                if 'player' in cat or 'pacman' in cat or 'pac-man' in cat:
                    return obj
        return None
    
    def _find_ghosts(self, objects):
        """Trova tutti i ghosts nella lista OCAtari"""
        ghosts = []
        for obj in objects:
            if hasattr(obj, 'category'):
                cat = obj.category.lower()
                if 'enemy' in cat or 'ghost' in cat:
                    ghosts.append(obj)
        return ghosts
    
    def _find_power_pills(self, objects):
        """Trova tutte le Power Pills"""
        pills = []
        for obj in objects:
            if hasattr(obj, 'category'):
                cat = obj.category.lower()
                # Le Power Pills sono solitamente "PowerPill" o "Ball" grandi
                if 'power' in cat or ('ball' in cat and hasattr(obj, 'w') and obj.w > 4):
                    pills.append(obj)
        return pills
    
    def _find_fruits(self, objects):
        """Trova i frutti bonus"""
        fruits = []
        for obj in objects:
            if hasattr(obj, 'category'):
                cat = obj.category.lower()
                if 'fruit' in cat or 'cherry' in cat or 'strawberry' in cat:
                    fruits.append(obj)
        return fruits
    
    def reset(self, **kwargs):
        """Reset dell'ambiente"""
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
    
    def step(self, action):
        """Step dell'ambiente con trasformazione dell'osservazione"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info