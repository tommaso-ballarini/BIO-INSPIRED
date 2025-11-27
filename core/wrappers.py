import gymnasium as gym
import numpy as np
import cv2
from gymnasium.spaces import Box

try:
    from ocatari.core import OCAtari
    OCATARI_AVAILABLE = True
except ImportError:
    OCATARI_AVAILABLE = False
    # Non stampiamo nulla qui per non sporcare i log se non serve

class PacmanHybridWrapper(gym.ObservationWrapper):
    def __init__(self, env, grid_rows=10, grid_cols=10):
        super().__init__(env)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        # --- Feature Vector Calculation ---
        # 0-1: Player (x, y)
        # 2-9: 4 Ghosts (dx, dy) relative to player
        # 10-11: Nearest PowerPill (dx, dy)
        # 12-13: Nearest Pellet (dx, dy)
        # 14-17: WALL SENSORS (Up, Right, Down, Left) -> NUOVO
        # 18-25: 4 Ghosts Velocity (vx, vy) -> SPOSTATO
        self.n_vector_features = 26 
        
        self.n_grid_features = grid_rows * grid_cols
        self.total_inputs = self.n_vector_features + self.n_grid_features
        
        self.observation_space = Box(
            low=-1.0, high=1.0, 
            shape=(self.total_inputs,), 
            dtype=np.float32
        )
        
        # Memoria per calcolare la velocità (4 fantasmi, x e y)
        self.prev_ghosts = np.zeros((4, 2), dtype=np.float32)

    def reset(self, **kwargs):
        # Reset della memoria quando inizia un nuovo episodio
        self.prev_ghosts.fill(0.0)
        return super().reset(**kwargs)

    def _get_wall_sensors(self, rgb_image, p_x, p_y):
        """
        Rileva la presenza di muri (blu) nelle 4 direzioni.
        Restituisce 4 float (0.0 o 1.0) per [UP, RIGHT, DOWN, LEFT].
        """
        sensors = np.zeros(4, dtype=np.float32)
        if rgb_image is None: 
            return sensors
            
        h, w, _ = rgb_image.shape
        
        # Stimiamo il centro di Pacman (la posizione Ocatari è top-left)
        # Lo sprite è circa 8x14
        cx = p_x + 4
        cy = p_y + 7
        
        # Distanza di controllo (un po' più in là del raggio del player)
        dist = 12 
        
        # Coordinate di controllo: Up, Right, Down, Left
        # Attenzione agli assi immagine: y è righe (verticale), x è colonne (orizzontale)
        check_points = [
            (cx, cy - dist), # UP
            (cx + dist, cy), # RIGHT
            (cx, cy + dist), # DOWN
            (cx - dist, cy)  # LEFT
        ]
        
        for i, (check_x, check_y) in enumerate(check_points):
            # Clamp coordinates per non uscire dall'immagine
            px = int(max(0, min(check_x, w-1)))
            py = int(max(0, min(check_y, h-1)))
            
            pixel = rgb_image[py, px]
            
            # Rilevamento Muro: I muri in Pacman sono blu scuro/doppia linea.
            # Colore tipico: R=33, G=33, B=150+
            # Criterio: Molto Blu e poco Rosso
            if pixel[2] > 100 and pixel[0] < 100:
                sensors[i] = 1.0
                
        return sensors

    def observation(self, obs):
        # Supporto sicuro per OCAtari
        objects = []
        if hasattr(self.env, "objects"):
            objects = [o for o in self.env.objects if o is not None]
        elif hasattr(self.env.unwrapped, "objects"):
            objects = [o for o in self.env.unwrapped.objects if o is not None]

        vector_input = np.zeros(self.n_vector_features, dtype=np.float32)
        
        # --- 1. PLAYER ---
        player = next((obj for obj in objects if "Player" in obj.category or "Pacman" in obj.category), None)
        p_x, p_y = 0.0, 0.0
        p_x_pix, p_y_pix = 0, 0
        
        if player:
            p_x_pix = player.x
            p_y_pix = player.y
            p_x = p_x_pix / 160.0
            p_y = p_y_pix / 210.0
            vector_input[0] = p_x
            vector_input[1] = p_y
            
        # --- 2. GHOSTS (Posizione + Velocità) ---
        ghosts = [obj for obj in objects if "Enemy" in obj.category or "Ghost" in obj.category]
        ghosts.sort(key=lambda x: x.category) # Ordine stabile fondamentale per la velocità
        
        for i in range(min(len(ghosts), 4)):
            g = ghosts[i]
            
            # Coordinate normalizzate correnti
            g_x_norm = g.x / 160.0
            g_y_norm = g.y / 210.0
            
            # Indici nel vettore
            pos_idx = 2 + (i * 2)      # Slot 2, 4, 6, 8
            vel_idx = 18 + (i * 2)     # Slot 18, 20, 22, 24 (SPOSTATI DOPO I SENSORI)
            
            # A. Posizione Relativa
            vector_input[pos_idx]   = g_x_norm - p_x
            vector_input[pos_idx+1] = g_y_norm - p_y
            
            # B. Velocità (Delta Posizione)
            vx = g_x_norm - self.prev_ghosts[i][0]
            vy = g_y_norm - self.prev_ghosts[i][1]
            
            vector_input[vel_idx]   = vx * 10.0
            vector_input[vel_idx+1] = vy * 10.0
            
            # Aggiorniamo la memoria
            self.prev_ghosts[i][0] = g_x_norm
            self.prev_ghosts[i][1] = g_y_norm

        # --- 3. NEAREST POWER PILL ---
        power_pills = [obj for obj in objects if "Power" in obj.category or "Ball" in obj.category]
        if player and power_pills:
            nearest = min(power_pills, key=lambda o: (o.x - player.x)**2 + (o.y - player.y)**2)
            vector_input[10] = (nearest.x / 160.0) - p_x
            vector_input[11] = (nearest.y / 210.0) - p_y

        # --- 4. NEAREST PELLET (Fallback se visibile) ---
        pellets = [obj for obj in objects if "Pellet" in obj.category or "Small" in obj.category]
        if player and pellets:
            nearest = min(pellets, key=lambda o: (o.x - player.x)**2 + (o.y - player.y)**2)
            vector_input[12] = (nearest.x / 160.0) - p_x
            vector_input[13] = (nearest.y / 210.0) - p_y
        
        # --- 5. WALL SENSORS (NUOVO) ---
        # Estrazione immagine RGB per i sensori
        rgb_screen = None
        try:
            rgb_screen = self.env.render()
            if isinstance(rgb_screen, list): rgb_screen = rgb_screen[0]
        except Exception:
            pass
            
        if player and rgb_screen is not None:
            sensors = self._get_wall_sensors(rgb_screen, p_x_pix, p_y_pix)
            vector_input[14:18] = sensors # Slot 14, 15, 16, 17
        
        # --- 6. VISUAL EXTRACTION (Migliorata) ---
        try:
            if rgb_screen is not None:
                gray = cv2.cvtColor(rgb_screen, cv2.COLOR_RGB2GRAY)
                
                # Crop dell'area di gioco (rimuove HUD punteggio in alto e vite in basso)
                # Pacman play area: circa da y=20 a y=190
                play_area = gray[20:190, :] 
                
                # Resize alla griglia desiderata (es. 10x10)
                small_grid = cv2.resize(play_area, (self.grid_cols, self.grid_rows), interpolation=cv2.INTER_AREA)
                
                # Normalizzazione
                grid_input = small_grid.flatten() / 255.0
            else:
                grid_input = np.zeros(self.n_grid_features, dtype=np.float32)
        except Exception:
            grid_input = np.zeros(self.n_grid_features, dtype=np.float32)
        
        return np.concatenate((vector_input, grid_input))


class FreewayOCAtariWrapper(gym.ObservationWrapper):
    """
    Wrapper che simula l'output di OCAtari leggendo direttamente la RAM.
    Output: Array di 11 float [Pollo_Y, Auto1_X, Auto2_X, ..., Auto10_X]
    """
    def __init__(self, env):
        # Prendiamo l'env esistente, non ne creiamo uno nuovo!
        super().__init__(env)
        
        self.num_lanes = 10
        # 11 features: 1 per il pollo (Y), 10 per le auto (X)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, 
            shape=(1 + self.num_lanes,), 
            dtype=np.float32
        )

    def observation(self, obs):
        """
        Trasforma i 128 byte della RAM in 11 feature pulite.
        """
        # obs qui è la RAM (array di 128 interi)
        
        state_vector = np.zeros(11, dtype=np.float32)
        
        # --- 1. POLLO (Byte 14) ---
        # Valore RAM: ~175 (basso/start) a ~15 (alto/goal)
        chicken_ram_y = obs[14]
        # Normalizziamo: 0.0 = Start, 1.0 = Goal
        # Usiamo 170.0 e 18.0 come range approssimativo sicuro
        y_norm = (170.0 - chicken_ram_y) / (170.0 - 18.0)
        state_vector[0] = np.clip(y_norm, 0.0, 1.0)
        
        # --- 2. AUTO (Byte 108 - 117) ---
        # In Freeway, i byte da 108 a 117 contengono la X delle auto nelle 10 corsie.
        # Valore RAM: 0 a 160 (circa)
        car_bytes = [108, 109, 110, 111, 112, 113, 114, 115, 116, 117]
        
        # Invertiamo l'ordine perché visivamente la corsia 1 è in basso
        # ma in memoria spesso l'ordine è inverso. Questo ordine funziona bene.
        for i, byte_idx in enumerate(car_bytes):
            if byte_idx < len(obs):
                car_x = obs[byte_idx]
                # Normalizziamo X tra 0.0 e 1.0
                state_vector[1 + i] = car_x / 160.0
            else:
                state_vector[1 + i] = 0.0 # Valore safe se la RAM è strana
                
        return state_vector