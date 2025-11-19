import gymnasium as gym
import numpy as np
import cv2
from gymnasium.spaces import Box

class HybridPacmanWrapper(gym.ObservationWrapper):
    def __init__(self, env, grid_rows=8, grid_cols=8):
        super().__init__(env)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        self.n_vector_features = 14
        self.n_grid_features = grid_rows * grid_cols
        self.total_inputs = self.n_vector_features + self.n_grid_features
        
        self.observation_space = Box(
            low=0.0, high=1.0, 
            shape=(self.total_inputs,), 
            dtype=np.float32
        )

    def observation(self, obs):
        # --- PARTE 1: ESTRAZIONE RAM (uguale a prima) ---
        if hasattr(self.env, "objects"):
            objects = self.env.objects
        else:
            objects = self.env.unwrapped.objects
            
        vector_input = np.zeros(self.n_vector_features, dtype=np.float32)
        
        # Player
        player = next((obj for obj in objects if "Pacman" in obj.category or "Player" in obj.category), None)
        p_x, p_y = 0.0, 0.0
        if player:
            p_x = player.x / 160.0
            p_y = player.y / 210.0
            vector_input[0] = p_x
            vector_input[1] = p_y
            
        # Ghosts
        ghosts = [obj for obj in objects if "Ghost" in obj.category or "Enemy" in obj.category]
        ghosts.sort(key=lambda x: x.category)
        for i in range(min(len(ghosts), 4)):
            g = ghosts[i]
            g_x = g.x / 160.0
            g_y = g.y / 210.0
            idx = 2 + (i * 3)
            vector_input[idx]   = g_x - p_x
            vector_input[idx+1] = g_y - p_y
            vector_input[idx+2] = 0.0 # Placeholder edible

        # --- PARTE 2: ESTRAZIONE VISIVA (CORRETTA) ---
        # ERRORE PRECEDENTE: cv2.cvtColor(obs, ...) -> obs non era l'immagine!
        # SOLUZIONE: Chiediamo esplicitamente il render RGB all'ambiente.
        
        try:
            # Recuperiamo l'immagine RGB (210, 160, 3)
            # Nota: Funziona solo se nel main hai messo render_mode="rgb_array"
            rgb_screen = self.env.render() 
            
            # Se per caso render() restituisce una lista (alcune versioni vecchie), prendi il primo elemento
            if isinstance(rgb_screen, list):
                rgb_screen = rgb_screen[0]

            # Converti in scala di grigi
            gray = cv2.cvtColor(rgb_screen, cv2.COLOR_RGB2GRAY)
            
            # Ritaglia area di gioco (rimuovi HUD punteggi in basso/alto)
            play_area = gray[0:172, :] 
            
            # Resize alla dimensione griglia (es. 8x8)
            small_grid = cv2.resize(play_area, (self.grid_cols, self.grid_rows), interpolation=cv2.INTER_AREA)
            
            # Normalizza (0.0 - 1.0)
            grid_input = small_grid.flatten() / 255.0
            
        except Exception as e:
            # Fallback di sicurezza per non far crashare tutto se il rendering fallisce
            print(f"Warning Visual Extraction: {e}")
            grid_input = np.zeros(self.n_grid_features, dtype=np.float32)
        
        # --- UNIONE ---
        final_input = np.concatenate((vector_input, grid_input))
        
        return final_input