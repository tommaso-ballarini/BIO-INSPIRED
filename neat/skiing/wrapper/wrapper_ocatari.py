import gymnasium as gym
import numpy as np
from ocatari.core import OCAtari
from ocatari.ram.skiing import Player, Flag, Tree

class BioSkiingOCAtariWrapper(gym.Wrapper):
    """
    BioSkiingOCAtariWrapper - MAGNETIC GUIDANCE
    
    Miglioramenti:
    1. MAGNET REWARD: Premia l'allineamento col target FRAME PER FRAME.
    2. COLLISION PENALTY: Punisce severamente il contatto con alberi/pali.
    """
    
    def __init__(self, render_mode=None):
        self.env = OCAtari("ALE/Skiing-v5", mode="ram", hud=False, render_mode=render_mode)
        super().__init__(self.env)
        
        self.input_shape = 9
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.input_shape,), dtype=np.float32)
        
        self.prev_gates = 32
        self.prev_x = 0
        self.stuck_counter = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_gates = 32
        self.stuck_counter = 0
        
        objects = self.env.objects
        player = [o for o in objects if isinstance(o, Player)]
        if player: self.prev_x = player[0].x
        else: self.prev_x = 80
        
        return self._get_smart_obs(), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # Estrai stato corrente
        ram = self.env._env.unwrapped.ale.getRAM()
        current_gates = int(ram[107])
        
        objects = self.env.objects
        player = [o for o in objects if isinstance(o, Player)]
        p = player[0] if player else None
        
        # --- RECALCULATE OBSERVATION (Serve per il reward magnetico) ---
        # Chiamiamo la funzione interna per avere i dati del target (Delta X, Distanza)
        # Ma per efficienza, la logica è duplicata o salvata. Qui la ricalcoliamo al volo per chiarezza.
        current_obs = self._get_smart_obs() 
        target_delta_x = current_obs[3] # -1.0 (sx) a +1.0 (dx). 0.0 è CENTRO PERFETTO.
        target_exists = current_obs[5]
        
        custom_reward = 0.0
        
        # 1. GATE BONUS (Il Jackpot)
        if current_gates < self.prev_gates:
            custom_reward += 500.0
            self.stuck_counter = 0
            
        # 2. MAGNET REWARD (Nuova Guida)
        # Se c'è una porta in vista (Target Exists == 1.0)
        if target_exists > 0.5:
            # Calcola l'errore di allineamento (0.0 = Perfetto, 1.0 = Lontano)
            alignment_error = abs(target_delta_x)
            
            # Se siamo ben allineati (errore < 10%), dai un premio continuo
            if alignment_error < 0.1:
                custom_reward += 1.0 # "Bravo, tieni questa linea!"
            else:
                # Più sei disallineato, piccola penalità per spingerlo a correggere
                custom_reward -= (alignment_error * 0.5)

        # 3. COLLISION DETECTION (Nuova Punizione)
        # OCAtari non ha collision event diretto, ma possiamo stimarlo dalla distanza
        # Se siamo TROPPO vicini a un ostacolo (non una porta), ahi!
        others = [o for o in objects if isinstance(o, (Tree, Flag)) and not isinstance(o, Player)]
        if p:
            for o in others:
                # Distanza euclidea approssimativa
                dist = abs(p.x - o.x) + abs(p.y - o.y) # Manhattan dist per velocità
                if dist < 5: # 5 pixel = collisione piena
                    custom_reward -= 10.0 # PUNIZIONE SEVERA
                    
            # Penalità Bordi
            if p.x < 10 or p.x > 150:
                custom_reward -= 5.0
            
            # Anti-Camping
            if abs(p.x - self.prev_x) < 0.1:
                self.stuck_counter += 1
                custom_reward -= 1.0 # Aumentata penalità camping
            else:
                self.stuck_counter = 0
                custom_reward += 0.1
            
            self.prev_x = p.x

        custom_reward -= 0.1 # Tempo
        
        if self.stuck_counter > 100:
            truncated = True
            custom_reward -= 50.0

        self.prev_gates = current_gates
        
        return current_obs, custom_reward, terminated, truncated, info

    def _get_smart_obs(self):
        objects = self.env.objects
        player = [o for o in objects if isinstance(o, Player)]
        flags = [o for o in objects if isinstance(o, Flag)]
        trees = [o for o in objects if isinstance(o, Tree)]
        
        obs = np.zeros(self.input_shape, dtype=np.float32)
        
        if not player: return obs
        p = player[0]
        
        # 1. PLAYER
        obs[0] = (p.x - 80) / 80.0
        orientation = getattr(p, "orientation", 0)
        obs[1] = (orientation - 128.0) / 128.0
        obs[2] = (p.x - self.prev_x)
        
        # 2. TARGET TUNNEL
        upcoming_flags = sorted([f for f in flags if f.y > p.y], key=lambda f: f.y)
        gate_found = False
        target_x = 0
        target_dist = 0
        
        for i in range(len(upcoming_flags)-1):
            f1 = upcoming_flags[i]
            f2 = upcoming_flags[i+1]
            if abs(f1.y - f2.y) < 5 and abs(f1.x - f2.x) < 60:
                gate_found = True
                target_x = (f1.x + f2.x) / 2.0 
                target_dist = f1.y - p.y
                break
        
        if gate_found:
            # Calcolo Delta Normalizzato
            # Se target_x è 50 e p.x è 40, delta = +10. Normalizzato su 80px (metà schermo)
            obs[3] = (target_x - p.x) / 80.0 
            obs[4] = target_dist / 200.0     
            obs[5] = 1.0                     
        else:
            obs[3] = 0.0 
            obs[4] = 1.0 
            obs[5] = 0.0 
            
        # 3. THREATS
        threats = trees + flags
        upcoming_threats = [t for t in threats if t.y > p.y]
        if upcoming_threats:
            nearest = min(upcoming_threats, key=lambda t: (t.x-p.x)**2 + (t.y-p.y)**2)
            obs[6] = (nearest.x - p.x) / 80.0
            obs[7] = (nearest.y - p.y) / 200.0
            if isinstance(nearest, Tree): obs[8] = -1.0 
            else: obs[8] = -0.5 
            
        return obs



# import gymnasium as gym
# import numpy as np
# from ocatari.core import OCAtari
# from ocatari.ram.skiing import Player, Flag, Tree

# class BioSkiingOCAtariWrapper(gym.Wrapper):
#     """
#     BioSkiingOCAtariWrapper - THE SNIPER AGENT
    
#     Combina:
#     1. OCAtari: Per estrarre oggetti stabili dalla RAM (Player, Flag, Tree).
#     2. Geometria: Calcola il punto medio tra due bandiere (Tunnel).
    
#     INPUT (9 Valori Float32):
#     [0] Player X Norm (-1 sx, +1 dx)
#     [1] Player Orientamento (-1 sx, +1 dx)
#     [2] Player Velocità X (Inerzia)
#     [3] Target Tunnel X (Delta X verso il centro della porta)
#     [4] Target Tunnel Dist (Distanza verticale dalla porta)
#     [5] Target Exists (1.0 se vedo una porta, 0.0 se no)
#     [6] Threat X (Delta X verso l'ostacolo più vicino)
#     [7] Threat Dist (Distanza dall'ostacolo)
#     [8] Threat Type (-1 Albero, -0.5 Bandiera singola)
#     """
    
#     def __init__(self, render_mode=None):
#         # Inizializza OCAtari in modalità RAM (Veloce)
#         # hud=False rimuove punteggi e loghi dalla lista oggetti
#         self.env = OCAtari("ALE/Skiing-v5", mode="ram", hud=False, render_mode=render_mode)
#         super().__init__(self.env)
        
#         self.input_shape = 9
#         self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.input_shape,), dtype=np.float32)
        
#         # Variabili di stato per il Reward
#         self.prev_gates = 32
#         self.prev_x = 0
#         self.stuck_counter = 0

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         self.prev_gates = 32
#         self.stuck_counter = 0
        
#         # Recupera X iniziale
#         objects = self.env.objects
#         player = [o for o in objects if isinstance(o, Player)]
#         if player: self.prev_x = player[0].x
#         else: self.prev_x = 80
        
#         return self._get_smart_obs(), info

#     def step(self, action):
#         # 1. Step Fisico
#         obs, _, terminated, truncated, info = self.env.step(action)
        
#         # 2. Estrazione Dati per Reward
#         # OCAtari gestisce gli oggetti, ma per le porte rimanenti leggiamo la RAM grezza per sicurezza
#         # (A volte OCAtari non traccia il contatore porte come oggetto)
#         ram = self.env._env.unwrapped.ale.getRAM()
#         current_gates = int(ram[107])
        
#         objects = self.env.objects
#         player = [o for o in objects if isinstance(o, Player)]
#         p = player[0] if player else None
        
#         custom_reward = 0.0
        
#         # --- REWARD SHAPING ---
        
#         # A. GATE BONUS (Massiccio)
#         if current_gates < self.prev_gates:
#             custom_reward += 500.0
#             self.stuck_counter = 0
            
#         # B. PROGRESSO E VITA
#         if p:
#             # Penalità Muri (Radioattivi)
#             if p.x < 10 or p.x > 150:
#                 custom_reward -= 5.0
            
#             # Anti-Camping (Basato sul movimento X che implica discesa attiva in Skiing)
#             # In Skiing, se ti muovi lateralmente o cambi orientamento, stai scendendo.
#             # Se sei fermo immobile in X e Orientamento, sei bloccato.
#             # (Approssimazione valida per questo gioco)
#             if abs(p.x - self.prev_x) < 0.1:
#                 self.stuck_counter += 1
#                 custom_reward -= 0.5
#             else:
#                 self.stuck_counter = 0
#                 custom_reward += 0.1 # Bonus "Vivo e attivo"
            
#             self.prev_x = p.x

#         # C. TIMESTEP (Fretta)
#         custom_reward -= 0.1
        
#         # D. TERMINAZIONE
#         if self.stuck_counter > 100:
#             truncated = True
#             custom_reward -= 50.0

#         self.prev_gates = current_gates
        
#         return self._get_smart_obs(), custom_reward, terminated, truncated, info

#     def _get_smart_obs(self):
#         objects = self.env.objects
#         player = [o for o in objects if isinstance(o, Player)]
#         flags = [o for o in objects if isinstance(o, Flag)]
#         trees = [o for o in objects if isinstance(o, Tree)]
        
#         obs = np.zeros(self.input_shape, dtype=np.float32)
        
#         if not player: return obs
#         p = player[0]
        
#         # --- 1. PLAYER INFO ---
#         # X Normalizzata (0-160 -> -1 a 1)
#         obs[0] = (p.x - 80) / 80.0
        
#         # Orientamento (Fondamentale!)
#         # OCAtari lo estrae. Di solito va da 0 a 255 o range ridotto. Normalizziamo.
#         orientation = getattr(p, "orientation", 0)
#         # Empiricamente in Skiing: <128 sinistra, >128 destra (o viceversa)
#         obs[1] = (orientation - 128.0) / 128.0
        
#         # Velocità X (Stimata)
#         obs[2] = (p.x - self.prev_x) # Delta X frame-by-frame
        
#         # --- 2. TARGET TUNNEL (Geometrico) ---
#         # Cerchiamo porte DAVANTI a noi (y > p.y)
#         # Nota: In OCAtari/Skiing la Y cresce andando giù? O viceversa?
#         # Verifichiamo standard Atari: (0,0) in alto a sinistra. Quindi Y aumenta scendendo.
#         upcoming_flags = sorted([f for f in flags if f.y > p.y], key=lambda f: f.y)
        
#         gate_found = False
#         target_x = 0
#         target_dist = 0
        
#         # Algoritmo accoppiamento
#         for i in range(len(upcoming_flags)-1):
#             f1 = upcoming_flags[i]
#             f2 = upcoming_flags[i+1]
            
#             # Stessa altezza (tolleranza 4px) e vicine in X (max 60px)
#             if abs(f1.y - f2.y) < 5 and abs(f1.x - f2.x) < 60:
#                 gate_found = True
#                 target_x = (f1.x + f2.x) / 2.0 # Centro Porta
#                 target_dist = f1.y - p.y
#                 break
        
#         if gate_found:
#             obs[3] = (target_x - p.x) / 80.0 # Delta X verso il tunnel
#             obs[4] = target_dist / 200.0     # Distanza
#             obs[5] = 1.0                     # Ho un target
#         else:
#             obs[3] = 0.0 # Vai dritto
#             obs[4] = 1.0 # Lontano
#             obs[5] = 0.0 # Nessun target
            
#         # --- 3. NEAREST THREAT (Safety) ---
#         # Cerchiamo l'oggetto più vicino (Albero o Bandiera singola) da evitare
#         threats = trees + flags
#         upcoming_threats = [t for t in threats if t.y > p.y]
        
#         if upcoming_threats:
#             # Ordina per distanza Euclidea
#             nearest = min(upcoming_threats, key=lambda t: (t.x-p.x)**2 + (t.y-p.y)**2)
            
#             obs[6] = (nearest.x - p.x) / 80.0
#             obs[7] = (nearest.y - p.y) / 200.0
#             if isinstance(nearest, Tree): obs[8] = -1.0 # Albero
#             else: obs[8] = -0.5 # Palo bandiera (meno peggio ma da evitare)
            
#         return obs