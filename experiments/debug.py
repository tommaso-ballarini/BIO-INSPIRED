import gymnasium as gym
import numpy as np
import time
import sys
import os

# Aggiungiamo la root del progetto al path per importare il wrapper
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from core.wrappers import FreewayOCAtariWrapper
except ImportError:
    print("ERRORE: Impossibile importare FreewayOCAtariWrapper.")
    print("Assicurati di eseguire questo script dalla cartella 'experiments/' o che 'core/' sia visibile.")
    sys.exit(1)

def debug_freeway():
    # Creiamo l'ambiente con render_mode='human' per vedere il gioco
    # Usa 'ALE/Freeway-v5' o 'Freeway-v4' a seconda della tua installazione
    try:
        env = gym.make('Freeway-v4', render_mode='human', obs_type='ram')
    except:
        env = gym.make('ALE/Freeway-v5', render_mode='human', obs_type='ram')
    
    # Applichiamo il wrapper per avere la logica di conversione
    # NOTA: Il wrapper originale gymnasium restituisce la RAM, 
    # noi useremo la funzione .observation() del tuo wrapper manualmente 
    # per vedere il "prima" (RAM) e il "dopo" (Feature).
    wrapper = FreewayOCAtariWrapper(env)
    
    observation, _ = env.reset()
    
    print("\n--- AVVIO DEBUGGING FREEWAY ---")
    print("Controlla la console mentre guardi la finestra di gioco.")
    print("Premi CTRL+C nella console per terminare.\n")
    
    try:
        while True:
            # 1. Ottieni i valori grezzi dalla RAM
            # Il Pollo è ipotizzato al byte 14
            raw_chicken_y = observation[14]
            
            # Le Auto sono ipotizzate ai byte 108-117
            car_indices = [108, 109, 110, 111, 112, 113, 114, 115, 116, 117]
            raw_cars_x = [observation[i] for i in car_indices]
            
            # 2. Ottieni i valori normalizzati dal Wrapper
            # (Chiamiamo direttamente il metodo interno per vedere cosa "vede" l'agente)
            features = wrapper.observation(observation)
            norm_chicken = features[0]
            norm_cars = features[1:]
            
            # 3. Stampa di confronto
            # Usiamo caratteri speciali per pulire la riga e sovrascrivere (simil-animazione)
            # oppure stampiamo a cascata se preferisci lo storico.
            
            os.system('cls' if os.name == 'nt' else 'clear') # Pulisce la console (opzionale)
            
            print(f"--- FRAME DEBUG ---")
            print(f"POLLO (Y):")
            print(f"  RAM [14]     : {raw_chicken_y} (Valore grezzo: solitamente ~170 basso, ~18 alto)")
            print(f"  INPUT AGENTE : {norm_chicken:.4f} (0.0 = Start, 1.0 = Goal)")
            
            print(f"\nAUTO (X) - 10 Corsie:")
            print(f"  {'Corsia':<6} | {'Byte':<5} | {'Valore RAM (0-160)':<18} | {'Input Agente (0.0-1.0)':<20}")
            print("-" * 60)
            
            for i, (byte_idx, raw_val, norm_val) in enumerate(zip(car_indices, raw_cars_x, norm_cars)):
                # Visualizzazione grafica testuale della posizione auto
                bar_len = 10
                pos = int(norm_val * bar_len)
                bar = "[" + " " * pos + "C" + " " * (bar_len - pos - 1) + "]"
                
                print(f"  #{i+1:<5} | {byte_idx:<5} | {raw_val:<3} {bar:<14} | {norm_val:.4f}")

            # 4. Step del gioco (Azione casuale o NOOP per osservare)
            # Mettiamo azione 1 (UP) ogni tanto per far muovere il pollo
            action = 1 if np.random.rand() > 0.9 else 0 
            
            observation, reward, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                observation, _ = env.reset()
                
            # Rallentiamo per leggere i numeri (0.1s = 10 FPS)
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nDebug terminato dall'utente.")
        env.close()

if __name__ == "__main__":
    debug_freeway()