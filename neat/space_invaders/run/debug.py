import sys
import os
import time
import numpy as np
import gymnasium as gym

# --- GESTIONE PERCORSI CORRETTA ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from ocatari.core import OCAtari
    from wrapper.wrapper_si_ego import SpaceInvadersEgocentricWrapper
except ImportError as e:
    print(f"❌ Errore importazione: {e}")
    sys.exit(1)

def main():
    print("🔬 DEBUG EGOCENTRICO (Visualizzazione Completa)...")
    
    try:
        # Tenta render 'human' per vedere la finestra, altrimenti headless
        env = OCAtari("SpaceInvadersNoFrameskip-v4", mode="ram", hud=False, render_mode="human")
    except:
        print("⚠️ Render grafico non disponibile. Avvio in background.")
        env = OCAtari("SpaceInvadersNoFrameskip-v4", mode="ram", hud=False, render_mode=None)

    env = SpaceInvadersEgocentricWrapper(env, skip=4)
    obs, info = env.reset(seed=42)
    
    print(f"✅ Wrapper inizializzato. N° Input: {len(obs)}")
    time.sleep(1) # Un secondo per leggere

    for step in range(2000):
        action = np.random.randint(0, 4)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- VISUALIZZAZIONE DASHBOARD ---
        # Logica trigger: Mostra se c'è pericolo (sensori > 0), UFO attivo o periodicamente
        max_sensor_val = np.max(obs[1:6])
        has_danger = max_sensor_val > 0.0
        ufo_active = obs[18] > 0.5  # Indice aggiornato per v2.0
        
        if has_danger or ufo_active or step % 20 == 0:
            os.system('cls' if os.name == 'nt' else 'clear') 
            
            print(f"🏃 STEP: {step:04d} | ACTION: {['NOOP', 'FIRE', 'RIGHT', 'LEFT'][action]}")
            # Verifica Normalizzazione: deve essere ~0.0 a sinistra e ~1.0 a destra
            print(f"📍 PLAYER X: {obs[0]:.2f} (Norm 0.0-1.0)")
            
            # --- SEZIONE RADAR (ASCII ART) ---
            # Barre grafiche per i sensori [1-5]
            bars = [" " if v == 0 else "█" * int(v * 10) for v in obs[1:6]]
            # Valori numerici dei delta [6-10]
            deltas = [f"{v:+.2f}" for v in obs[6:11]]
            
            print("\n--- RADAR PROIETTILI (Altezza Minaccia) ---")
            # Layout visivo che rispecchia i coni sopra il player
            print(f"SX  [ {bars[0]:<10} ] Val: {obs[1]:.2f} (Δ {deltas[0]})")
            print(f"    [ {bars[1]:<10} ] Val: {obs[2]:.2f} (Δ {deltas[1]})")
            print(f"CTR [ {bars[2]:<10} ] Val: {obs[3]:.2f} (Δ {deltas[2]}) <--- CENTER")
            print(f"    [ {bars[3]:<10} ] Val: {obs[4]:.2f} (Δ {deltas[3]})")
            print(f"DX  [ {bars[4]:<10} ] Val: {obs[5]:.2f} (Δ {deltas[4]})")
            
            # --- SEZIONE TATTICA ---
            print("\n--- TATTICA ---")
            # Target Alien [11]
            tgt_dir = "CENTRO"
            if obs[11] < -0.1: tgt_dir = "<< SX"
            elif obs[11] > 0.1: tgt_dir = "DX >>"
            print(f"🎯 TARGET ALIEN X: {obs[11]:+.2f} ({tgt_dir})")
            
            # UFO [17=Pos, 18=Active]
            ufo_status = "ATTIVO!" if obs[18] > 0.5 else "Assente"
            print(f"🛸 UFO: {ufo_status} (Pos Rel: {obs[17]:+.2f})")
            
            # Densità Alieni [12-15]
            print(f"\n👥 DENSITÀ LOCALE (Q1..Q4):")
            print(f"   [SX Vicino: {obs[12]:.2f}] [DX Vicino: {obs[13]:.2f}]")
            print(f"   [SX Lontano: {obs[14]:.2f}] [DX Lontano: {obs[15]:.2f}]")
            
            # Progressione Livello [16] (Ex Delta Orda)
            # Mostriamo sia la frazione che il numero stimato di alieni vivi
            estimated_aliens = int(obs[16] * 36)
            print(f"📉 STATO ORDA: {obs[16]:.2f} ({estimated_aliens}/36 Alieni vivi)")

            time.sleep(0.08) # Piccolo delay per rendere leggibile il flusso

        if terminated or truncated:
            print("\n💀 PARTITA FINITA")
            break

    env.close()

if __name__ == "__main__":
    main()