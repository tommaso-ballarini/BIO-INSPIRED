import os
import sys
import neat
import pickle
import numpy as np
import gymnasium as gym

# --- 1. SETUP PERCORSI ---
# Aggiunge la cartella principale al path per trovare il wrapper
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import del Wrapper OCAtari
try:
    from wrapper.wrapper_ocatari import BioSkiingOCAtariWrapper
    print("✅ BioSkiingOCAtariWrapper importato correttamente.")
except ImportError:
    print("❌ CRITICAL: Non trovo 'wrapper/wrapper_ocatari.py'.")
    print("   Assicurati di aver salvato il wrapper OCAtari nella cartella wrapper.")
    sys.exit(1)

# --- CONFIGURAZIONE ---
# Cartella risultati (deve coincidere con quella del training)
RESULTS_DIR = os.path.join(project_root, "evolution_results", "ocatari_run")
# Configurazione NEAT
CONFIG_PATH = os.path.join(project_root, "config", "config_ocatari.txt")

def get_latest_winner():
    """Trova il file .pkl più recente nella cartella results."""
    from glob import glob
    search_pattern = os.path.join(RESULTS_DIR, "*.pkl")
    list_of_files = glob(search_pattern)
    
    if not list_of_files:
        print(f"❌ Nessun file trovato in: {RESULTS_DIR}")
        sys.exit(1)
        
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"📂 Caricamento file: {os.path.basename(latest_file)}")
    return latest_file

def visualize():
    # 1. Carica Configurazione
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Config non trovato: {CONFIG_PATH}")
        return

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)

    # Verifica che il config sia impostato per OCAtari
    if config.genome_config.num_inputs != 9:
        print(f"⚠️ ATTENZIONE: Il config specifica {config.genome_config.num_inputs} input.")
        print("   Il wrapper OCAtari ne richiede 9.")
        print("   Assicurati di usare il config corretto o il replay fallirà.")

    # 2. Carica il Campione (Genoma)
    winner_path = get_latest_winner()
    with open(winner_path, "rb") as f:
        winner = pickle.load(f)

    # Gestione compatibilità (se salvato come tupla o oggetto singolo)
    if isinstance(winner, tuple):
        winner = winner[0]

    print(f"🏆 Fitness registrata: {winner.fitness}")

    # 3. Crea la Rete Neurale
    # OCAtari/Skiing richiede memoria, usiamo RecurrentNetwork
    net = neat.nn.RecurrentNetwork.create(winner, config)

    # 4. Avvia Ambiente (Modalità Human per vedere il gioco)
    print("\n🎮 Avvio OCAtari (Skiing-v5)...")
    try:
        # render_mode="human" apre la finestra di gioco
        env = BioSkiingOCAtariWrapper(render_mode="human")
    except Exception as e:
        print(f"❌ Errore avvio OCAtari: {e}")
        print("   Assicurati di aver installato: pip install ocatari[all]")
        return

    observation, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    
    print("\n--- INIZIO DISCESA (Premi Ctrl+C per interrompere) ---")
    print(f"   Input Rete: {len(observation)} (Attesi: 9)")

    try:
        while not done:
            # OCAtari Wrapper ci dà già i 9 float puliti
            inputs = observation
            
            # Attivazione Rete
            output = net.activate(inputs)
            action = np.argmax(output) # 0, 1, 2 (Noop, Right, Left)
            
            # Step Gioco
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            done = terminated or truncated
            steps += 1
            
            # Stampa info debug ogni tanto
            if steps % 60 == 0: # Ogni ~2 secondi
                # input[3] è il Delta Target, input[5] è "Vedo Porta?"
                target_status = "CERCANDO..."
                if inputs[5] > 0.5:
                    target_status = f"TARGET LOCKED (Dist: {inputs[4]:.2f})"
                print(f"Step {steps} | Reward: {total_reward:.1f} | {target_status}")

    except KeyboardInterrupt:
        print("\n🛑 Interrotto dall'utente.")
    finally:
        env.close()
        print(f"\n🏁 Partita terminata.")
        print(f"   Punteggio Totale (Fitness): {total_reward:.2f}")

if __name__ == "__main__":
    visualize()