# ======================================================================
# 🎮 FILE: visualize_agent.py - Visualizza l'agente che gioca
# ======================================================================
import gymnasium as gym
import ale_py
import numpy as np
import json
import os
import sys
from functools import partial
from agent_policy import decide_move


# Registra gli ambienti Atari
gym.register_envs(ale_py)

def load_best_weights(json_path):
    """Carica i pesi migliori dal file JSON"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File non trovato: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    weights = np.array(data['best_weights'])
    fitness = data.get('best_fitness', 'N/A')
    
    print(f"✅ Pesi caricati con successo!")
    print(f"   Fitness ottenuta durante l'evoluzione: {fitness}")
    print(f"   Numero di parametri: {len(weights)}")
    
    return weights

def play_game_visual(weights, num_episodes=3, max_steps=2000, render_mode='human'):
    """
    Fa giocare l'agente visualizzando il gioco.
    
    Args:
        weights: I pesi della rete neurale (cromosoma)
        num_episodes: Quante partite giocare
        max_steps: Limite massimo di passi per partita
        render_mode: 'human' per finestra interattiva, 'rgb_array' per registrare
    """
    
    # Crea l'ambiente CON rendering
    env = gym.make("ALE/BankHeist-v5", 
                   obs_type="ram",           # Usiamo sempre la RAM come input
                   render_mode=render_mode)  # Abilita la visualizzazione
    
    # Crea la funzione-agente specifica
    agent_func = partial(decide_move, weights=weights)
    
    print(f"\n🎮 Inizio simulazione visuale ({num_episodes} partite)...")
    print("   Premi 'Q' nella finestra del gioco per interrompere\n")
    
    for episode in range(num_episodes):
        game_state, info = env.reset()
        total_reward = 0
        
        print(f"--- Partita {episode + 1}/{num_episodes} ---")
        
        for step in range(max_steps):
            # L'agente decide la mossa basandosi sulla RAM
            action = agent_func(game_state)
            
            # Assicura che l'azione sia valida
            if isinstance(action, np.integer):
                action = int(action)
            if not 0 <= action < 18:
                action = 0
            
            # Esegui l'azione nell'ambiente
            game_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Il rendering avviene automaticamente ad ogni step
            
            if terminated or truncated:
                print(f"   Partita terminata al passo {step}")
                print(f"   Punteggio totale: {total_reward}")
                break
        
        if step == max_steps - 1:
            print(f"   Raggiunto limite massimo di passi ({max_steps})")
            print(f"   Punteggio totale: {total_reward}")
    
    env.close()
    print("\n✅ Simulazione completata!")

def find_latest_solution():
    """Trova automaticamente l'ultimo file JSON nella cartella evolution_results"""
    results_dir = os.path.join(os.path.dirname(__file__), "BIO-INSPIRED/evolution_results")
    
    if not os.path.exists(results_dir):
        print(f"❌ Cartella {results_dir} non trovata!")
        print("   Esegui prima 'python run_ga_bankheist.py' per generare una soluzione.")
        return None
    
    json_files = [f for f in os.listdir(results_dir) if f.startswith('best_solution_') and f.endswith('.json')]
    
    if not json_files:
        print(f"❌ Nessun file di soluzione trovato in {results_dir}")
        print("   Esegui prima 'python run_ga_bankheist.py' per generare una soluzione.")
        return None
    
    # Ordina per data (dal nome del file) e prendi l'ultimo
    json_files.sort(reverse=True)
    latest_file = os.path.join(results_dir, json_files[0])
    
    print(f"📁 File più recente trovato: {json_files[0]}")
    return latest_file

if __name__ == "__main__":
    print("=" * 60)
    print("🎮 VISUALIZZATORE AGENTE BANK HEIST")
    print("=" * 60)
    
    # Opzione 1: Specificare manualmente il file JSON
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        print(f"\n📂 Caricamento da: {json_path}")
    else:
        # Opzione 2: Trova automaticamente l'ultimo file
        print("\n🔍 Ricerca automatica dell'ultima soluzione...")
        json_path = find_latest_solution()
        
        if json_path is None:
            sys.exit(1)
    
    try:
        # Carica i pesi
        weights = load_best_weights(json_path)
        
        # Chiedi quante partite visualizzare
        print("\n" + "=" * 60)
        try:
            num_episodes = int(input("Quante partite vuoi visualizzare? [default: 3]: ") or "3")
        except ValueError:
            num_episodes = 3
        
        # Avvia la visualizzazione
        play_game_visual(weights, num_episodes=num_episodes)
        
    except FileNotFoundError as e:
        print(f"\n❌ Errore: {e}")
        print("\nUSO:")
        print("  1. Automatico: python visualize_agent.py")
        print("  2. Manuale:    python visualize_agent.py evolution_results/best_solution_XXXXXX.json")
    except Exception as e:
        print(f"\n❌ Errore durante la visualizzazione: {e}")
        import traceback
        traceback.print_exc()