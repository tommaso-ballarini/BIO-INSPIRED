import sys
import os
import neat
import numpy as np
import pickle
import gymnasium as gym
import time

# IMPORTANTE: Registra gli ambienti ALE
try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    print("❌ ale-py non installato. Installa con: pip install ale-py")
    sys.exit(1)

# --- CONFIGURAZIONE ---
ENV_ID = "ALE/Skiing-v5"
MAX_STEPS = 20000
FRAME_DELAY = 0.03  # Secondi tra frame (per rallentare la visualizzazione)


def normalize_ram(ram_state):
    """Normalizza lo stato RAM (128 bytes) in [0, 1]"""
    return np.array(ram_state, dtype=np.float32) / 255.0


def load_winner(winner_path):
    """Carica il genoma vincitore e la configurazione"""
    if not os.path.exists(winner_path):
        raise FileNotFoundError(f"❌ File winner non trovato: {winner_path}")
    
    with open(winner_path, 'rb') as f:
        winner, config = pickle.load(f)
    
    print("✅ Winner caricato con successo!")
    print(f"   Fitness: {winner.fitness}")
    print(f"   Genome ID: {winner.key}")
    
    return winner, config


def visualize_gameplay(genome, config, num_episodes=3, render_mode='human'):
    """
    Visualizza il gameplay del genoma NEAT
    
    Args:
        genome: Il genoma NEAT da visualizzare
        config: La configurazione NEAT
        num_episodes: Numero di episodi da visualizzare
        render_mode: 'human' per finestra, 'rgb_array' per array numpy
    """
    # Crea l'ambiente con rendering
    env = gym.make(
        ENV_ID,
        obs_type="ram",
        frameskip=1,
        repeat_action_probability=0.0,
        render_mode=render_mode
    )
    
    # Crea la rete neurale dal genoma
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    print("\n" + "="*80)
    print(f"🎮 VISUALIZZAZIONE SKIING - {num_episodes} episodi")
    print("="*80)
    print(f"🎯 Fitness del genoma: {genome.fitness}")
    print(f"🎬 Render mode: {render_mode}")
    print(f"🎮 Azioni disponibili: {env.unwrapped.get_action_meanings()}")
    print("="*80 + "\n")
    
    for episode in range(num_episodes):
        print(f"\n🎬 EPISODIO {episode + 1}/{num_episodes}")
        print("-" * 80)
        
        observation, info = env.reset()
        done = False
        steps = 0
        total_reward = 0.0
        action_counts = {0: 0, 1: 0, 2: 0}  # LEFT, NOOP, RIGHT
        
        while not done and steps < MAX_STEPS:
            # Normalizza osservazione
            norm_obs = normalize_ram(observation)
            
            # Ottieni output dalla rete
            output = net.activate(norm_obs)
            action = int(np.argmax(output)) % 3
            action_counts[action] += 1
            
            # Esegui azione
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            # Rallenta per visualizzazione
            if render_mode == 'human':
                time.sleep(FRAME_DELAY)
            
            # Mostra info ogni 500 steps
            if steps % 500 == 0:
                print(f"   Step {steps:5d} | Reward: {total_reward:8.1f} | "
                      f"Lives: {info.get('lives', 'N/A')}")
        
        # Statistiche finali episodio
        print("\n📊 STATISTICHE EPISODIO:")
        print(f"   Steps totali: {steps}")
        print(f"   Reward finale: {total_reward}")
        print(f"   Terminato: {terminated}")
        print(f"   Vite rimanenti: {info.get('lives', 'N/A')}")
        print(f"   Fitness calcolata: {abs(total_reward)}")
        print("\n📈 DISTRIBUZIONE AZIONI:")
        total_actions = sum(action_counts.values())
        for action_idx, count in action_counts.items():
            action_name = env.unwrapped.get_action_meanings()[action_idx]
            percentage = (count / total_actions * 100) if total_actions > 0 else 0
            print(f"   {action_name:6s}: {count:5d} ({percentage:5.1f}%)")
        print("-" * 80)
    
    env.close()
    print("\n✅ Visualizzazione completata!\n")


def main():
    """Funzione principale"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualizza il gameplay di un genoma NEAT addestrato su Skiing'
    )
    parser.add_argument(
        'winner_path',
        type=str,
        help='Path al file winner.pkl (es: evolution_results/skiing/winner_20241202_120000.pkl)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=3,
        help='Numero di episodi da visualizzare (default: 3)'
    )
    parser.add_argument(
        '--no-render',
        action='store_true',
        help='Non mostrare finestra di gioco (solo statistiche)'
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=0.03,
        help='Ritardo tra frame in secondi (default: 0.03, più basso = più veloce)'
    )
    
    args = parser.parse_args()
    
    # Imposta velocità
    global FRAME_DELAY
    FRAME_DELAY = args.speed
    
    # Carica winner
    try:
        winner, config = load_winner(args.winner_path)
    except Exception as e:
        print(f"❌ Errore nel caricamento del winner: {e}")
        sys.exit(1)
    
    # Visualizza
    render_mode = None if args.no_render else 'human'
    
    try:
        visualize_gameplay(winner, config, num_episodes=args.episodes, render_mode=render_mode)
    except KeyboardInterrupt:
        print("\n\n⏸ Visualizzazione interrotta dall'utente")
    except Exception as e:
        print(f"\n\n❌ Errore durante visualizzazione: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Esempio di utilizzo se chiamato direttamente senza argomenti
    if len(sys.argv) == 1:
        print("="*80)
        print("🎮 VISUALIZZATORE SKIING NEAT")
        print("="*80)
        print("\n📖 UTILIZZO:")
        print("\nPer visualizzare un genoma salvato:")
        print("  python visualize_skiing.py path/to/winner.pkl")
        print("\nOpzioni:")
        print("  --episodes N        Numero di episodi (default: 3)")
        print("  --no-render         Solo statistiche, senza finestra di gioco")
        print("  --speed 0.01        Velocità visualizzazione (default: 0.03s)")
        print("\n📝 ESEMPI:")
        print("  # Visualizza con finestra di gioco")
        print("  python visualize_skiing.py evolution_results/skiing/winner_20241202_120000.pkl")
        print("\n  # Più veloce")
        print("  python visualize_skiing.py winner.pkl --speed 0.01")
        print("\n  # Solo statistiche")
        print("  python visualize_skiing.py winner.pkl --no-render")
        print("\n  # 5 episodi")
        print("  python visualize_skiing.py winner.pkl --episodes 5")
        print("="*80)
        sys.exit(0)
    
    main()