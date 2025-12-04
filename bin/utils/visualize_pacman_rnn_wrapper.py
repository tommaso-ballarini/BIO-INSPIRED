# FILE: utils/visualize_best_pacman.py
"""
Script per visualizzare e testare il miglior agente NEAT su Pacman.
Compatibile con il framework OCAtari + PacmanFeatureWrapper.
"""

import os
import sys
import pickle
import numpy as np
import neat
import time

# --- Setup paths ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Monkey patch OCAtari (stesso del training)
import ocatari.core
from itertools import chain

def patched_ns_state(self):
    valid_objects = [o for o in self.objects if o is not None and hasattr(o, '_nsrepr')]
    return list(chain.from_iterable([o._nsrepr for o in valid_objects]))

ocatari.core.OCAtari.ns_state = patched_ns_state

# Import ambiente
from ocatari.core import OCAtari
from core.wrappers_pacman import PacmanFeatureWrapper


def get_latest_winner(result_dir):
    """Trova il file winner_*.pkl più recente nella cartella."""
    candidates = [f for f in os.listdir(result_dir) if f.startswith("winner_") and f.endswith(".pkl")]
    
    if not candidates:
        return None
    
    # Ordina per data di modifica (più recente prima)
    candidates.sort(
        key=lambda f: os.path.getmtime(os.path.join(result_dir, f)), 
        reverse=True
    )
    
    return os.path.join(result_dir, candidates[0])


def visualize_agent(winner_path, config_path, num_episodes=5, max_steps=5000, render=True):
    """
    Carica e testa il miglior agente NEAT su più episodi.
    
    Args:
        winner_path: Path al file winner_*.pkl
        config_path: Path al config NEAT
        num_episodes: Numero di episodi da testare
        max_steps: Massimo numero di step per episodio
        render: Se True, mostra il gioco (solo primo episodio)
    """
    
    # --- CARICAMENTO GENOMA E CONFIG ---
    print("=" * 80)
    print("🎮 VISUALIZZAZIONE AGENTE NEAT - PACMAN")
    print("=" * 80)
    print(f"📁 Winner: {os.path.basename(winner_path)}")
    print(f"📁 Config: {os.path.basename(config_path)}")
    print("=" * 80)
    
    # Carica winner
    with open(winner_path, 'rb') as f:
        winner, _ = pickle.load(f)  # (genome, config)
    
    print(f"🧬 Genome ID: {winner.key}")
    print(f"🏆 Fitness training: {winner.fitness:.2f}")
    
    # Carica config NEAT
    if not os.path.isfile(config_path):
        print(f"❌ Config non trovata: {config_path}")
        return
    
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    
    # Crea rete RNN
    net = neat.nn.RecurrentNetwork.create(winner, neat_config)
    
    # --- TEST MULTI-EPISODIO ---
    print(f"\n🔄 Test su {num_episodes} episodi...")
    print(f"⏱️  Max steps per episodio: {max_steps}")
    print("=" * 80)
    
    fitness_scores = []
    steps_survived = []
    
    for episode in range(num_episodes):
        # Renderizza solo il primo episodio
        render_mode = "human" if (render and episode == 0) else "rgb_array"
        
        # --- CREAZIONE AMBIENTE ---
        env = OCAtari(
            "Pacman",
            mode="ram",
            obs_mode="obj",
            render_mode=render_mode,
            hud=False
        )
        
        # Disabilita sticky actions
        if hasattr(env.unwrapped, 'ale'):
            env.unwrapped.ale.setFloat('repeat_action_probability', 0.0)
        
        # Applica wrapper
        env = PacmanFeatureWrapper(env, grid_rows=10, grid_cols=10)
        
        # Reset rete RNN (importante!)
        net.reset()
        
        # --- EPISODIO ---
        observation, info = env.reset()
        done = False
        steps = 0
        total_score = 0.0
        
        print(f"\n▶️  Episodio {episode + 1}/{num_episodes}", end=" ", flush=True)
        
        while not done and steps < max_steps:
            # Decisione rete
            output = net.activate(observation)
            action = np.argmax(output)
            
            # Step ambiente
            try:
                observation, reward, terminated, truncated, info = env.step(action)
                total_score += reward
                steps += 1
                done = terminated or truncated
                
                # Rendering delay (solo se human)
                if render_mode == "human":
                    time.sleep(0.01)  # 100 FPS circa
                
            except Exception as e:
                print(f"\n⚠️ Errore durante step: {e}")
                break
        
        env.close()
        
        # Salva risultati
        fitness_scores.append(total_score)
        steps_survived.append(steps)
        
        print(f"→ Score: {total_score:.0f} | Steps: {steps}")
    
    # --- STATISTICHE FINALI ---
    print("\n" + "=" * 80)
    print("📊 RISULTATI AGGREGATI")
    print("=" * 80)
    print(f"🎯 Score Medio:    {np.mean(fitness_scores):.2f} ± {np.std(fitness_scores):.2f}")
    print(f"📈 Score Min/Max:  {np.min(fitness_scores):.0f} / {np.max(fitness_scores):.0f}")
    print(f"⏱️  Steps Medi:     {np.mean(steps_survived):.0f} ± {np.std(steps_survived):.0f}")
    print(f"🏆 Fitness Training: {winner.fitness:.2f}")
    print("=" * 80)
    
    return {
        "scores": fitness_scores,
        "steps": steps_survived,
        "mean_score": np.mean(fitness_scores),
        "std_score": np.std(fitness_scores)
    }


if __name__ == "__main__":
    # --- CONFIGURAZIONE ---
    RESULT_DIR = os.path.join(PROJECT_ROOT, "evolution_results", "pacman")
    CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "neat_rnn_wr_pacman_config.txt")
    
    # Parametri test
    NUM_EPISODES = 10      # Numero di episodi di test
    MAX_STEPS = 5000      # Allineato al training
    RENDER_FIRST = True   # Mostra solo il primo episodio
    
    # --- TROVA WINNER PIÙ RECENTE ---
    if not os.path.isdir(RESULT_DIR):
        print(f"❌ Cartella {RESULT_DIR} non trovata.")
        print("   Esegui prima: python experiments/run_pacman_neat.py")
        sys.exit(1)
    
    winner_path = get_latest_winner(RESULT_DIR)
    
    if winner_path is None:
        print(f"❌ Nessun file winner_*.pkl trovato in {RESULT_DIR}")
        sys.exit(1)
    
    # --- VISUALIZZA ---
    results = visualize_agent(
        winner_path=winner_path,
        config_path=CONFIG_PATH,
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS,
        render=RENDER_FIRST
    )
    
    print("\n✨ Test completato!")
    print(f"   Per ri-testare: python utils/visualize_best_pacman.py")