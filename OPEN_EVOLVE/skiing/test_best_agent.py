import sys
import os
import time
import importlib.util
import numpy as np
from tqdm import tqdm

# --- CONFIGURAZIONE ---
# NOTA: Usa gli slash normali (/) anche su Windows per evitare errori
AGENT_PATH = "OPEN_EVOLVE/skiing/results/run_skiing_20260109_120738/best/best_program.py"
VISUALIZATION_SEED = 8
TEST_SEEDS_RANGE = range(0, 100) # I 100 seed "non visti" (Test Set)
MAX_STEPS = 2000

# --- SETUP IMPORT WRAPPER ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))
sys.path.append(os.path.join(current_dir, 'wrapper'))

try:
    from skiing_wrapper import SkiingOCAtariWrapper
except ImportError:
    try:
        from src.skiing_wrapper import SkiingOCAtariWrapper
    except ImportError as e:
        print(f"❌ Errore: Impossibile importare SkiingOCAtariWrapper. {e}")
        sys.exit(1)

def load_agent(path):
    if not os.path.exists(path):
        print(f"❌ File agente non trovato: {path}")
        sys.exit(1)
    
    spec = importlib.util.spec_from_file_location("best_agent", path)
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    
    if not hasattr(agent_module, 'get_action'):
        print("❌ L'agente non ha la funzione 'get_action(observation)'!")
        sys.exit(1)
        
    return agent_module.get_action

def run_simulation(action_func, seed, render=False):
    render_mode = "human" if render else None
    try:
        env = SkiingOCAtariWrapper(render_mode=render_mode)
        obs, info = env.reset(seed=seed)
        
        total_native_score = 0.0
        total_custom_fitness = 0.0
        steps = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and steps < MAX_STEPS:
            try:
                action = int(action_func(obs))
            except Exception as e:
                break
            
            obs, custom_reward, terminated, truncated, info = env.step(action)
            total_custom_fitness += custom_reward
            native_reward = info.get('native_reward', 0.0)
            total_native_score += native_reward
            steps += 1
            
            if render: time.sleep(0.01)
                
        env.close()
        return total_native_score, total_custom_fitness
        
    except Exception as e:
        print(f"Errore critico simulazione (Seed {seed}): {e}")
        return -9999, -9999

def main():
    print(f"\n❄️  TESTING AGENT: {os.path.basename(AGENT_PATH)} ❄️")
    print("="*60)
    
    get_action = load_agent(AGENT_PATH)
    print("✅ Agente caricato.")
    
    # 1. Visualizzazione Iniziale
    print(f"🎥  PREVIEW (Seed {VISUALIZATION_SEED})...")
    run_simulation(get_action, seed=VISUALIZATION_SEED, render=True)
    print("-" * 60)
    
    # 2. Benchmark su 100 Seed
    print(f"\n📊  BENCHMARK SU {len(TEST_SEEDS_RANGE)} SEED (Generalizzazione)...")
    
    detailed_results = [] # Lista di dizionari per tracciare seed e punteggi
    
    for seed in tqdm(TEST_SEEDS_RANGE, desc="Simulando", unit="game"):
        n_score, c_score = run_simulation(get_action, seed=seed, render=False)
        
        # Salviamo tutto il pacchetto dati
        detailed_results.append({
            'seed': seed,
            'native': n_score,
            'custom': c_score
        })
        
    # 3. Analisi Statistica
    native_scores = [r['native'] for r in detailed_results]
    custom_scores = [r['custom'] for r in detailed_results]
    
    avg_native = np.mean(native_scores)
    avg_custom = np.mean(custom_scores)
    
    # --- 4. TROVA I MIGLIORI SEED (Nuova Logica) ---
    # Nota: In Skiing, Native score è negativo (es. -5000). 
    # Quindi -3000 > -5000. Usiamo max() per trovare il migliore.
    
    best_native_run = max(detailed_results, key=lambda x: x['native'])
    best_custom_run = max(detailed_results, key=lambda x: x['custom'])
    worst_native_run = min(detailed_results, key=lambda x: x['native'])

    print("\n" + "="*60)
    print("📈  RISULTATI FINALI")
    print("="*60)
    print(f"MEDIA GENERALE:")
    print(f"   Native Score:  {avg_native:.2f} (± {np.std(native_scores):.2f})")
    print(f"   Custom Fitness: {avg_custom:.2f} (± {np.std(custom_scores):.2f})")
    print("-" * 30)
    
    print("💎  HALL OF FAME (I Migliori Seed Trovati):")
    print(f"   🏆 BEST SPEED (Native): Seed {best_native_run['seed']} -> Score {best_native_run['native']:.1f}")
    print(f"   🧠 BEST TECH (Custom):  Seed {best_custom_run['seed']} -> Score {best_custom_run['custom']:.1f}")
    print("-" * 30)
    
    print("💀  HALL OF SHAME (Il Peggiore Seed):")
    print(f"   💩 WORST RUN:           Seed {worst_native_run['seed']} -> Score {worst_native_run['native']:.1f}")
    
    print("="*60)
    
    # 5. Domanda Interattiva Finale
    choice = input(f"\nVuoi rivedere la partita MIGLIORE (Seed {best_native_run['seed']})? [y/n]: ")
    if choice.lower() == 'y':
        print(f"Rilancio Seed {best_native_run['seed']}...")
        run_simulation(get_action, seed=best_native_run['seed'], render=True)

if __name__ == "__main__":
    main()