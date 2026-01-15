import sys
import os
import time
import importlib.util
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
AGENT_PATH = r"OPEN_EVOLVE\skiing\results\run_skiing_20260108_195559_seedbest\best\best_program.py" # PASTE HERE YOUR EXACT AGENT PATH 
VISUALIZATION_SEED = 61 
TEST_SEEDS_RANGE = range(0, 100) # The 100 "unseen" seeds (Test Set)
MAX_STEPS = 2000
FINISH_LINE_THRESHOLD = 9500.0 # Soglia per considerare la pista "finita"

# --- WRAPPER IMPORT SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))
sys.path.append(os.path.join(current_dir, 'wrapper'))

try:
    from skiing_wrapper import SkiingOCAtariWrapper
except ImportError:
    try:
        from src.skiing_wrapper import SkiingOCAtariWrapper
    except ImportError as e:
        print(f"❌ Error: Cannot import SkiingOCAtariWrapper. {e}")
        sys.exit(1)

def load_agent(path):
    if not os.path.exists(path):
        print(f"❌ Agent file not found: {path}")
        sys.exit(1)
    
    spec = importlib.util.spec_from_file_location("best_agent", path)
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    
    if not hasattr(agent_module, 'get_action'):
        print("❌ Agent missing 'get_action(observation)' function!")
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
            except Exception:
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
        print(f"Critical simulation error (Seed {seed}): {e}")
        return -9999, -9999

def main():
    print(f"\n❄️  TESTING AGENT: {os.path.basename(AGENT_PATH)} ❄️")
    print("="*60)
    
    get_action = load_agent(AGENT_PATH)
    print("✅ Agent loaded.")
    
    # 1. Initial Preview
    print(f"🎥  PREVIEW (Seed {VISUALIZATION_SEED})...")
    run_simulation(get_action, seed=VISUALIZATION_SEED, render=True)
    print("-" * 60)
    
    # 2. Benchmark on 100 Seeds
    print(f"\n📊  BENCHMARK ON {len(TEST_SEEDS_RANGE)} SEEDS (Generalization)...")
    
    results = [] 
    
    for seed in tqdm(TEST_SEEDS_RANGE, desc="Simulating", unit="game"):
        n_score, c_score = run_simulation(get_action, seed=seed, render=False)
        
        # --- NUOVA LOGICA DI PUNTEGGIO ---
        has_finished = c_score > FINISH_LINE_THRESHOLD
        
        # Calcoliamo un "Native Score Aggiustato" per penalizzare chi si blocca
        # Se non finisci, ti prendi -30.000 punti nativi extra.
        if not has_finished:
            adjusted_native = n_score - 30000.0
        else:
            adjusted_native = n_score

        results.append({
            'seed': seed,
            'native': n_score,          # Punteggio reale visualizzato
            'adjusted_native': adjusted_native, # Punteggio per ordinamento
            'custom': c_score,
            'finished': has_finished
        })
        
    # 3. Statistical Analysis
    native_scores = [r['native'] for r in results]
    custom_scores = [r['custom'] for r in results]
    finished_count = sum(r['finished'] for r in results)
    
    avg_native = np.mean(native_scores)
    avg_custom = np.mean(custom_scores)
    
    # --- 4. FIND BEST SEEDS (Nuova Logica Intelligente) ---
    
    # Filtriamo solo le run che hanno completato la pista
    completed_runs = [r for r in results if r['finished']]
    
    if completed_runs:
        # SCENARIO A: Qualcuno ha finito!
        # Vince chi ha il miglior Native Score (tempo minore) tra quelli che hanno finito.
        # Nota: In skiing -10000 è meglio di -20000, quindi usiamo max()
        best_run = max(completed_runs, key=lambda x: x['native'])
        best_label = "🏆 BEST COMPLETED RUN (Speed King)"
    else:
        # SCENARIO B: Nessuno ha finito (tutti incastrati o lenti)
        # Vince chi ha la Custom Fitness più alta (chi è arrivato più lontano)
        best_run = max(results, key=lambda x: x['custom'])
        best_label = "⚠️ BEST AVAILABLE (No finish found)"

    # Per il peggiore, usiamo il native score "aggiustato" (quindi chi si è bloccato finisce qui)
    worst_run = min(results, key=lambda x: x['adjusted_native'])

    print("\n" + "="*60)
    print("📈  FINAL RESULTS")
    print("="*60)
    print(f"COMPLETION RATE: {finished_count}/{len(TEST_SEEDS_RANGE)} runs finished (> {FINISH_LINE_THRESHOLD})")
    print(f"OVERALL AVERAGE:")
    print(f"   Native Score:   {avg_native:.2f} (± {np.std(native_scores):.2f})")
    print(f"   Custom Fitness: {avg_custom:.2f} (± {np.std(custom_scores):.2f})")
    print("-" * 30)
    
    print("💎  HALL OF FAME:")
    print(f"   {best_label}:")
    print(f"     -> Seed {best_run['seed']}")
    print(f"     -> Native: {best_run['native']:.1f}")
    print(f"     -> Custom: {best_run['custom']:.1f}")
    print("-" * 30)
    
    print("💀  HALL OF SHAME:")
    print(f"   💩 WORST RUN (Seed {worst_run['seed']}):")
    print(f"     -> Native: {worst_run['native']:.1f}")
    print(f"     -> Status: {'Finished' if worst_run['finished'] else 'Stuck/Slow'}")
    
    print("="*60)
    
    # 5. Interactive Replay
    choice = input(f"\nWatch the BEST run (Seed {best_run['seed']})? [y/n]: ")
    if choice.lower() == 'y':
        print(f"Replaying Seed {best_run['seed']}...")
        run_simulation(get_action, seed=best_run['seed'], render=True)

if __name__ == "__main__":
    main()