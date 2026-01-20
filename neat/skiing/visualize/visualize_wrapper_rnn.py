import os
import sys
import neat
import pickle
import numpy as np
import gymnasium as gym
from glob import glob
import time
from tqdm import tqdm 

# --- 1. PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from wrapper.wrapper import BioSkiingOCAtariWrapper
    print("✅ BioSkiingOCAtariWrapper imported successfully.")
except ImportError:
    print("❌ CRITICAL: 'wrapper/wrapper.py' not found.")
    sys.exit(1)

# --- CONFIGURATION ---
RESULTS_DIR = os.path.join(project_root, "evolution_results", "wrapper_rnn_run")
CONFIG_PATH = "config_wrapper_rnn.txt"
TEST_SEEDS = range(100)  # Benchmark on seeds 0 to 99

def get_latest_winner():
    """Finds the most recent pickle file."""
    if not os.path.exists(RESULTS_DIR):
        print(f" Error: Directory {RESULTS_DIR} not found.")
        sys.exit(1)

    all_files = glob(os.path.join(RESULTS_DIR, "*.pkl"))
    if not all_files:
        print(f" No .pkl files found in {RESULTS_DIR}")
        sys.exit(1)
    
    latest_file = max(all_files, key=os.path.getctime)
    print(f" Loading genome from: {os.path.basename(latest_file)}")
    
    with open(latest_file, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, tuple):
        for item in data:
            if hasattr(item, 'connections'): 
                return item
        
        return data[0]

    elif isinstance(data, str):
        sys.exit(1)

    elif hasattr(data, 'connections'):
        return data

    elif hasattr(data, 'best_genome'):
        return data.best_genome
    else:
        sys.exit(1)

def run_simulation(genome, config, seed, render=False):
    """
    Runs a single episode using a Recurrent Network.
    Returns: (native_score, custom_fitness)
    """
    # 1. Environment Setup
    render_mode = "human" if render else None
    
    try:
        env = BioSkiingOCAtariWrapper(render_mode=render_mode)
    except Exception as e:
        if render: print(f"⚠️ Render warning: {e}")
        env = BioSkiingOCAtariWrapper(render_mode=None)

    observation, info = env.reset(seed=seed)
    
    # 2. Network Setup 
    net = neat.nn.RecurrentNetwork.create(genome, config)

    done = False
    total_custom_fitness = 0.0
    total_native_score = 0.0
    steps = 0
    
    try:
        while not done:
            inputs = observation
            
            output = net.activate(inputs)
            action = np.argmax(output) 
            
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_custom_fitness += reward
            
            native_r = info.get('native_reward', 0.0)
            total_native_score += native_r
            
            done = terminated or truncated
            is_stuck = truncated
            steps += 1
            
            if render and steps % 60 == 0:
                target_status = "SEARCHING..."
                if len(inputs) > 5 and inputs[5] > 0.5:
                    target_status = f"TARGET LOCKED (Dist: {inputs[4]:.2f})"
                
                sys.stdout.write(f"\rStep {steps} | Fitness: {total_custom_fitness:.1f} | {target_status}   ")
                sys.stdout.flush()
                time.sleep(0.01) 

    except KeyboardInterrupt:
        if render: print("\nStopped by user.")
    except Exception as e:
        print(f"Simulation Error (Seed {seed}): {e}")
    finally:
        env.close()

    return total_native_score, total_custom_fitness, is_stuck
def main():
    # 1. Setup
    genome = get_latest_winner()
    config_path = os.path.join(project_root, "config", CONFIG_PATH)
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    print(f"\n🚀 STARTING BENCHMARK ({len(TEST_SEEDS)} SEEDS)...")
    print("=" * 60)

    results = []
    
    # 2. Benchmark Loop
    iterator = tqdm(TEST_SEEDS, desc="Running") if 'tqdm' in sys.modules else TEST_SEEDS
    
    for seed in iterator:
        native, custom, stuck = run_simulation(genome, config, seed, render=False)
        
        is_actually_stuck = stuck and (custom < 1000.0)

        if is_actually_stuck:
            adjusted_native = native - 30000.0
            final_stuck_status = True
        else:
            adjusted_native = native
            final_stuck_status = False

        results.append({
            'seed': seed,
            'native': adjusted_native,
            'real_native': native,
            'custom': custom,
            'stuck': final_stuck_status
        })

    # --- 3. Analysis  ---
    real_native_scores = [r['real_native'] for r in results]
    custom_scores = [r['custom'] for r in results]
    stuck_count = sum(r['stuck'] for r in results)

    avg_native = np.mean(real_native_scores)
    avg_custom = np.mean(custom_scores)

    perfect_runs = [r for r in results if r['custom'] > 9700]
    num_perfect = len(perfect_runs)
    
    if num_perfect > 0:
        perfect_raw_scores = [r['real_native'] for r in perfect_runs]
        
        avg_native_perfect = np.mean(perfect_raw_scores)
        std_native_perfect = np.std(perfect_raw_scores)
        
        best_overall_run = max(perfect_runs, key=lambda x: x['real_native'])
        best_label = f"🏆 BEST PERFECT RUN (>9700)"
    else:
        avg_native_perfect = 0.0
        std_native_perfect = 0.0
        best_overall_run = max(results, key=lambda x: x['custom'])
        best_label = "⚠️ BEST AVAILABLE (No perfect run found)"

    worst_native_run = min(results, key=lambda x: x['native'])

    # --- 4. Final Report  ---
    print("\n" + "="*60)
    print(" FINAL RESULTS (RNN)")
    print("="*60)
    print(f"Total Runs: {len(TEST_SEEDS)} | Stuck/Truncated: {stuck_count}")
    
    print("-" * 30)
    print("🎯 PERFECT RUNS FILTER (> 9700 fitness):")
    print(f"   -> Count: {num_perfect} / {len(TEST_SEEDS)} samples")
    if num_perfect > 0:
        print(f"   -> Avg Raw Score: {avg_native_perfect:.2f} (± {std_native_perfect:.2f})")
    else:
        print(f"   -> Avg Raw Score: N/A (0 runs passed the filter)")
    print("-" * 30)

    print(f"GLOBAL AVERAGE PERFORMANCE (All Runs):")
    print(f"   ❄️  Native Score:   {avg_native:.2f}  (± {np.std(real_native_scores):.2f})")
    print(f"   🎯  Custom Fitness: {avg_custom:.2f}  (± {np.std(custom_scores):.2f})")
    print("-" * 30)
    
    print("🏅 HALL OF FAME:")
    print(f"  {best_label}:")
    print(f"    -> Seed {best_overall_run['seed']}")
    print(f"    -> Native: {best_overall_run['real_native']:.1f}")
    print(f"    -> Custom: {best_overall_run['custom']:.1f}")
    
    print("-" * 30)
    print(f"💩 WORST RUN (Seed {worst_native_run['seed']}): {worst_native_run['real_native']:.1f}")
    print("="*60)

    choice = input(f"\n🎥 Watch the BEST RUN (Seed {best_overall_run['seed']})? [y/N]: ").strip().lower()
    if choice == 'y':
        print(f"Replaying Seed {best_overall_run['seed']}...")
        run_simulation(genome, config, best_overall_run['seed'], render=True)
        print("Done.")
    else:
        print("Skipping visualization.")


if __name__ == "__main__":
    main()