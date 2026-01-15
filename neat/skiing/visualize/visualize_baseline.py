import os
import sys
import neat
import pickle
import numpy as np
import gymnasium as gym
from glob import glob
import time
from tqdm import tqdm

# --- PATH SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Try importing Custom Wrapper
try:
    from wrapper.wrapper import BioSkiingOCAtariWrapper
    WRAPPER_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: Wrapper not found. Custom fitness will equal raw score.")
    WRAPPER_AVAILABLE = False

# --- CONFIGURATION ---
RESULTS_DIR = os.path.join(project_root, "evolution_results", "baseline_run")
CONFIG_FILENAME = "config_baseline.txt"
TEST_SEEDS = range(100)  # Seeds 0-99

def load_latest_winner():
    """Finds the most recent pickle file."""
    if not os.path.exists(RESULTS_DIR):
        print(f"❌ Error: Directory {RESULTS_DIR} not found.")
        sys.exit(1)

    all_files = glob(os.path.join(RESULTS_DIR, "*.pkl"))
    if not all_files:
        print(f"❌ No .pkl files found in {RESULTS_DIR}")
        sys.exit(1)
    
    latest_file = max(all_files, key=os.path.getctime)
    print(f"📂 Loading genome: {os.path.basename(latest_file)}")
    
    with open(latest_file, 'rb') as f:
        data = pickle.load(f)
    return data[0] if isinstance(data, tuple) else data

def run_simulation(genome, config, seed, render=False):
    """Runs a full episode. Returns (native_score, custom_fitness)."""
    
    # 1. Environment Setup
    render_mode = "human" if render else None
    
    if WRAPPER_AVAILABLE:
        try:
            env = BioSkiingOCAtariWrapper(render_mode=render_mode)
        except:
            env = BioSkiingOCAtariWrapper(render_mode=None)
    else:
        env = gym.make("ALE/Skiing-v5", obs_type="ram", render_mode=render_mode)

    # 2. Reset
    observation, info = env.reset(seed=seed)
    
    # 3. Network Setup
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    done = False
    total_native = 0.0
    total_custom = 0.0
    
    # Check input expectations (RAM vs Wrapper)
    input_dim = config.genome_config.num_inputs
    use_ram_override = (input_dim == 128) and WRAPPER_AVAILABLE

    try:
        while not done:
            # --- Input Handling ---
            if use_ram_override:
                ram = env.env._env.unwrapped.ale.getRAM()
                inputs = ram / 255.0
            elif not WRAPPER_AVAILABLE:
                inputs = observation / 255.0
            else:
                inputs = observation

            # --- Action ---
            output = net.activate(inputs)
            action = np.argmax(output)
            
            # --- Step ---
            observation, reward, terminated, truncated, info = env.step(action)
            
            # --- Data Collection ---
            if WRAPPER_AVAILABLE:
                total_custom += reward
                total_native += info.get('native_reward', 0.0)
            else:
                total_native += reward
                total_custom += reward

            done = terminated or truncated
            
            if render:
                time.sleep(0.005)

    except Exception as e:
        print(f"Sim Error (Seed {seed}): {e}")
    finally:
        env.close()

    return total_native, total_custom

def main():
    # 1. Setup
    genome = load_latest_winner()
    config_path = os.path.join(project_root, "config", CONFIG_FILENAME)
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    print(f"\n🚀 STARTING BENCHMARK ({len(TEST_SEEDS)} SEEDS)...")
    print("=" * 60)

    results = []
    
    # 2. Benchmark Loop
    iterator = tqdm(TEST_SEEDS, desc="Running") if 'tqdm' in sys.modules else TEST_SEEDS
    
    for seed in iterator:
        native, custom = run_simulation(genome, config, seed, render=False)
        results.append({
            'seed': seed,
            'native': native,
            'custom': custom
        })

    # 3. Analysis
    native_scores = [r['native'] for r in results]
    custom_scores = [r['custom'] for r in results]

    avg_native = np.mean(native_scores)
    avg_custom = np.mean(custom_scores)
    
    completed_runs = [r for r in results if r['custom'] > 9500]
    
    if completed_runs:
        best_overall_run = max(completed_runs, key=lambda x: x['native'])
        best_label = "🏆 BEST COMPLETED RUN (Custom > 9500)"
    else:
        best_overall_run = max(results, key=lambda x: x['custom'])
        best_label = "⚠️ BEST CUSTOM (No run > 9500 found)"

    worst_native_run = min(results, key=lambda x: x['native'])

    # 4. Final Report
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    print(f"AVERAGE ({len(TEST_SEEDS)} runs):")
    print(f"  ❄️  Native Score:   {avg_native:.2f}  (± {np.std(native_scores):.2f})")
    print(f"  🎯  Custom Fitness: {avg_custom:.2f}  (± {np.std(custom_scores):.2f})")
    print("-" * 30)
    
    print("🏅 HALL OF FAME:")
    print(f"  {best_label}:")
    print(f"     -> Seed {best_overall_run['seed']}")
    print(f"     -> Native: {best_overall_run['native']:.1f}")
    print(f"     -> Custom: {best_overall_run['custom']:.1f}")
    print("-" * 30)
    
    print(f"💩 WORST RUN (Seed {worst_native_run['seed']}): {worst_native_run['native']:.1f}")
    print("="*60)

    # 5. Visualization Prompt
    choice = input(f"\n🎥 Watch the BEST COMPLETED game (Seed {best_overall_run['seed']})? [y/N]: ").strip().lower()
    if choice == 'y':
        print(f"Replaying Seed {best_overall_run['seed']}...")
        run_simulation(genome, config, best_overall_run['seed'], render=True)
        print("Done.")
    else:
        print("Skipping visualization.")

if __name__ == "__main__":
    main()