import gymnasium as gym
import pygame
import sys
import os
import numpy as np
import json
import ale_py
from datetime import datetime

# --- 1. PATH SETUP & WRAPPER IMPORT ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

# Auto-locate wrapper folder
if "wrapper" not in os.listdir(current_dir):
    candidate_dir = current_dir
    while "wrapper" not in os.listdir(candidate_dir) and os.path.dirname(candidate_dir) != candidate_dir:
        candidate_dir = os.path.dirname(candidate_dir)
    if "wrapper" in os.listdir(candidate_dir):
        sys.path.append(candidate_dir)
    else:
        sys.path.append(project_root)

try:
    from wrapper.freeway_wrapper import FreewaySpeedWrapper
    print("✅ Loaded: FreewaySpeedWrapper")
except ImportError:
    print("❌ Critical: FreewaySpeedWrapper not found. Check path.")
    sys.exit(1)

# --- CONFIGURATION ---
FPS = 60
JSON_FILE = os.path.join(current_dir, "human_stats_freeway_comprehensive.json")

def load_history():
    if not os.path.exists(JSON_FILE): return []
    try:
        with open(JSON_FILE, 'r') as f: return json.load(f)
    except: return []

def save_history(history):
    with open(JSON_FILE, 'w') as f: json.dump(history, f, indent=4)

def play_game():
    """Runs a manual game calculating ALL fitness variants."""
    
    try:
        raw_env = gym.make("ALE/Freeway-v5", obs_type="ram", render_mode="human")
        env = FreewaySpeedWrapper(raw_env, normalize=True, mirror_last_5=True)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    meanings = env.unwrapped.get_action_meanings()
    try:
        up_idx = meanings.index("UP")
        down_idx = meanings.index("DOWN")
        noop_idx = meanings.index("NOOP")
    except:
        up_idx, down_idx, noop_idx = 2, 5, 0

    obs, info = env.reset()
    if not pygame.get_init(): pygame.init()
    clock = pygame.time.Clock()

    print(f"\n🎮 GAME STARTING (Comprehensive Mode)...")
    print("   [UP] Move Up | [DOWN] Move Down | [ESC] Quit")

    raw_score = 0.0
    max_y_reached = 0.0
    collision_count = 0
    steps = 0
    prev_y = obs[0] 
    
    running = True

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        action = noop_idx
        if keys[pygame.K_UP] or keys[pygame.K_w]: action = up_idx
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: action = down_idx

        obs, reward, terminated, truncated, info = env.step(action)
        
        current_y = obs[0]
        raw_score += float(reward)

        if current_y > max_y_reached:
            max_y_reached = current_y

        if current_y < prev_y - 0.05:
            collision_count += 1
            
        steps += 1
        prev_y = current_y

        # --- FITNESS FORMULAS ---
        fit_aggressive = (raw_score * 50.0) + (max_y_reached * 10.0) - (collision_count * 0.5)
        fit_balanced = (raw_score * 20.0) + (max_y_reached * 10.0) - (collision_count * 0.5)
        fit_timepen = (raw_score * 100.0) + (max_y_reached * 20.0) - (collision_count * 2.0) - (steps * 0.01)
        fit_simple = raw_score + (max_y_reached * 0.1)

        if terminated or truncated:
            running = False

        if steps % 30 == 0:
            print(f"\rNative: {raw_score:.0f} | Aggr: {fit_aggressive:.1f} | Bal: {fit_balanced:.1f} | TimePen: {fit_timepen:.1f}", end="")

    env.close()
    pygame.quit()
    
    # --- RETURN DATA (NATIVE TYPES) ---
    return {
        "raw_score": float(raw_score),
        "steps": int(steps),
        "fit_aggressive": float(fit_aggressive),
        "fit_balanced": float(fit_balanced),
        "fit_timepen": float(fit_timepen),
        "fit_simple": float(fit_simple)
    }

def main():
    print(f"\n👤 HUMAN BENCHMARK - FREEWAY (ALL METRICS)")
    print(f"   Calculating stats across ALL sessions.")
    print("-------------------------------------------------")

    # 1. Play
    stats = play_game()
    
    # 2. Save
    history = load_history()
    stats["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history.append(stats)
    save_history(history)
    
    # 3. Calculate Averages
    total_runs = len(history)
    
    # Helper to safely get mean (default 0 if key missing in old json entries)
    def get_avg(key):
        values = [g.get(key, 0.0) for g in history]
        return np.mean(values) if values else 0.0

    avg_raw = get_avg("raw_score")
    avg_aggr = get_avg("fit_aggressive")
    avg_bal = get_avg("fit_balanced")
    avg_time = get_avg("fit_timepen")
    avg_simp = get_avg("fit_simple")
    
    # 4. Report
    print(f"\n\n✅ RUN FINISHED.")
    print("=" * 60)
    print(f"📊 LIFETIME STATISTICS (Total Runs: {total_runs})")
    print("=" * 60)
    
    print(f"🔸 FITNESS Metrics (Average):")
    print(f"   1. Raw Score:      {avg_raw:.2f}  (Last: {stats['raw_score']:.0f})")
    print(f"   2. Aggressive:    {avg_aggr:.2f}  [Last: {stats['fit_aggressive']:.2f}]")
    print(f"      (Formula: Score*50 + Y*10 - Col*0.5)")
    print(f"   3. Balanced:      {avg_bal:.2f}   [Last: {stats['fit_balanced']:.2f}]")
    print(f"      (Formula: Score*20 + Y*10 - Col*0.5)")
    print(f"   4. Time Penalty:  {avg_time:.2f}  [Last: {stats['fit_timepen']:.2f}]")
    print(f"      (Formula: Score*100 + ... - Time)")
    print(f"   5. Simple:        {avg_simp:.2f}   [Last: {stats['fit_simple']:.2f}]")
    print(f"      (Formula: Score + Y*0.1)")
    print("=" * 60)
    print(f"💾 Data updated in {os.path.basename(JSON_FILE)}")

if __name__ == "__main__":
    main()