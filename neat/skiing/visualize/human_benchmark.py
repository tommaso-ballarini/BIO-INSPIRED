import gymnasium as gym
import pygame
import sys
import os
import numpy as np
import ale_py
import json
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
        print("❌ Error: 'wrapper' folder not found.")
        sys.exit(1)

try:
    from wrapper.wrapper_rnn import BioSkiingOCAtariWrapper
    WRAPPER_NAME = "RNN Wrapper"
except ImportError:
    try:
        from wrapper.wrapper_ffnn import BioSkiingOCAtariWrapper
        WRAPPER_NAME = "FFNN Wrapper"
    except ImportError:
        print("❌ Critical: Wrapper not found.")
        sys.exit(1)

# --- CONFIGURATION ---
FPS = 60
JSON_FILE = os.path.join(current_dir, "human_stats.json")

def load_history():
    """Loads existing game history from JSON."""
    if not os.path.exists(JSON_FILE):
        return []
    try:
        with open(JSON_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_history(history):
    """Saves updated history to JSON."""
    with open(JSON_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def play_game():
    """Runs a single manual game."""
    try:
        env = BioSkiingOCAtariWrapper(render_mode="human")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    env.reset()
    if not pygame.get_init():
        pygame.init()
    
    clock = pygame.time.Clock()
    
    print(f"\n🎮 GAME STARTING...")
    print("   [ARROWS] Steer | [ESC] Give Up")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"   {i}...", end="\r")
        pygame.time.wait(1000)
    print("   GO!   ")

    run_native = 0.0
    run_custom = 0.0
    running = True
    step = 0
    
    while running:
        clock.tick(FPS) # Stable 60 FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False 

        keys = pygame.key.get_pressed()
        action = 0 

        if keys[pygame.K_RIGHT]: action = 1
        elif keys[pygame.K_LEFT]: action = 2
        elif keys[pygame.K_DOWN]: action = 0 

        obs, reward, terminated, truncated, info = env.step(action)
        
        run_custom += reward
        run_native += info.get('native_reward', 0.0)
        step += 1

        if terminated or truncated:
            running = False

        if step % 60 == 0:
            print(f"\r   Score: {run_native:.0f} | Custom: {run_custom:.1f}", end="")

    env.close()
    pygame.quit()
    
    return run_native, run_custom

def main():
    print(f"\n👤 HUMAN BENCHMARK (Persistent Mode)")
    print(f"   History File: {os.path.basename(JSON_FILE)}")
    print("-------------------------------------------------")

    # 1. Play One Game
    native, custom = play_game()
    print(f"\n✅ Run Finished. Native: {native} | Custom: {custom:.1f}")

    # 2. Update History
    history = load_history()
    
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "native": native,
        "custom": custom
    }
    history.append(new_entry)
    save_history(history)

    # 3. Calculate Cumulative Stats
    all_natives = [g['native'] for g in history]
    all_customs = [g['custom'] for g in history]

    avg_native = np.mean(all_natives)
    std_native = np.std(all_natives)
    avg_custom = np.mean(all_customs)
    std_custom = np.std(all_customs)
    
    best_native = max(all_natives)
    best_custom = max(all_customs)

    # 4. Report
    print("\n" + "="*60)
    print("📊 UPDATED LIFETIME STATS")
    print("="*60)
    print(f"TOTAL GAMES PLAYED: {len(history)}")
    print("-" * 30)
    print(f"AVERAGE:")
    print(f"  ❄️  Native Score:   {avg_native:.2f} (± {std_native:.2f})")
    print(f"  🎯  Custom Fitness: {avg_custom:.2f} (± {std_custom:.2f})")
    print("-" * 30)
    print(f"RECORDS:")
    print(f"  Best Native: {best_native}")
    print(f"  Best Custom: {best_custom:.1f}")
    print("="*60)

if __name__ == "__main__":
    main()