import gymnasium as gym
import pygame
import sys
import os
import numpy as np
import json
from datetime import datetime
from ocatari.core import OCAtari

# --- 1. PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

if "wrapper" not in os.listdir(current_dir):
    candidate_dir = current_dir
    while "wrapper" not in os.listdir(candidate_dir) and os.path.dirname(candidate_dir) != candidate_dir:
        candidate_dir = os.path.dirname(candidate_dir)
    if "wrapper" in os.listdir(candidate_dir):
        sys.path.append(candidate_dir)
    else:
        print("❌ Error: 'wrapper' folder not found.")
        sys.exit(1)

# Import Wrapper Specifico (Egocentric)
try:
    from wrapper.wrapper_si_ego import SpaceInvadersEgocentricWrapper
    WRAPPER_NAME = "EgoWrapper (Survival+Aim)"
except ImportError:
    print("❌ Critical: 'wrapper_si_ego.py' not found.")
    sys.exit(1)

# --- CONFIGURATION ---
FPS = 60
JSON_FILE = os.path.join(current_dir, "human_stats_si_ego.json")

def load_history():
    if not os.path.exists(JSON_FILE): return []
    try:
        with open(JSON_FILE, 'r') as f: return json.load(f)
    except: return []

def save_history(history):
    with open(JSON_FILE, 'w') as f: json.dump(history, f, indent=4)

def play_game():
    """Runs a manual game with Ego Fitness calculation."""
    
    # 1. Init Environment
    try:
        # OCAtari base
        base_env = OCAtari("ALE/SpaceInvaders-v5", mode="ram", hud=False, render_mode="human")
        # Applica Wrapper Egocentric
        env = SpaceInvadersEgocentricWrapper(base_env, skip=4)
    except Exception as e:
        print(f"❌ Error creating env: {e}")
        sys.exit(1)

    # 2. Action Mapping
    # Space Invaders: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=RIGHTFIRE, 5=LEFTFIRE
    # Mappiamo per tastiera
    action_noop = 0
    action_fire = 1
    action_right = 2 # Check if 2 is right or left in your version, usually 2=RIGHT
    action_left = 3
    
    # Verify meanings if possible
    try:
        meanings = env.unwrapped.get_action_meanings()
        action_fire = meanings.index("FIRE")
        action_right = meanings.index("RIGHT")
        action_left = meanings.index("LEFT")
    except:
        pass # Keep defaults

    env.reset()
    if not pygame.get_init(): pygame.init()
    clock = pygame.time.Clock()

    print(f"\n🎮 GAME STARTING (Space Invaders - Ego Mode)...")
    print("   [ARROWS] Move | [SPACE] Fire | [ESC] Quit")

    # Countdown
    for i in range(3, 0, -1):
        print(f"   {i}...", end="\r")
        pygame.time.wait(1000)
    print("   GO!   ")

    # Stats
    raw_score = 0.0
    custom_fitness = 0.0
    
    running = True
    steps = 0
    
    obs, info = env.reset()

    while running:
        clock.tick(FPS)

        # Input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        action = action_noop

        # Simple Logic: Move takes precedence, Space adds Fire (if combo actions exist)
        # For simplicity, separate actions or prioritize fire
        if keys[pygame.K_SPACE]:
            action = action_fire
            # If moving while firing is needed, we need combo actions (RIGHTFIRE/LEFTFIRE)
            # But let's keep it simple for mapping 1:1 with standard logic
            if keys[pygame.K_RIGHT]:
                try: action = meanings.index("RIGHTFIRE")
                except: action = action_fire
            elif keys[pygame.K_LEFT]:
                try: action = meanings.index("LEFTFIRE")
                except: action = action_fire
        elif keys[pygame.K_RIGHT]:
            action = action_right
        elif keys[pygame.K_LEFT]:
            action = action_left

        # --- FITNESS LOGIC (Replicating 6_run_ego_RNN_fit.py) ---
        # Pre-calculation based on observation BEFORE step
        if len(obs) == 19:
            danger_level = obs[3]
            is_safe = danger_level < 0.25
            
            # 1. Penalità Sparo (Anti-Spam)
            if action in [1, 4, 5]: # Fire actions
                custom_fitness -= 0.05
            
            # 2. Survival / Aiming Logic
            if is_safe:
                # Bonus Mira (rel_x < 0.15)
                rel_x = obs[11]
                if abs(rel_x) < 0.15:
                    custom_fitness += 0.02
            else:
                # Penalità Pericolo
                custom_fitness -= (danger_level * 0.2)
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 3. Reward (Kill)
        raw_score += reward # Standard Atari score
        if reward > 0:
            custom_fitness += reward
            custom_fitness += (reward * 0.5) # Bonus Extra Kill

        steps += 1
        if terminated or truncated:
            running = False

        # Live Stats
        if steps % 60 == 0:
            print(f"\r   Score: {raw_score:.0f} | Ego Fit: {custom_fitness:.2f}", end="")

    env.close()
    pygame.quit()
    
    # Min fitness cap
    if custom_fitness <= 0:
        custom_fitness = max(0.001, steps / 10000.0)

    return raw_score, custom_fitness

def main():
    print(f"\n👤 HUMAN BENCHMARK - SPACE INVADERS (EGO)")
    print(f"   Logic Reference: 6_run_ego_RNN_fit.py")
    print(f"   Storage: {os.path.basename(JSON_FILE)}")
    print("-------------------------------------------------")

    # 1. Play
    raw, custom = play_game()
    print(f"\n✅ Run Finished.")
    print(f"   Raw Score:    {raw}")
    print(f"   Custom Fit:   {custom:.2f}")

    # 2. Save
    history = load_history()
    history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_score": raw,
        "custom_fitness_ego": custom,
        "fitness_source": "6_run_ego_RNN_fit"
    })
    save_history(history)

    # 3. Report
    all_raw = [g['raw_score'] for g in history]
    all_cust = [g['custom_fitness_ego'] for g in history if 'custom_fitness_ego' in g]

    print("\n" + "="*60)
    print("📊 LIFETIME STATS (EGO MODE)")
    print("="*60)
    print(f"GAMES PLAYED: {len(history)}")
    print("-" * 30)
    if all_raw:
        print(f"RAW SCORE Avg:   {np.mean(all_raw):.2f} (Max: {np.max(all_raw)})")
    if all_cust:
        print(f"EGO FITNESS Avg: {np.mean(all_cust):.2f} (Max: {np.max(all_cust):.2f})")
    print("="*60)

if __name__ == "__main__":
    main() 