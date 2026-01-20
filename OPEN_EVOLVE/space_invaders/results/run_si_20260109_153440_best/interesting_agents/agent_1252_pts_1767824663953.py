import numpy as np

def get_action(observation):
    """
    Improved Space Invaders Agent: DODGE AND FIRE.
    Input: observation (numpy array of size 19).
    Output: Action (0, 1, 2, or 3).
    """

    # --- INPUTS ---
    center_danger = observation[3]
    left_danger = observation[2]
    right_danger = observation[4]
    target_x = observation[11]

    NOOP = 0
    FIRE = 1
    RIGHT = 2
    LEFT = 3

    # --- ACTIONS ---
    action = NOOP
    
    # SURVIVAL STATE (Priority #1)
    if center_danger > 0.4:  # Higher threshold for survival to ensure dodging is prioritized
        safer_side = LEFT if left_danger < right_danger else RIGHT
        action = safer_side
    
    # ADAPTIVE THRESHOLD FOR FIRE TO IMPROVE ACCURACY AND AVOID SHOTGUNNING
    fire_threshold = 0.15 + abs(target_x) * 0.2  # Adjusted adaptive threshold based on target distance to ensure better alignment

    # CHECK FOR NEARBY BULLETS TO ADJUST FIRE THRESHOLD
    nearby_bullets_left = observation[1] < -0.1
    nearby_bullets_right = observation[5] > 0.1
    
    if nearby_bullets_left:
        fire_threshold -= 0.05
    elif nearby_bullets_right:
        fire_threshold += 0.05
    
    # ATTACK STATE (Priority #2)
    if abs(target_x) < fire_threshold:
        action = FIRE
    elif target_x < -0.3:  # Chase more aggressively if target is too far left
        action = LEFT
    elif target_x > 0.3:  # Chase more aggressively if target is too far right
        action = RIGHT
    
    return action

# Test the function with a sample observation
observation = np.array([0.5, 0.0, 0.0, 0.4, 0.5, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
action = get_action(observation)
print(f"Action: {action}")