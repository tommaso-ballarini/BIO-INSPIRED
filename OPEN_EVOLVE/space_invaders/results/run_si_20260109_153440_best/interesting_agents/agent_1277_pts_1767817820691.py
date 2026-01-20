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

    # Constants for clarity and tuning
    NOOP = 0
    FIRE = 1
    RIGHT = 2
    LEFT = 3

    action = NOOP
    
    # SURVIVAL STATE (Priority #1)
    if center_danger > 0.25:  # Increased threshold for survival to ensure dodging is prioritized
        safer_side = LEFT if left_danger < right_danger else RIGHT
        if not (left_danger == 1 and right_danger == 1):  # Only move if there's space to do so
            action = safer_side
    
    # ADAPTIVE THRESHOLD FOR FIRE BASED ON TARGET DISTANCE AND BULLETS
    fire_threshold = 0.15 + abs(target_x) * 0.1  # Adaptive threshold based on target distance to ensure better alignment
    
    nearby_bullets = observation[1] < -0.1 or observation[5] > 0.1
    if nearby_bullets:
        fire_threshold -= 0.05  # Reduce firing threshold if bullets are close
    
    # Additional heuristic: Avoid shooting when target is too far away and not directly in range
    if abs(target_x) > fire_threshold * 2:
        action = NOOP
    
    # ATTACK STATE (Priority #2)
    if abs(target_x) < fire_threshold:
        action = FIRE
    elif target_x < -0.3:  # Chase more aggressively if target is too far left
        action = LEFT
    elif target_x > 0.3:  # Chase more aggressively if target is too far right
        action = RIGHT
    
    return action

# Test the function with a sample observation
observation = np.array([0.5, 0.0, 0.0, 0.4, 0.5, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
action = get_action(observation)
print(f"Action: {action}")