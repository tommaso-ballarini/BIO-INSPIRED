import numpy as np

def get_action(observation):
    """
    Improved Space Invaders Agent: DODGE AND FIRE.
    Input: observation (numpy array of size 19).
    Output: Action (0, 1, 2, or 3).
    """

    # --- INPUTS ---
    s_left = observation[1]
    s_near_left = observation[2]
    s_center = observation[3]
    s_near_right = observation[4]
    s_far_right = observation[5]
    target_x = observation[11]

    # --- ACTIONS ---
    NOOP = 0
    FIRE = 1
    RIGHT = 2
    LEFT = 3

    action = NOOP
    
    # SURVIVAL STATE (Priority #1)
    if s_center > 0.3 or s_near_left > 0.3 or s_near_right > 0.3:
        safer_side = LEFT if s_near_left < s_near_right else RIGHT
        action = safer_side
    
    # ADAPTIVE THRESHOLD FOR FIRE BASED ON TARGET DISTANCE AND BULLET PROXIMITY
    firing_threshold = 0.15 + abs(target_x) * 0.1
    nearby_bullets = observation[1] < -0.1 or observation[5] > 0.1

    if nearby_bullets:
        firing_threshold *= 0.75  # Reduce firing threshold if bullets are close
    
    # AGGRESSIVE CHASING LOGIC (Priority #2)
    if abs(target_x) < firing_threshold:
        action = FIRE
    elif target_x < -0.3:  # More aggressive chase when far left
        action = LEFT
    elif target_x > 0.3:  # More aggressive chase when far right
        action = RIGHT
    
    return action

# Example usage
if __name__ == "__main__":
    observation = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    action = get_action(observation)
    print(f"Action: {action}")