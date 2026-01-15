import numpy as np

def get_action(observation):
    """
    Improved Space Invaders Agent: DODGE AND FIRE.
    Input: observation (numpy array of size 19).
    Output: Action (0, 1, 2, or 3).
    """

    # --- INPUTS ---
    danger_center = observation[3]
    danger_left = observation[2]
    danger_right = observation[4]
    target_x = observation[11]

    # Define constants for clarity
    NOOP = 0
    FIRE = 1
    RIGHT = 2
    LEFT = 3

    action = NOOP
    
    # SURVIVAL STATE (Priority #1)
    if danger_center > 0.4:
        safer_side = LEFT if danger_left < danger_right else RIGHT
        action = safer_side
    
    # ADAPTIVE THRESHOLD FOR FIRE BASED ON TARGET DISTANCE AND DANGER LEVEL
    firing_threshold = 0.2 + abs(target_x) * 0.15
    danger_factor = max(0.7, 1 - danger_center * 0.3)
    
    # ADJUST FIRING THRESHOLD WITH DANGER FACTOR
    firing_threshold *= danger_factor
    
    # ATTACK STATE (Priority #2)
    if abs(target_x) < firing_threshold:
        action = FIRE
    elif target_x < -0.4:  # Chase more aggressively if target is too far left
        action = LEFT
    elif target_x > 0.4:  # Chase more aggressively if target is too far right
        action = RIGHT
    
    return action

# Example usage:
if __name__ == "__main__":
    observation = np.array([0.5, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    action = get_action(observation)
    print(f"Action: {action}")