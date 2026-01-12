import numpy as np

def get_action(observation):
    """
    Improved Space Invaders Agent: DODGE AND FIRE.
    Input: observation (numpy array of size 19).
    Output: Action (0, 1, 2, or 3).
    """

    # --- INPUTS ---
    center_danger = observation[3]
    target_x = observation[11]
    bullet_left = abs(observation[5]) > 0.1
    bullet_right = abs(observation[6]) > 0.1

    NOOP = 0
    FIRE = 1
    RIGHT = 2
    LEFT = 3

    # --- ACTIONS ---
    action = NOOP

    # SURVIVAL STATE (Priority #1)
    if center_danger > 0.25:  # Increased threshold for survival
        safer_side = LEFT if observation[2] < observation[4] else RIGHT
        action = safer_side
    
    # ADAPTIVE THRESHOLD FOR FIRE BASED ON TARGET DISTANCE AND NEARBY BULLETS
    firing_threshold = 0.15 + abs(target_x) * 0.1
    
    if target_x < -0.3 and not bullet_left:  # More aggressive chasing if no bullet is nearby on the left
        action = LEFT
    elif target_x > 0.3 and not bullet_right:  # More aggressive chasing if no bullet is nearby on the right
        action = RIGHT
    elif abs(target_x) < firing_threshold:
        action = FIRE
    
    return action

# Example usage:
if __name__ == "__main__":
    observation = np.array([0.5, 0.0, 0.0, 0.4, 0.5, -0.1, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    action = get_action(observation)
    print(f"Action: {action}")