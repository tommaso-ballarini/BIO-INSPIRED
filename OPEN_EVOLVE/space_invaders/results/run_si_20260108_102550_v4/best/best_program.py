import numpy as np

def get_action(observation):
    """
    Improved Space Invaders Agent: DODGE AND FIRE.
    Input: observation (numpy array of size 19).
    Output: Action (0, 1, 2, or 3).
    """
    
    # --- ACTIONS ---
    NOOP = 0
    FIRE = 1
    RIGHT = 2
    LEFT = 3
    
    # ** SURVIVAL STATE (Priority #1) **
    center_danger = observation[3]
    if center_danger > 0.5:  # Higher threshold to prioritize movement when danger is significant
        if observation[2] < observation[4]:  # Left is safer
            return LEFT
        else:
            return RIGHT
    
    # ** ATTACK STATE (Priority #2) **
    target_alignment = abs(observation[11])
    if target_alignment < 0.3:  # Lower threshold to encourage firing when aligned with the target
        return FIRE
    elif observation[11] < 0:  # Target is to the left
        return LEFT
    else:  # Target is to the right
        return RIGHT
    
    return NOOP  # Default action if no specific condition met