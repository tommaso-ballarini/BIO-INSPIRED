import numpy as np
import random

def get_action(observation):
    """
    Initial Space Invaders Agent: RANDOM BEHAVIOR.
    Input: observation (numpy array of size 19).
    Output: Action (0, 1, 2, or 3).
    """
    
    # --- INPUTS (See config.yaml for details) ---
    # obs[0]: Player X
    # obs[1-5]: Proximity Sensors (S1..S5)
    # obs[11]: Target Alien Relative X

    # Random choice
    action = random.choice([0, 1, 2, 3])
    
    return action