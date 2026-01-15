import numpy as np
import random

def get_action(observation):
    """
    Initial Skiing Agent: RANDOM BEHAVIOR.
    Input: observation (numpy array of size 9).
    Output: Action (0, 1, or 2).
    """
    
    # --- SENSORS (See config.yaml for details) ---
    # obs[0]: Player X
    # obs[3]: Target Delta X (The most important one!)
    
    # --- CURRENT LOGIC: RANDOM ---
    
    action = random.choice([0, 1, 2])
    
    return action