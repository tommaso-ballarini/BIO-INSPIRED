import numpy as np
import random

def get_action(observation):
    """
    Initial Freeway Agent: Random Behavior.
    Input: observation (size 22)
    Output: 0 (NOOP), 1 (UP), 2 (DOWN)
    """
    # obs[0] is Chicken Y. Goal is to maximize it.
    
    # Simple Random Baseline
    return random.choice([0, 1, 2]) 