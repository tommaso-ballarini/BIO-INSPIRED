import numpy as np

def get_action(observation):    
    # The current logic is "dumb": it chooses a random action.
    # Modify this part.
    return np.random.choice([0, 1, 2])