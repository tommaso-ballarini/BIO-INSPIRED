import numpy as np

def get_action(observation):
    """
    Strategy: Go UP, but try to look at RAM.
    """
    # Let's look at a random byte to 'simulate' sensing
    sensor = observation[14] 
    
    # Currently ignoring the sensor and just going UP (1)
    # TODO: The AI needs to change this logic to use 'sensor'
    return 1