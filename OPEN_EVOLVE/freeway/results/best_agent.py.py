import numpy as np
import random

def get_action(observation):
    """
    Initial Agent: RANDOM BEHAVIOR.
    Goal: The evolutionary engine must replace this with smart logic.
    """
    
    # --- SENSORS (Available for the LLM to use) ---
    # Chicken Vertical Position (0.0 = Start, 1.0 = Goal)
    chicken_y = observation[0]
    
    # Cars Horizontal Positions (Lane 1 to 10)
    # Value is normalized X pos. If close to chicken X (about 0.25), collision risk!
    cars = observation[1:11] 
    
    # --- CURRENT LOGIC: RANDOM ---
    # Action 0: Stay/Idle
    # Action 1: Up (Forward)
    # Action 2: Down (Backward)
    
    current_lane = int(observation[0] * 10)
    current_lane = max(0, min(current_lane, 9))
    danger = observation[1 + current_lane]
    
    if danger < 0.15:
        action = 0  # Wait
    elif danger > 0.95:
        action = 1  # Move Up
    else:
        action = 1  # Move Up
    
    return action