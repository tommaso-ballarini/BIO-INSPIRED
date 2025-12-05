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
    # Value is normalized X pos. If close to chicken X (usually center), collision risk!
    cars = observation[1:11] 
    
    # --- CURRENT LOGIC: RANDOM ---
    # Action 0: Stay/Idle
    # Action 1: Up (Forward)
    # Action 2: Down (Backward)
    
    action = random.choice([0, 1, 2])
    
    return action