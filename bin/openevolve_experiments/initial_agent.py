import numpy as np

def get_action(observation):
    """
    Scaffolding Agent: Structure ready, logic missing.
    Current behavior: KAMIKAZE (Always runs forward).
    """
    
    # 1. SENSOR EXTRACTION (The LLM will use these ready-made variables)
    chicken_y = observation[0]
    
    # Nearby cars 
    car_lane_1 = observation[1] 
    car_lane_2 = observation[2] 
    #other cars can be added similarly
    
    # 2. MUTABLE PARAMETERS (Numbers that evolution loves to change)
    # Currently set effectively to 0.0 to have no effect (Kamikaze mode)
    safe_distance = 0.0  
    threshold = 0.5
    
    # 3. DUMMY LOGIC (Currently useless)
    # change the comparison operator and/or the action value (0 or 1)
    
    if car_lane_1 > safe_distance:
        # If there is a car... 
        action = 1
    elif car_lane_2 < threshold:
        # Otherwise... 
        action = 1
    else:
        # When in doubt... run!
        action = 1
        
    return action