def get_action(observation):
    """
    Improved Skiing Agent.
    Input: observation (numpy array of size 9).
    Output: Action (0, 1, or 2).
    """
    
    # Extract relevant observations
    target_delta_x = observation[3]
    nearest_threat_dx = observation[6]
    nearest_threat_dist_y = observation[7]
    threat_type = observation[8]
    
    # Alignment logic
    if abs(target_delta_x) > 0.1:
        action = 1 if target_delta_x > 0 else 2  # steer towards the gate more aggressively
    elif abs(target_delta_x) <= 0.1 and nearest_threat_dist_y < 5:
        if threat_type == -1:  # Tree
            action = 2 if nearest_threat_dx > 0 else 1  # steer away from trees more aggressively
        elif threat_type == -0.5:  # Flag Pole
            if target_delta_x >= 0:
                action = 0 if nearest_threat_dist_y < 3 else (1 if nearest_threat_dist_y < 4 else 2)
            else:
                action = (1 if nearest_threat_dist_y < 3 else 2)
    else:
        action = 0  # maintain current direction
    
    return action