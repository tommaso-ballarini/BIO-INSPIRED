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
    
    # Avoidance logic
    elif abs(target_delta_x) <= 0.1 and nearest_threat_dist_y < 5:
        if threat_type == -1:  # Tree
            if nearest_threat_dx > 0:
                action = 2  # slightly left to avoid the tree
            else:
                action = 1  # slightly right to avoid the tree
        elif threat_type == -0.5:  # Flag Pole
            if target_delta_x >= 0 and nearest_threat_dist_y < 3:
                action = 0  # glide when flag poles are close and aligned
            else:
                action = (1 if nearest_threat_dx > 0 else 2)  # steer away from the threat
    
    # Default: Maintain current direction
    else:
        action = 0
    
    return action