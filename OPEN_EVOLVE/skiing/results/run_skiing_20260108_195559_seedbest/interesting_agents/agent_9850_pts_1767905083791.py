def get_action(observation):
    """
    Improved Skiing Agent.
    Input: observation (numpy array of size 9).
    Output: Action (0, 1, or 2).
    """
    
    # Extract relevant observations
    player_x = observation[0]
    target_delta_x = observation[3]
    nearest_threat_dx = observation[6]
    nearest_threat_dist_y = observation[7]
    threat_type = observation[8]
    speed_x = observation[2]
    
    # Alignment logic (Magnet Reward)
    if target_delta_x > 0:
        action = 1
    elif target_delta_x < 0:
        action = 2
    else:
        action = 0
    
    # Avoidance logic with enhanced sensitivity and additional checks
    if nearest_threat_dist_y < 3 and nearest_threat_dx * player_x > 0 and threat_type == -1.0:
        if nearest_threat_dx > 0:
            action = 2
        else:
            action = 1
    
    # Additional logic to prioritize alignment over avoidance with a margin
    margin = 0.3 * speed_x
    if abs(target_delta_x) > margin:
        action = 1 if target_delta_x > margin else 2
    
    return action