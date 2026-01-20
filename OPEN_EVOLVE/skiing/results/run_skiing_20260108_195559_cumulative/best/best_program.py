import numpy as np

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
    alignment_threshold = 0.5
    action = 1 if target_delta_x > alignment_threshold * speed_x else 2
    
    # Avoidance logic with enhanced sensitivity and additional checks
    avoidance_threshold = -5
    if nearest_threat_dist_y < 3 and nearest_threat_dx * player_x > avoidance_threshold and threat_type == -1.0:
        action = 2 if nearest_threat_dx > 0 else 1
    
    # Additional logic to prioritize alignment over avoidance with a margin
    alignment_margin = 0.3
    if abs(target_delta_x) > alignment_margin * speed_x:
        action = 1 if target_delta_x > alignment_margin * speed_x else 2
    
    return action

# Example usage
observation = np.array([10, 20, 3, -5, 45, -2, 1, -1, 1])
action = get_action(observation)
print(action)