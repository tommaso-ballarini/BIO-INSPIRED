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
    alignment_factor = 0.5
    if target_delta_x > 0:
        action = 1
    elif target_delta_x < 0:
        action = 2
    else:
        action = 0
    
    # Avoidance logic with enhanced sensitivity and additional checks
    avoidance_factor = 1.5
    if nearest_threat_dist_y < 3 and nearest_threat_dx * player_x > 0 and threat_type == -1.0:
        if abs(nearest_threat_dx) > abs(target_delta_x):
            action = 2 if nearest_threat_dx > 0 else 1
    
    # Additional logic to prioritize alignment over avoidance with a margin
    if abs(target_delta_x) > 0.3 * speed_x:
        action = 1 if target_delta_x > 0 else 2
    
    # Prediction logic: Predict gate position based on speed and distance
    predicted_gate_dx = speed_x * (abs(target_delta_x) / 5)
    
    # Adjust action based on predicted gate position and threat distance
    if nearest_threat_dist_y < 3 and nearest_threat_dx * player_x > 0 and threat_type == -1.0:
        if nearest_threat_dx + predicted_gate_dx > 0:
            action = 2
        elif nearest_threat_dx + predicted_gate_dx < 0:
            action = 1
    
    # Damping logic: Smooth steering inputs to reduce oscillations
    damping_factor = 0.5
    if action == 1:
        player_x += damping_factor * speed_x
    elif action == 2:
        player_x -= damping_factor * speed_x
    
    # Ensure the action is within valid range
    action = max(0, min(action, 2))
    
    return action

# Example usage:
observation = np.array([1.0, 2.0, 3.0, -4.0, 5.0, -6.0, 7.0, 8.0, 9.0])
action = get_action(observation)
print(action)  # Output should be either 0, 1, or 2