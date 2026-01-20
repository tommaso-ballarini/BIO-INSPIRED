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
    
    # Alignment logic (Simplified)
    if target_delta_x > 0:
        action = 1
    elif target_delta_x < 0:
        action = 2
    else:
        action = 0
    
    # Avoidance logic with enhanced sensitivity and additional checks
    avoidance_threshold = -5
    if nearest_threat_dist_y < 3 and nearest_threat_dx * player_x > avoidance_threshold and threat_type == -1.0:
        if nearest_threat_dx > 0:
            action = 2
        else:
            action = 1
    
    # Additional logic to prioritize alignment over avoidance with a margin
    margin = speed_x * 0.3
    if abs(target_delta_x) > margin:
        action = 1 if target_delta_x > margin else 2
    
    # Prediction logic: Predict gate position based on speed and threat distance
    predicted_gate_dx = predict_gate_position(speed_x, target_delta_x)
    
    # Fine-tune the prediction logic to balance alignment and avoidance
    if nearest_threat_dist_y < 3 and nearest_threat_dx * player_x > 0 and threat_type == -1.0:
        adjusted_prediction_threshold = margin + speed_x * 0.2
        if predicted_gate_dx > adjusted_prediction_threshold:
            action = 2
        elif predicted_gate_dx < -adjusted_prediction_threshold:
            action = 1
    
    return action

def predict_gate_position(speed_x, target_delta_x):
    """
    Predicts the gate position based on speed and target delta x.
    
    Args:
        speed_x (float): Current speed of the player.
        target_delta_x (float): Delta x to the target.
    
    Returns:
        float: Predicted gate position.
    """
    return speed_x * (abs(target_delta_x) / 5)

# Example usage
observation = np.array([10, 20, 3, -5, 45, -2, 1, -1, 1])
action = get_action(observation)
print(action)