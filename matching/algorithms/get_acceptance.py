import math

def get_acceptance(utility, previous_success_rate, resources_left, time_passed, coefficients, temporal_decay=True):
    alpha, beta, gamma = coefficients if len(coefficients) == 3 else [1.0, 1.0, 1.0]
    
    if temporal_decay:
        decay = math.exp(-0.1 * time_passed) if time_passed > 0 else 0.9
    else:
        decay = 1.0
    
    z = (alpha * utility) + (beta * previous_success_rate * decay) + (gamma * resources_left)
    
    try:
        return 1 / (1 + math.exp(-z))
    except OverflowError:
        return 0 if z < 0 else 1