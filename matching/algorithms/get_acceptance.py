import math

def get_acceptance(utility, previous_success_rate, resources_left, time_passed, coefficients):
    alpha, beta, gamma = coefficients if len(coefficients) == 3 else [1.0, 1.0, 1.0]
    
    # Hybrid EMA + Temporal Decay (New Novelty)
    decay = math.exp(-0.1 * time_passed) if time_passed > 0 else 0.9
    z = (alpha * utility) + (beta * previous_success_rate * decay) + (gamma * resources_left)
    
    try:
        return 1 / (1 + math.exp(-z))
    except OverflowError:
        return 0 if z < 0 else 1