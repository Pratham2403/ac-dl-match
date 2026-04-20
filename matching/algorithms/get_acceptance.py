import math

def get_acceptance(utility, previous_success_rate, resources_left, time_passed, coefficients, temporal_decay=True):
    if isinstance(coefficients, tuple) and len(coefficients) == 2:
        feats, intercept = coefficients
        alpha, beta, gamma = feats if len(feats) == 3 else [1.0, 1.0, 1.0]
    else:
        alpha, beta, gamma = coefficients if len(coefficients) == 3 else [1.0, 1.0, 1.0]
        intercept = 0.0

    if temporal_decay:
        decay = math.exp(-0.1 * time_passed) if time_passed > 0 else 1.0
    else:
        decay = 1.0

    # ------------------------------------------------------------------
    # ACADEMIC FIX: Bayesian Reversion to the Mean
    # If we haven't seen a fog in a long time (decay -> 0), our confidence 
    # should revert to a neutral prior (0.5), NOT 0 (certain failure).
    # ------------------------------------------------------------------
    effective_pi = (previous_success_rate * decay) + (0.5 * (1.0 - decay))
    
    # INTERCEPT FIX: Retain logistic regression offset for properly calibrated probabilities
    z = intercept + (alpha * utility) + (beta * effective_pi) + (gamma * resources_left)
    
    try:
        return 1 / (1 + math.exp(-z))
    except OverflowError:
        return 0 if z < 0 else 1