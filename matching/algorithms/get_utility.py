import math

def get_utility(delay, energy, reliability, cost, task_weights, hops=1, k_max=3):
    w1, w2, w3, w4 = task_weights
    # Context-Aware Utility
    raw_utility = (w1 / (delay + 1e-5)) + (w2 / (energy + 1e-5)) + (w3 * reliability) - (w4 * cost)
    
    # k-hop Penalty
    k_hop_penalty = 1 - (0.3 * hops / k_max)
    raw_utility *= k_hop_penalty
    
    try:
        return 1 / (1 + math.exp(-raw_utility))
    except OverflowError:
        return 0 if raw_utility < 0 else 1