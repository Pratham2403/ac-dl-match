def get_utility(norm_delay, norm_energy, norm_rel, norm_cost, task_weights, hops=1, k_max=3):
    w1, w2, w3, w4 = task_weights
    
    # Pure Weighted Average of [0,1] normalized SLA parameters
    total_weight = sum(task_weights)
    weighted_sum = (w1 * norm_delay) + (w2 * norm_energy) + (w3 * norm_rel) + (w4 * norm_cost)
    raw_utility = weighted_sum / (total_weight + 1e-5)
    
    # k-hop Spatial Penalty
    k_hop_penalty = 1.0 - (0.3 * hops / k_max)
    final_utility = raw_utility * k_hop_penalty
    
    # Clamp safely between 0 and 1 (NO SIGMOID)
    return max(0.0, min(1.0, final_utility))