import random
from algorithms.get_utility import get_utility
from algorithms.get_acceptance import get_acceptance

def policy_RANDOM(available_fogs, edge):
    """Ethical Random Benchmark that strictly logs apples-to-apples utility"""
    best_fog = random.choice(available_fogs)
    m = edge.fog_metrics[best_fog.id]
    best_utility = get_utility(m["delay"], m["energy"], m["reliability"], best_fog.cost, edge.weights, 1)
    return best_fog, best_utility, 1.0

def policy_GREEDY(available_fogs, edge):
    """Picks node strictly based on maximum utility without prediction models"""
    best_fog, best_score, best_utility, best_prob = None, -float('inf'), 0, 0
    for fog in available_fogs:
        m = edge.fog_metrics[fog.id]
        u = get_utility(m["delay"], m["energy"], m["reliability"], fog.cost, edge.weights, 1)
        p = 1.0
        if (u * p) > best_score:
            best_score, best_fog, best_utility, best_prob = (u * p), fog, u, p
    return best_fog, best_utility, best_prob

def policy_BLM_TS(available_fogs, edge, current_coeffs):
    """Baseline matching prediction lacking spatial/temporal decay awareness"""
    best_fog, best_score, best_utility, best_prob = None, -float('inf'), 0, 0
    for fog in available_fogs:
        m = edge.fog_metrics[fog.id]
        u = get_utility(m["delay"], m["energy"], m["reliability"], fog.cost, edge.weights, 1)
        p = get_acceptance(u, m["last_pi"], fog.resources_left, 0, current_coeffs)
        if (u * p) > best_score:
            best_score, best_fog, best_utility, best_prob = (u * p), fog, u, p
    return best_fog, best_utility, best_prob

def policy_ORIGINAL_DL_MATCH(available_fogs, edge, current_coeffs):
    """First-generation DL-MATCH framework"""
    best_fog, best_score, best_utility, best_prob = None, -float('inf'), 0, 0
    for fog in available_fogs:
        m = edge.fog_metrics[fog.id]
        u = get_utility(m["delay"], m["energy"], m["reliability"], fog.cost, edge.weights, 1)
        p = get_acceptance(u, m["last_pi"], fog.resources_left, 0, current_coeffs)
        if (u * p) > best_score:
            best_score, best_fog, best_utility, best_prob = (u * p), fog, u, p
    return best_fog, best_utility, best_prob

def policy_AC_DL_MATCH(available_fogs, edge, t, current_coeffs, epsilon):
    """Novel Architecture: k-Hop spatial awareness + Temporal Decay + Epsilon Exploration"""
    if random.random() < epsilon:
        # Epsilon-Greedy Exploration
        best_fog = random.choice(available_fogs)
        m = edge.fog_metrics[best_fog.id]
        best_utility = get_utility(m["delay"], m["energy"], m["reliability"], best_fog.cost, edge.weights, m["hops"])
        return best_fog, best_utility, 0.5
        
    best_fog, best_score, best_utility, best_prob = None, -float('inf'), 0, 0
    for fog in available_fogs:
        last_time = edge.interaction_history.get(fog.id, -1)
        time_passed = (t - last_time) if last_time != -1 else 10 
        m = edge.fog_metrics[fog.id]
        
        u = get_utility(m["delay"], m["energy"], m["reliability"], fog.cost, edge.weights, m["hops"])
        p = get_acceptance(u, m["last_pi"], fog.resources_left, time_passed, current_coeffs)
        
        if (u * p) > best_score:
            best_score, best_fog, best_utility, best_prob = (u * p), fog, u, p
            
    return best_fog, best_utility, best_prob

def run_policy(policy_name, available_fogs, edge, t, current_coeffs, epsilon):
    """Router function to evaluate the explicitly decoupled algorithms"""
    if policy_name == "RANDOM":
        return policy_RANDOM(available_fogs, edge)
    elif policy_name == "GREEDY":
        return policy_GREEDY(available_fogs, edge)
    elif policy_name == "BLM_TS":
        return policy_BLM_TS(available_fogs, edge, current_coeffs)
    elif policy_name == "ORIGINAL_DL_MATCH":
        return policy_ORIGINAL_DL_MATCH(available_fogs, edge, current_coeffs)
    elif policy_name == "AC_DL_MATCH":
        return policy_AC_DL_MATCH(available_fogs, edge, t, current_coeffs, epsilon)
    else:
        raise ValueError(f"Unknown benchmarking policy requested: {policy_name}")
