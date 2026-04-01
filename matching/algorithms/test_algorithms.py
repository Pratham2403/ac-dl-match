import random
import torch
import torch.nn as nn
import torch.optim as optim
from algorithms.get_utility import get_utility
from algorithms.get_acceptance import get_acceptance
import pyswarms as ps
import numpy as np
import logging

# Automatically detect available GPU device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, state_size, num_fogs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.out = nn.Linear(24, num_fogs)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

_drl_agent = None
_drl_optimizer = None
_last_state = None
_last_action = None

def reset_drl():
    global _drl_agent, _drl_optimizer, _last_state, _last_action
    _drl_agent = None
    _drl_optimizer = None
    _last_state = None
    _last_action = None

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
    """Corrected Thompson Sampling (Beta Distribution) Baseline"""
    best_fog, best_score, best_utility, best_prob = None, -float('inf'), 0, 0
    for fog in available_fogs:
        m = edge.fog_metrics[fog.id]
        u = get_utility(m["delay"], m["energy"], m["reliability"], fog.cost, edge.weights, 1)
        
        # True Thompson Sampling: Sample from Beta(successes + 1, failures + 1)
        alpha_ts = m.get("successes", 0) + 1
        beta_ts = m.get("failures", 0) + 1
        sampled_prob = random.betavariate(alpha_ts, beta_ts)
        
        if (u * sampled_prob) > best_score:
            best_score, best_fog, best_utility, best_prob = (u * sampled_prob), fog, u, sampled_prob
    return best_fog, best_utility, best_prob

def policy_ORIGINAL_DL_MATCH(available_fogs, edge, t, current_coeffs):
    """First-generation DL-MATCH framework"""
    import math
    best_fog, best_score, best_utility, best_prob = None, -float('inf'), 0, 0
    for fog in available_fogs:
        m = edge.fog_metrics[fog.id]
        last_time = edge.interaction_history.get(fog.id, -1)
        time_passed = (t - last_time) if last_time != -1 else 10 
        
        # Original Paper's mathematically strict Utility formula: U_ij(t) = D_ij^{-1}(t) - C_ij(t)
        # Note: No task differentiation, no energy matrices, and no spatial "k-hops"
        raw_utility = (1.0 / (m["delay"] + 1e-5)) - fog.cost
        try:
            u = 1 / (1 + math.exp(-raw_utility))
        except OverflowError:
            u = 0 if raw_utility < 0 else 1
            
        p = get_acceptance(u, m["last_pi"], fog.resources_left, time_passed, current_coeffs)
        if (u * p) > best_score:
            best_score, best_fog, best_utility, best_prob = (u * p), fog, u, p
    return best_fog, best_utility, best_prob

def policy_DRL(available_fogs, edge, t):
    """Deep Q-Network Baseline for Research Comparison"""
    global _drl_agent, _drl_optimizer, _last_state, _last_action
    
    state_size = 4 * len(available_fogs)
    
    if _drl_agent is None or _drl_agent.out.out_features != len(available_fogs):
        _drl_agent = DQN(state_size, len(available_fogs)).to(device)
        _drl_optimizer = optim.Adam(_drl_agent.parameters(), lr=0.01)
        
    epsilon_drl = max(0.1, 1.0 - (t / 50.0))
    
    current_state = []
    for fog in available_fogs:
        m = edge.fog_metrics[fog.id]
        current_state.extend([m["delay"], m["energy"], fog.cost, m["reliability"]])
        
    state_tensor = torch.FloatTensor(current_state).to(device)
    _last_state = state_tensor
    
    if random.random() < epsilon_drl:
        action_idx = random.randint(0, len(available_fogs)-1)
    else:
        with torch.no_grad():
            q_values = _drl_agent(state_tensor)
            action_idx = torch.argmax(q_values).item()
            
    _last_action = action_idx
    best_fog = available_fogs[action_idx]
    
    m = edge.fog_metrics[best_fog.id]
    best_utility = get_utility(m["delay"], m["energy"], m["reliability"], best_fog.cost, edge.weights, 1)
    
    return best_fog, best_utility, 1.0

def train_DRL(outcome, best_utility):
    global _drl_agent, _drl_optimizer, _last_state, _last_action
    if _drl_agent is None or _last_state is None:
        return
        
    reward = best_utility if outcome == 1 else -1.0
    
    q_values = _drl_agent(_last_state)
    target_q_values = q_values.clone()
    target_q_values[_last_action] = reward
    
    loss = nn.MSELoss()(q_values, target_q_values)
    _drl_optimizer.zero_grad()
    loss.backward()
    _drl_optimizer.step()

logging.getLogger("pyswarms").setLevel(logging.CRITICAL)

def policy_PSO(available_fogs, edge):
    """Meta-Heuristic Baseline (Particle Swarm Optimization)"""
    def fitness_func(positions):
        n_particles = positions.shape[0]
        fitness = np.zeros(n_particles)
        for i in range(n_particles):
            pos_val = positions[i, 0]
            if np.isnan(pos_val):
                idx = random.randint(0, len(available_fogs)-1)
            else:
                idx = int(round(pos_val))
            idx = max(0, min(len(available_fogs)-1, idx))
            fog = available_fogs[idx]
            m = edge.fog_metrics[fog.id]
            u = get_utility(m["delay"], m["energy"], m["reliability"], fog.cost, edge.weights, 1)
            fitness[i] = -u # PSO minimizes the cost function
        return fitness

    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    bounds = (np.array([0.0]), np.array([float(len(available_fogs)-1)]))
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=1, options=options, bounds=bounds, velocity_clamp=(-1.0, 1.0))
    best_cost, best_pos = optimizer.optimize(fitness_func, iters=10, verbose=False)
    
    if np.isnan(best_pos[0]):
        action_idx = random.randint(0, len(available_fogs)-1)
    else:
        action_idx = int(round(best_pos[0]))
        
    action_idx = max(0, min(len(available_fogs)-1, action_idx))
    best_fog = available_fogs[action_idx]
    
    m = edge.fog_metrics[best_fog.id]
    best_utility = get_utility(m["delay"], m["energy"], m["reliability"], best_fog.cost, edge.weights, 1)
    return best_fog, best_utility, 1.0

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
        return policy_ORIGINAL_DL_MATCH(available_fogs, edge, t, current_coeffs)
    elif policy_name == "DRL":
        return policy_DRL(available_fogs, edge, t)
    elif policy_name == "META_PSO":
        return policy_PSO(available_fogs, edge)
    elif policy_name == "AC_DL_MATCH":
        return policy_AC_DL_MATCH(available_fogs, edge, t, current_coeffs, epsilon)
    else:
        raise ValueError(f"Unknown benchmarking policy requested: {policy_name}")
