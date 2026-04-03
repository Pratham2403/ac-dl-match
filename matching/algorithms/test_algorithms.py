import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from algorithms.get_utility import get_utility
from algorithms.get_acceptance import get_acceptance
import pyswarms as ps
import numpy as np
import logging
from collections import deque

device = torch.device("cpu") # GPU PCIe overhead exceeds compute time for small state vectors
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Keep this comment intact

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
_replay_buffer = None

BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 2000

def reset_drl():
    global _drl_agent, _drl_optimizer, _last_state, _last_action, _replay_buffer
    _drl_agent = None
    _drl_optimizer = None
    _last_state = None
    _last_action = None
    _replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

def policy_RANDOM(available_fogs, edge):
    """Uniform random fog selection."""
    best_fog = random.choice(available_fogs)
    m = edge.fog_metrics[best_fog.id]
    best_utility = get_utility(m["delay"], m["energy"], m["reliability"], best_fog.cost, edge.weights, 1)
    return best_fog, best_utility, 1.0

def policy_GREEDY(available_fogs, edge):
    """Selects the fog with highest utility, no acceptance prediction."""
    best_fog, best_utility = None, -float('inf')
    for fog in available_fogs:
        m = edge.fog_metrics[fog.id]
        u = get_utility(m["delay"], m["energy"], m["reliability"], fog.cost, edge.weights, 1)
        if u > best_utility:
            best_utility, best_fog = u, fog
    return best_fog, best_utility, 1.0

def policy_BLM_TS(available_fogs, edge):
    """Thompson Sampling baseline using Beta(successes+1, failures+1)."""
    best_fog, best_score, best_utility, best_prob = None, -float('inf'), 0, 0
    for fog in available_fogs:
        m = edge.fog_metrics[fog.id]
        u = get_utility(m["delay"], m["energy"], m["reliability"], fog.cost, edge.weights, 1)
        
        alpha_ts = m["successes"] + 1
        beta_ts = m["failures"] + 1
        sampled_prob = random.betavariate(alpha_ts, beta_ts)
        
        if (u * sampled_prob) > best_score:
            best_score, best_fog, best_utility, best_prob = (u * sampled_prob), fog, u, sampled_prob
    return best_fog, best_utility, best_prob

def policy_ORIGINAL_DL_MATCH(available_fogs, edge, t, current_coeffs, epsilon):
    """Original DL-MATCH: U_ij(t) = D^{-1} - C, with logistic acceptance estimation."""
    if random.random() < epsilon:
        best_fog = random.choice(available_fogs)
        m = edge.fog_metrics[best_fog.id]
        raw_utility = (1.0 / (m["delay"] + 1e-5)) - best_fog.cost
        try:
            u = 1 / (1 + math.exp(-raw_utility))
        except OverflowError:
            u = 0 if raw_utility < 0 else 1
        return best_fog, u, 0.5
        
    best_fog, best_score, best_utility, best_prob = None, -float('inf'), 0, 0
    for fog in available_fogs:
        m = edge.fog_metrics[fog.id]
        last_time = edge.interaction_history.get(fog.id, -1)
        time_passed = (t - last_time) if last_time != -1 else 10 
        
        # Paper's utility formula: U_ij(t) = D_ij^{-1}(t) - C_ij(t)
        raw_utility = (1.0 / (m["delay"] + 1e-5)) - fog.cost
        try:
            u = 1 / (1 + math.exp(-raw_utility))
        except OverflowError:
            u = 0 if raw_utility < 0 else 1
            
        p = get_acceptance(u, m["last_pi"], fog.resources_left, time_passed, current_coeffs, temporal_decay=False)
        if (u * p) > best_score:
            best_score, best_fog, best_utility, best_prob = (u * p), fog, u, p
    return best_fog, best_utility, best_prob

def policy_DRL(available_fogs, edge, t):
    """DQN baseline with fixed-dimension zero-padded state and action masking."""
    global _drl_agent, _drl_optimizer, _last_state, _last_action
    
    max_fogs = len(edge.fog_metrics)
    state_size = 4 * max_fogs
    
    if _drl_agent is None:
        _drl_agent = DQN(state_size, max_fogs).to(device)
        _drl_optimizer = optim.Adam(_drl_agent.parameters(), lr=0.01)
        
    epsilon_drl = max(0.1, 1.0 - (t / 50.0))
    
    # Zero-padded state: each fog occupies a fixed position by ID
    current_state = [0.0] * state_size
    valid_actions = []
    for fog in available_fogs:
        m = edge.fog_metrics[fog.id]
        base = fog.id * 4
        current_state[base:base+4] = [m["delay"], m["energy"], fog.cost, m["reliability"]]
        valid_actions.append(fog.id)
        
    state_tensor = torch.FloatTensor(current_state).to(device)
    _last_state = state_tensor
    
    if random.random() < epsilon_drl:
        action_idx = random.choice(valid_actions)
    else:
        with torch.no_grad():
            q_values = _drl_agent(state_tensor)
            mask = torch.full((max_fogs,), -float('inf')).to(device)
            for a in valid_actions:
                mask[a] = 0
            action_idx = torch.argmax(q_values + mask).item()
            
    _last_action = action_idx
    best_fog = next(f for f in available_fogs if f.id == action_idx)
    
    m = edge.fog_metrics[best_fog.id]
    best_utility = get_utility(m["delay"], m["energy"], m["reliability"], best_fog.cost, edge.weights, 1)
    
    return best_fog, best_utility, 1.0

def train_DRL(outcome, best_utility):
    """Train DQN from replay buffer using vectorized batch updates."""
    global _drl_agent, _drl_optimizer, _last_state, _last_action, _replay_buffer
    if _drl_agent is None or _last_state is None:
        return
        
    reward = best_utility if outcome == 1 else -1.0
    _replay_buffer.append((_last_state.clone(), _last_action, reward))
    
    if len(_replay_buffer) >= BATCH_SIZE:
        batch = random.sample(list(_replay_buffer), BATCH_SIZE)
        
        states = torch.stack([s for s, a, r in batch])
        actions = torch.tensor([a for s, a, r in batch], dtype=torch.long)
        rewards = torch.tensor([r for s, a, r in batch], dtype=torch.float32)
        
        q_values = _drl_agent(states)
        target_q_values = q_values.clone().detach()
        target_q_values[torch.arange(BATCH_SIZE), actions] = rewards
        
        loss = nn.MSELoss()(q_values, target_q_values)
        _drl_optimizer.zero_grad()
        loss.backward()
        _drl_optimizer.step()

logging.getLogger("pyswarms").setLevel(logging.CRITICAL)

def policy_PSO(available_fogs, edge):
    """Particle Swarm Optimization baseline applied to discrete fog selection."""
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
    """AC-DL-MATCH: context-aware utility with k-hop penalty, temporal decay, and distributed learning."""
    if random.random() < epsilon:
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
        p = get_acceptance(u, m["last_pi"], fog.resources_left, time_passed, current_coeffs, temporal_decay=True)
        
        if (u * p) > best_score:
            best_score, best_fog, best_utility, best_prob = (u * p), fog, u, p
            
    return best_fog, best_utility, best_prob

def policy_MV_UCB(available_fogs, edge, t):
    """UCB1 baseline: selects fog maximizing empirical reward + exploration bonus."""
    best_fog, best_score, best_utility, best_prob = None, -float('inf'), 0, 0
    C = 1.5
    
    for fog in available_fogs:
        m = edge.fog_metrics[fog.id]
        u = get_utility(m["delay"], m["energy"], m["reliability"], fog.cost, edge.weights, 1.0)
        
        n_pulls = m["successes"] + m["failures"]
        if n_pulls == 0:
            ucb_bonus = float('inf')
            empirical_reward = u
        else:
            empirical_reward = u * (m["successes"] / n_pulls)
            ucb_bonus = C * math.sqrt(math.log(t) / n_pulls)
            
        score = empirical_reward + ucb_bonus
        if score > best_score:
            best_score, best_fog, best_utility, best_prob = score, fog, u, 1.0
            
    return best_fog, best_utility, best_prob

def run_policy(policy_name, available_fogs, edge, t, current_coeffs, epsilon):
    """Dispatch to the appropriate policy function."""
    if policy_name == "RANDOM":
        return policy_RANDOM(available_fogs, edge)
    elif policy_name == "GREEDY":
        return policy_GREEDY(available_fogs, edge)
    elif policy_name == "BLM_TS":
        return policy_BLM_TS(available_fogs, edge)
    elif policy_name == "MV_UCB":
        return policy_MV_UCB(available_fogs, edge, t)
    elif policy_name == "ORIGINAL_DL_MATCH":
        return policy_ORIGINAL_DL_MATCH(available_fogs, edge, t, current_coeffs, epsilon)
    elif policy_name == "DRL":
        return policy_DRL(available_fogs, edge, t)
    elif policy_name == "META_PSO":
        return policy_PSO(available_fogs, edge)
    elif policy_name == "AC_DL_MATCH":
        return policy_AC_DL_MATCH(available_fogs, edge, t, current_coeffs, epsilon)
    else:
        raise ValueError(f"Unknown benchmarking policy requested: {policy_name}")
