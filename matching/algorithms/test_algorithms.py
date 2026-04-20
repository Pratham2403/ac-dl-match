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

CROSS_DOMAIN_PENALTY = 0.20 # Standard East-West federation interconnection penalty
MIN_UTILITY_THRESHOLD = 0.30  # Raw utility threshold
MIN_PROB_THRESHOLD = 0.15     # Allow algorithm to take a 15% chance if utility is amazing

class DQN(nn.Module):
    def __init__(self, state_size, max_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, max_actions)

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
REPLAY_BUFFER_SIZE = 100000

def reset_drl(max_total_fogs):
    global _drl_agent, _drl_optimizer, _last_state, _last_action, _replay_buffer
    _drl_agent = None
    _drl_optimizer = None
    _last_state = None
    _last_action = None
    _replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

def get_search_space(local_fogs, neighbor_fogs):
    return [(f, False) for f in local_fogs] + [(f, True) for f in neighbor_fogs]

def get_base_metrics(sdn, is_neighbor, fog, edge):
    m = sdn.get_metrics(edge.id, fog) if not is_neighbor else fog.sdn.get_metrics(edge.id, fog)
    u = get_utility(m["norm_delay"], m["norm_energy"], m["norm_rel"], m["norm_cost"], edge.weights)
    if is_neighbor:
        u = u * (1.0 - CROSS_DOMAIN_PENALTY)
    return m, u

def policy_RANDOM(sdn, edge, local_fogs, neighbor_fogs, t):
    search_space = get_search_space(local_fogs, neighbor_fogs)
    fog, is_neighbor = random.choice(search_space)
    m, u = get_base_metrics(sdn, is_neighbor, fog, edge)
    return fog, u, 1.0

def policy_GREEDY(sdn, edge, local_fogs, neighbor_fogs, t):
    search_space = get_search_space(local_fogs, neighbor_fogs)
    best_fog, best_utility = None, -float('inf')
    
    for fog, is_neighbor in search_space:
        m, u = get_base_metrics(sdn, is_neighbor, fog, edge)
        if u > best_utility and u >= MIN_UTILITY_THRESHOLD:
            best_utility, best_fog = u, fog
            
    return best_fog, max(0, best_utility), 1.0

def policy_BLM_TS(sdn, edge, local_fogs, neighbor_fogs, t):
    search_space = get_search_space(local_fogs, neighbor_fogs)
    best_fog, best_score, best_utility = None, -float('inf'), 0
    
    for fog, is_neighbor in search_space:
        m, u = get_base_metrics(sdn, is_neighbor, fog, edge)
        alpha_ts = m["successes"] + 1
        beta_ts = m["failures"] + 1
        sampled_prob = random.betavariate(alpha_ts, beta_ts)
        
        score = u * sampled_prob
        if score > best_score and score >= MIN_UTILITY_THRESHOLD:
            best_score, best_fog, best_utility = score, fog, u
            
    return best_fog, max(0, best_utility), 1.0

def policy_ORIGINAL_DL_MATCH(sdn, edge, local_fogs, neighbor_fogs, t, epsilon):
    search_space = get_search_space(local_fogs, neighbor_fogs)
    
    if random.random() < epsilon and search_space:
        fog, is_neighbor = random.choice(search_space)
        m, u = get_base_metrics(sdn, is_neighbor, fog, edge)
        return fog, u, 0.5
        
    best_fog, best_score, best_utility, best_prob = None, -float('inf'), 0, 0
    for fog, is_neighbor in search_space:
        m = sdn.get_metrics(edge.id, fog) if not is_neighbor else fog.sdn.get_metrics(edge.id, fog)
        
        # Evaluate D^-1 - C strictly through mathematically bounded domain coordinates
        cost_penalty = 1.0 - m["norm_cost"] 
        raw_utility = m["norm_delay"] - cost_penalty
        try:
            u_orig = 1 / (1 + math.exp(-raw_utility))
        except OverflowError:
            u_orig = 0 if raw_utility < 0 else 1
            
        if is_neighbor:
            u_orig = u_orig * (1.0 - CROSS_DOMAIN_PENALTY)
            
        time_passed = (t - m["last_time"]) if m["last_time"] > 0 else 10
        coeffs = sdn.domain_coeffs if not is_neighbor else fog.sdn.domain_coeffs
        p = get_acceptance(u_orig, m["last_pi"], fog.resources_left, time_passed, coeffs, temporal_decay=False)
        
        # ORIGINAL explicitly uses its own bounds
        if (u_orig * p) > best_score:
            best_score, best_fog, best_utility, best_prob = (u_orig * p), fog, u_orig, p
            
    return best_fog, best_utility, best_prob

def policy_DRL(sdn, edge, local_fogs, neighbor_fogs, t, max_total_fogs):
    global _drl_agent, _drl_optimizer, _last_state, _last_action
    
    search_space = get_search_space(local_fogs, neighbor_fogs)
    state_size = (4 * max_total_fogs) + 4
    
    if _drl_agent is None or _drl_agent.fc1.in_features != state_size:
        _drl_agent = DQN(state_size, max_total_fogs).to(device)
        _drl_optimizer = optim.Adam(_drl_agent.parameters(), lr=0.01)
        _replay_buffer.clear()
        
    epsilon_drl = max(0.05, 1.0 - (t / 400.0))
    current_state = [0.0] * state_size
    current_state[-4:] = [w / 10.0 for w in edge.weights]
    
    valid_actions = []
    for fog, is_neighbor in search_space:
        m, u = get_base_metrics(sdn, is_neighbor, fog, edge)
        base = fog.id * 4
        current_state[base:base+4] = [m["norm_delay"], m["norm_energy"], m["norm_cost"], m["norm_rel"]]
        valid_actions.append(fog.id)
        
    state_tensor = torch.FloatTensor(current_state).to(device)
    _last_state = state_tensor
    
    if random.random() < epsilon_drl and valid_actions:
        action_idx = random.choice(valid_actions)
    else:
        with torch.no_grad():
            q_values = _drl_agent(state_tensor)
            mask = torch.full((max_total_fogs,), -float('inf')).to(device)
            for a in valid_actions:
                if a < max_total_fogs:
                    mask[a] = 0
            action_idx = torch.argmax(q_values + mask).item()
            
    _last_action = action_idx
    best_fog = next((f for f, _ in search_space if f.id == action_idx), None)
    if not best_fog and search_space:
        best_fog, _ = random.choice(search_space)
        
    if best_fog:
        is_neighbor = best_fog not in local_fogs
        m, best_utility = get_base_metrics(sdn, is_neighbor, best_fog, edge)
        return best_fog, best_utility, 1.0
    return None, 0, 0

def train_DRL(outcome, best_utility):
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

def policy_PSO(sdn, edge, local_fogs, neighbor_fogs, t):
    search_space = get_search_space(local_fogs, neighbor_fogs)
    if not search_space: return None, 0, 0
    
    utils_cache = []
    for fog, is_neighbor in search_space:
        m, u = get_base_metrics(sdn, is_neighbor, fog, edge)
        utils_cache.append(u)
    utils_cache = np.array(utils_cache)

    def fitness_func(positions):
        idx_array = np.nan_to_num(positions[:, 0], nan=np.random.randint(0, len(search_space)))
        idx_array = np.clip(np.round(idx_array), 0, len(search_space)-1).astype(int)
        return -utils_cache[idx_array]

    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    bounds = (np.array([0.0]), np.array([float(len(search_space)-1)]))
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=1, options=options, bounds=bounds, velocity_clamp=(-1.0, 1.0))
    best_cost, best_pos = optimizer.optimize(fitness_func, iters=10, verbose=False)
    
    action_idx = int(np.clip(np.round(np.nan_to_num(best_pos[0], nan=0)), 0, len(search_space)-1))
    best_fog, _ = search_space[action_idx]
    best_utility = utils_cache[action_idx]
    
    if best_utility >= MIN_UTILITY_THRESHOLD:
        return best_fog, best_utility, 1.0
    return None, 0, 0

def policy_AC_DL_MATCH(sdn, edge, local_fogs, neighbor_fogs, t, epsilon, temporal_decay=True, use_lr=True):
    search_space = get_search_space(local_fogs, neighbor_fogs)
    
    if random.random() < epsilon and search_space:
        fog, is_neighbor = random.choice(search_space)
        m, u = get_base_metrics(sdn, is_neighbor, fog, edge)
        return fog, u, 0.5
        
    best_fog, best_score, best_utility, best_prob = None, -float('inf'), 0, 0
    for fog, is_neighbor in search_space:
        m, u = get_base_metrics(sdn, is_neighbor, fog, edge)
        
        if use_lr:
            time_passed = (t - m["last_time"]) if m["last_time"] > 0 else 10 
            coeffs = sdn.domain_coeffs if not is_neighbor else fog.sdn.domain_coeffs
            p = get_acceptance(u, m["last_pi"], fog.resources_left, time_passed, coeffs, temporal_decay=temporal_decay)
        else:
            p = 1.0
        
        if (u * p) > best_score:
            best_score, best_fog, best_utility, best_prob = (u * p), fog, u, p
            
    return best_fog, best_utility, best_prob

def policy_MV_UCB(sdn, edge, local_fogs, neighbor_fogs, t):
    search_space = get_search_space(local_fogs, neighbor_fogs)
    best_fog, best_score, best_utility = None, -float('inf'), 0
    C = 1.5
    
    for fog, is_neighbor in search_space:
        m, u = get_base_metrics(sdn, is_neighbor, fog, edge)
        
        n_pulls = m["successes"] + m["failures"]
        if n_pulls == 0:
            ucb_bonus = float('inf')
            empirical_reward = u
        else:
            empirical_reward = u * (m["successes"] / n_pulls)
            ucb_bonus = C * math.sqrt(math.log(t) / n_pulls)
            
        score = empirical_reward + ucb_bonus
        if score > best_score and u >= MIN_UTILITY_THRESHOLD:
            best_score, best_fog, best_utility = score, fog, u
            
    return best_fog, max(0, best_utility), 1.0

def run_policy(policy_name, sdn, edge, local_fogs, neighbor_fogs, t, epsilon, max_total_fogs):
    if policy_name == "RANDOM":
        return policy_RANDOM(sdn, edge, local_fogs, neighbor_fogs, t)
    elif policy_name == "GREEDY":
        return policy_GREEDY(sdn, edge, local_fogs, neighbor_fogs, t)
    elif policy_name == "BLM_TS":
        return policy_BLM_TS(sdn, edge, local_fogs, neighbor_fogs, t)
    elif policy_name == "MV_UCB":
        return policy_MV_UCB(sdn, edge, local_fogs, neighbor_fogs, t)
    elif policy_name == "ORIGINAL_DL_MATCH":
        return policy_ORIGINAL_DL_MATCH(sdn, edge, local_fogs, neighbor_fogs, t, epsilon)
    elif policy_name == "DRL":
        return policy_DRL(sdn, edge, local_fogs, neighbor_fogs, t, max_total_fogs)
    elif policy_name == "META_PSO":
        return policy_PSO(sdn, edge, local_fogs, neighbor_fogs, t)
    elif policy_name == "AC_DL_MATCH":
        return policy_AC_DL_MATCH(sdn, edge, local_fogs, neighbor_fogs, t, epsilon)
    elif policy_name == "AC_NO_LR":
        return policy_AC_DL_MATCH(sdn, edge, local_fogs, neighbor_fogs, t, epsilon, use_lr=False)
    elif policy_name == "AC_NO_DECAY":
        return policy_AC_DL_MATCH(sdn, edge, local_fogs, neighbor_fogs, t, epsilon, temporal_decay=False)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
