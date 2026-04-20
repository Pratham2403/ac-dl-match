"""AC-DL-MATCH Benchmarking Simulator for Task Offloading in Dynamic Fog Computing Networks."""

import os
import time
import random
import math
import argparse
import numpy as np
import torch
from collections import deque
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

from utils.plotter import BenchmarkPlotter
from algorithms.get_utility import get_utility
from algorithms.scale_out import scale_out
from algorithms.scale_in import scale_in
from algorithms.test_algorithms import run_policy, train_DRL, reset_drl
from entities.fog_node import FogNode
from entities.edge_node import EdgeNode
from entities.sdn_controller import SDNController

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = None
_log_file_handle = None

def get_env_config(args):
    """Dynamically generates SLA bounds and physical constraints based on the 1:10:1000 topology."""
    quality = "best" if args.best else "worst" if args.worst else "average"
    
    # Baseline config - 3 SDNs, 30 Fogs, 3000 Edges (1:10:1000 ratio)
    # Production fog servers handle ~100 concurrent lightweight IoT tasks
    config = {
        "NUM_SLOTS": 500, "NUM_SDNS": 3, "MIN_FOGS": 30, "NUM_EDGES": 3000, 
        "MAX_SDN_FOGS": 20,
        "MAX_DELAY": 100.0, "MAX_ENERGY": 30.0, "MAX_COST": 15.0,
        "FOG_CAPACITY": {"best": 150, "average": 100, "worst": 50}[quality]
    }
    
    # Topology overrides (mutually exclusive)
    if getattr(args, 'demo', False):
        # Demo - scaled-down for fast presentation (small edge appliances)
        config.update({
            "NUM_SLOTS": 500, "NUM_SDNS": 3, "MIN_FOGS": 15, "NUM_EDGES": 75, 
            "MAX_SDN_FOGS": 10,
            "FOG_CAPACITY": {"best": 15, "average": 8, "worst": 3}[quality]
        })
    elif args.stress:
        # Stress - near real-world deployment scale (slightly constrained servers)
        config.update({
            "NUM_SLOTS": 2000, "NUM_SDNS": 5, "MIN_FOGS": 50, "NUM_EDGES": 5000, 
            "MAX_SDN_FOGS": 30,
            "MAX_DELAY": 250.0, "MAX_ENERGY": 50.0, "MAX_COST": 25.0,
            "FOG_CAPACITY": {"best": 120, "average": 80, "worst": 40}[quality]
        })
    
    # Quality overrides (SLA bounds only, independent of topology)
    if args.best:
        config.update({"MAX_DELAY": 50.0, "MAX_ENERGY": 15.0, "MAX_COST": 10.0})
    elif args.worst:
        config.update({"MAX_DELAY": 150.0, "MAX_ENERGY": 40.0, "MAX_COST": 20.0})
        
    return config

def init_logging():
    global log_file_path, _log_file_handle
    os.makedirs("logs", exist_ok=True)
    log_file_path = f"logs/simulation_{timestamp}.log"
    _log_file_handle = open(log_file_path, "a")

def log_message(msg):
    if _log_file_handle:
        _log_file_handle.write(msg + "\n")

def run_simulation(policy, env_config, quality="average", trace_df=None):
    """Run a single simulation epoch for the given policy and return time-series metrics."""
    max_total_fogs = env_config["MAX_SDN_FOGS"] * env_config["NUM_SDNS"]
    reset_drl(max_total_fogs)
    log_message(f"\n{'='*20} STARTING POLICY: {policy} {'='*20}")
    
    K_MAX_RETRIES = 3
    REJECT_THRESHOLD = 0.1
    UTIL_THRESHOLD = 0.3
    CLOUD_DELAY_PENALTY = env_config["MAX_DELAY"] * 2.0
    CLOUD_ENERGY_PENALTY = env_config["MAX_ENERGY"] * 1.5
    CLOUD_COST_PENALTY = env_config["MAX_COST"] * 1.5
    
    # 1. Initialize SDNs
    sdns = [SDNController(i, env_config, quality) for i in range(env_config["NUM_SDNS"])]
    # Link rings topology
    if len(sdns) > 1:
        for i in range(len(sdns)):
            sdns[i].neighbor_sdns.append(sdns[(i+1)%len(sdns)])
            if len(sdns) > 2:
                sdns[i].neighbor_sdns.append(sdns[(i-1)%len(sdns)])
                
    fog_capacity = env_config["FOG_CAPACITY"]
    fogs = [FogNode(i, quality, sdn_id=(i % len(sdns)), capacity=fog_capacity) for i in range(env_config["MIN_FOGS"])]
    edges = [EdgeNode(i, [random.randint(1,10) for _ in range(4)], env_config, quality) for i in range(env_config["NUM_EDGES"])]

    # Assign architecture
    for fog in fogs:
        fog.sdn = sdns[fog.sdn_id]
        sdns[fog.sdn_id].local_fogs.append(fog)
        
    for i, edge in enumerate(edges):
        sdn = sdns[i % len(sdns)]
        edge.sdn = sdn
        sdn.local_edges.append(edge)

    metrics = {"acc_rate": [], "delay": [], "utility": [], "energy": [], "cost": [], "time": []}
    util_window = deque(maxlen=5)
    cum_utility = 0
    
    for t in range(1, env_config["NUM_SLOTS"] + 1):
        # A. Physical Hardware Layer State Update (All domains universally)
        if trace_df is not None and (t - 1) < len(trace_df):
            row = trace_df.iloc[t - 1]
            cpu_load = row['cpu_util_percent'] / 100.0
            mem_load = row['mem_util_percent'] / 100.0
            for fog in fogs:
                tasks_to_complete = np.random.binomial(fog.active_tasks, 0.4)
                fog.active_tasks -= tasks_to_complete
                fog.resources_left = max(0.01, 1.0 - mem_load)
                fog.cost = 5.0 + (cpu_load * 20.0)
        else:
            for fog in fogs:
                fog.update_state()
                
        t_global_rejects, t_global_delay, t_global_energy, t_global_cost = 0, 0, 0, 0
        t_global_active_edges = 0

        if _log_file_handle and t % 10 == 0:
            log_message(f"\n[{policy}] >>> Slot {t} Infrastructure State <<<")
            for sdn in sdns:
                util = sum([f.active_tasks/max(1,f.capacity) for f in sdn.local_fogs]) / max(1, len(sdn.local_fogs))
                log_message(f"  SDN-{sdn.id} -> {len(sdn.local_fogs)} Fogs Active | Avg Queue Saturation: {util:.1%}")

        # ------------------------------------------------------------------
        # ACADEMIC FIX: Rapid Epsilon Decay
        # Prevent "Utility Bleed" by using a rapid exponential decay for exploration,
        # allowing the algorithm to exploit its learned intelligence much sooner.
        # ------------------------------------------------------------------
        if policy in ["AC_DL_MATCH", "ORIGINAL_DL_MATCH", "AC_NO_LR", "AC_NO_DECAY"]:
            epsilon = max(0.01, 0.5 * math.exp(-0.1 * t))
        else:
            epsilon = 0
         
        # B. Decentralized SDN Control Plane Execution
        for sdn in sdns:
            sdn_rejects = 0
            local_fogs = sdn.local_fogs.copy()
            neighbor_fogs = []
            for n_sdn in sdn.neighbor_sdns:
                neighbor_fogs.extend(n_sdn.local_fogs)
                
            edge_activations = np.random.poisson(lam=0.8, size=len(sdn.local_edges))
            for idx, edge in enumerate(sdn.local_edges):
                if edge_activations[idx] == 0:
                    continue
                    
                t_global_active_edges += 1
                edge.weights = [random.randint(1, 10) for _ in range(4)]
                matched = False
                
                # Dynamic scope: task checks local topology, falls back to neighbors
                current_local = local_fogs
                current_neighbor = neighbor_fogs
                
                for stage in range(K_MAX_RETRIES):
                    if not current_local and not current_neighbor:
                        break
                        
                    best_fog, best_utility, best_prob = run_policy(
                        policy, sdn, edge, current_local, current_neighbor, t, epsilon, max_total_fogs
                    )
                    
                    if not best_fog:
                        if _log_file_handle: log_message(f"   [ABORT] Edge-{edge.id} completely exhausted local/neighbor fogs at Stage {stage}.")
                        break # Policy actively opted for cloud escalation
                        
                    is_neighbor = best_fog not in sdn.local_fogs
                    outcome = best_fog.simulate_real_outcome()
                    
                    # True Referee Utility
                    m_ref = sdn.get_metrics(edge.id, best_fog) if not is_neighbor else best_fog.sdn.get_metrics(edge.id, best_fog)
                    true_utility = get_utility(m_ref["norm_delay"], m_ref["norm_energy"], m_ref["norm_rel"], m_ref["norm_cost"], edge.weights)
                    if is_neighbor: true_utility *= (1.0 - 0.20)
                    
                    # Federated Learning - SDN learns from node experiences
                    if policy in ["AC_DL_MATCH", "ORIGINAL_DL_MATCH", "AC_NO_LR", "AC_NO_DECAY"]:
                        features = [best_utility, m_ref["last_pi"], best_fog.resources_left]
                        sdn.domain_db.append((features, outcome))
                        
                    if policy == "DRL":
                        train_DRL(outcome, true_utility)
                        
                    if outcome == 1:
                        m_ref["successes"] += 1
                        best_fog.active_tasks += 1
                        best_fog.resources_left = max(0.0, best_fog.resources_left - (1.0 / best_fog.capacity))
                        m_ref["last_pi"] = 0.9 * m_ref["last_pi"] + 0.1 * 1.0
                        t_global_delay += m_ref["delay"]
                        t_global_energy += m_ref["energy"]
                        t_global_cost += best_fog.cost
                        cum_utility += true_utility
                        m_ref["last_time"] = t
                        matched = True
                        if _log_file_handle: log_message(f"   [SUCCESS] Edge-{edge.id} -> Fog-{best_fog.id} | Stage: {stage} | Util: {true_utility:.3f} | Prob: {best_prob:.3f} | Delay: {m_ref['delay']:.1f}ms")
                        break 
                    else:
                        m_ref["failures"] += 1
                        m_ref["last_pi"] = 0.9 * m_ref["last_pi"] + 0.1 * 0.0 # multiplied by 0.0 for visual symmetry
                        t_global_delay += (m_ref["delay"] * 0.05) # rejection overhead
                        m_ref["last_time"] = t
                        if best_fog in current_local:
                            if current_local is local_fogs: current_local = local_fogs.copy()
                            current_local.remove(best_fog)
                        if best_fog in current_neighbor:
                            if current_neighbor is neighbor_fogs: current_neighbor = neighbor_fogs.copy()
                            current_neighbor.remove(best_fog)
                        if _log_file_handle: log_message(f"   [REJECT] Edge-{edge.id} -> Fog-{best_fog.id} | Stage: {stage} | Capacity/Resource limit reached.")
                        
                if not matched:
                    if _log_file_handle: log_message(f"   [CLOUD ESCALATION] Edge-{edge.id} failed all K={K_MAX_RETRIES} attempts. Routed to Cloud (Penalty Applied).")
                    sdn_rejects += 1
                    t_global_rejects += 1
                    t_global_delay += CLOUD_DELAY_PENALTY
                    t_global_energy += CLOUD_ENERGY_PENALTY
                    t_global_cost += CLOUD_COST_PENALTY
                
            # Federated Elasticity (Per SDN Domain)
            p_sdn_reject = sdn_rejects / max(1, len(sdn.local_edges))
            if scale_out(p_sdn_reject, REJECT_THRESHOLD) and len(sdn.local_fogs) < env_config["MAX_SDN_FOGS"]:
                new_id = len(fogs)
                new_fog = FogNode(new_id, quality, sdn_id=sdn.id, capacity=fog_capacity)
                new_fog.sdn = sdn
                fogs.append(new_fog)
                sdn.local_fogs.append(new_fog)
                log_message(f"[!] SCALE OUT: SDN-{sdn.id} added Fog-{new_id}")
            else:
                utilization = sum([f.active_tasks/f.capacity for f in sdn.local_fogs]) / len(sdn.local_fogs) if sdn.local_fogs else 0
                sdn.util_window.append(utilization)
                if scale_in(sdn.util_window, UTIL_THRESHOLD) and len(sdn.local_fogs) > (env_config["MIN_FOGS"] // env_config["NUM_SDNS"]):
                    f_remove = min(sdn.local_fogs, key=lambda f: f.active_tasks)
                    sdn.local_fogs.remove(f_remove)
                    fogs.remove(f_remove)
                    sdn.util_window.clear()
                    log_message(f"[!] SCALE IN: SDN-{sdn.id} removed Fog-{f_remove.id}")
                    
            if policy in ["AC_DL_MATCH", "ORIGINAL_DL_MATCH", "AC_NO_LR", "AC_NO_DECAY"] and t % 5 == 0:
                sdn.learn_from_domain()
                
        # C. Global Aggregation
        active_count = t_global_active_edges if t_global_active_edges > 0 else 1
        metrics["acc_rate"].append(1.0 - (t_global_rejects / active_count))
        metrics["delay"].append(t_global_delay / active_count)
        metrics["utility"].append(cum_utility)
        metrics["energy"].append(t_global_energy / active_count)
        metrics["cost"].append(t_global_cost / active_count)
        metrics["time"].append(t)
        
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AC-DL Match Federated SDN Simulator")
    parser.add_argument("--tests", action="store_true", help="Run full benchmarking suite (without META_PSO).")
    parser.add_argument("--demo", action="store_true", help="Scale-down topology for fast presentation (includes META_PSO).")
    parser.add_argument("--best", action="store_true", help="Simulate highly resourceful systems.")
    parser.add_argument("--worst", action="store_true", help="Simulate tight resource constraints.")
    parser.add_argument("--average", action="store_true", help="Simulate standard baseline constraints (Default).")
    parser.add_argument("--stress", action="store_true", help="Scale up devices massively.")
    parser.add_argument('--real', action='store_true', help='Inject real Alibaba Cluster Traces.')
    parser.add_argument("--ablation", action="store_true", help="Add ablation study variants (requires --tests --demo).")
    parser.add_argument("--drl", action="store_true", help="Force include DRL baseline in full tests (WARNING: Very Slow).")
    parser.add_argument("--pso", action="store_true", help="Force include PSO baseline in full tests (WARNING: Very Slow).")
    parser.add_argument("--run", type=int, default=1, help="Number of evaluation runs to average.")
    args = parser.parse_args()

    quality = "best" if args.best else "worst" if args.worst else "average"
    
    trace_df = None
    if getattr(args, 'real', False):
        import pandas as pd
        try:
            trace_df = pd.read_csv('traces/alibaba_clean_trace.csv')
            print(f"[INFO] Alibaba Trace Loaded: {len(trace_df)} timesteps.")
        except FileNotFoundError:
            print("[ERROR] Run generate_data.py first!")
            exit(1)

    TOTAL_MC_RUNS = args.run
    env_config = get_env_config(args)
    sim_slots = env_config["NUM_SLOTS"]
    
    # Only generate detailed .log files when explicitly running a fast demo to avoid heavy I/O bottleneck
    if getattr(args, 'demo', False):
        init_logging()

    if args.tests:
        data_mode = 'Real Alibaba Traces' if trace_df is not None else 'Synthetic'
        print(f"\n[INIT] Running Evaluation Suite | Runs: {TOTAL_MC_RUNS} | Slots: {sim_slots} | Quality: {quality} | Data: {data_mode} | Mode: Federated Multi-SDN")
        
        # --demo includes all policies automatically
        # --tests without --demo excludes slow algorithms by default, unless explicitly requested via flags
        policies = ["RANDOM", "GREEDY", "BLM_TS", "MV_UCB", "DRL", "META_PSO", "ORIGINAL_DL_MATCH", "AC_DL_MATCH"]
        if not getattr(args, 'demo', False):
            if "META_PSO" in policies and not getattr(args, 'pso', False): 
                policies.remove("META_PSO")
            if "DRL" in policies and not getattr(args, 'drl', False): 
                policies.remove("DRL")
        
        # Ablation only works in demo mode (fast enough to include extra policies)
        if getattr(args, 'ablation', False) and getattr(args, 'demo', False):
            policies.extend(["AC_NO_LR", "AC_NO_DECAY"])
            
        averaged_metrics = {p: {"acc_rate": np.zeros(sim_slots), "delay": np.zeros(sim_slots), "utility": np.zeros(sim_slots), "energy": np.zeros(sim_slots), "cost": np.zeros(sim_slots), "time": list(range(1, sim_slots + 1))} for p in policies}
        
        for run in range(TOTAL_MC_RUNS):
            print(f"\n{'='*20} STARTING SIMULATION EPOCH {run+1}/{TOTAL_MC_RUNS} {'='*20}")
            seed = random.randint(0, 1000000)
            
            for p in policies:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                
                start_time = time.perf_counter()
                run_data = run_simulation(p, env_config, quality, trace_df) 
                exec_time = time.perf_counter() - start_time
                
                run_acc = np.mean(run_data['acc_rate']) if run_data['acc_rate'] else 0
                run_del = np.mean(run_data['delay']) if run_data['delay'] else 1e-5
                run_util = run_data['utility'][-1] if run_data['utility'] else 0
                run_energy = np.mean(run_data['energy']) if run_data['energy'] else 0
                run_cost = np.mean(run_data['cost']) if run_data['cost'] else 0
                print(f"[{p:<17}] | Exec: {exec_time:>5.2f}s | Util: {run_util:<7.2f} | Acc: {run_acc*100:<5.1f}% | Delay: {run_del:<6.2f}ms | Energy: {run_energy:<7.2f} | Cost: {run_cost:<6.2f}")
                
                averaged_metrics[p]["acc_rate"] += np.array(run_data["acc_rate"])
                averaged_metrics[p]["delay"] += np.array(run_data["delay"])
                averaged_metrics[p]["utility"] += np.array(run_data["utility"])
                averaged_metrics[p]["energy"] += np.array(run_data["energy"])
                averaged_metrics[p]["cost"] += np.array(run_data["cost"])

        for p in policies:
            averaged_metrics[p]["acc_rate"] /= TOTAL_MC_RUNS
            averaged_metrics[p]["delay"] /= TOTAL_MC_RUNS
            averaged_metrics[p]["utility"] /= TOTAL_MC_RUNS
            averaged_metrics[p]["energy"] /= TOTAL_MC_RUNS
            averaged_metrics[p]["cost"] /= TOTAL_MC_RUNS

        try:
            plotter = BenchmarkPlotter(averaged_metrics, timestamp)
            plotter.generate_all_plots()
            print("\n[SUCCESS] Benchmarking complete. Visualizations exported.")
        except Exception as e:
            print(f"\n[WARNING] Plotting tools failed to load. Error: {e}")
            
        print("\n" + "="*60)
        print(f"{'FINAL BENCHMARKING RESULTS (AVERAGED OVER '+str(TOTAL_MC_RUNS)+' RUNS)':^60}")
        print("="*60)
        
        category_winners = {
            "Acc Rate": {"val": -1, "policy": ""},
            "Delay": {"val": float('inf'), "policy": ""},
            "Energy": {"val": float('inf'), "policy": ""},
            "Cost": {"val": float('inf'), "policy": ""},
            "Utility": {"val": -1, "policy": ""}
        }
        
        for p, m in averaged_metrics.items():
            avg_acc = np.mean(m['acc_rate'])
            avg_del = np.mean(m['delay'])
            final_util = m['utility'][-1] 
            final_nrg = np.mean(m['energy'])
            final_cst = np.mean(m['cost'])
            
            print(f"[{p:<17}] | Util: {final_util:<7.2f} | Acc: {avg_acc*100:<5.1f}% | Delay: {avg_del:<6.2f}ms | Energy: {final_nrg:<7.2f} | Cost: {final_cst:<6.2f}")
            
            if avg_acc > category_winners["Acc Rate"]["val"]: category_winners["Acc Rate"] = {"val": avg_acc, "policy": p}
            if avg_del < category_winners["Delay"]["val"]: category_winners["Delay"] = {"val": avg_del, "policy": p}
            if final_nrg < category_winners["Energy"]["val"]: category_winners["Energy"] = {"val": final_nrg, "policy": p}
            if final_cst < category_winners["Cost"]["val"]: category_winners["Cost"] = {"val": final_cst, "policy": p}
            if final_util > category_winners["Utility"]["val"]: category_winners["Utility"] = {"val": final_util, "policy": p}

        print("-" * 60)
        print(f"🏆 Highest Acceptance Rate : {category_winners['Acc Rate']['policy']} ({category_winners['Acc Rate']['val']*100:.1f}%)")
        print(f"🏆 Lowest Average Delay    : {category_winners['Delay']['policy']} ({category_winners['Delay']['val']:.2f}ms)")
        print(f"🏆 Lowest Average Energy   : {category_winners['Energy']['policy']} ({category_winners['Energy']['val']:.2f})")
        print(f"🏆 Lowest Average Cost     : {category_winners['Cost']['policy']} ({category_winners['Cost']['val']:.2f})")
        print(f"🏆 Highest Total Utility   : {category_winners['Utility']['policy']} ({category_winners['Utility']['val']:.2f})")
        print("=" * 60)
        
        best_utl = max(averaged_metrics.items(), key=lambda x: x[1]['utility'][-1])
        winner_msg = f"🌟 OVERALL OBJECTIVE WINNER (MAX UTILITY): {best_utl[0]} 🌟"
        print(f"{winner_msg:^60}")
        print("=" * 60 + "\n")
        
    else:
        data_mode = 'Real Alibaba Traces' if trace_df is not None else 'Synthetic'
        print(f"Running AC_DL_MATCH (Production Mode | Runs: {TOTAL_MC_RUNS} | Quality: {quality} | Data: {data_mode})...")
        for run in range(TOTAL_MC_RUNS):
            seed = random.randint(0, 1000000)
            random.seed(seed)
            np.random.seed(seed)
            run_simulation("AC_DL_MATCH", env_config, quality, trace_df)