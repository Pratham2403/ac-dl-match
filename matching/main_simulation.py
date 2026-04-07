"""AC-DL-MATCH Benchmarking Simulator for Task Offloading in Dynamic Fog Computing Networks."""

import os
import time
import random
import argparse
import numpy as np
import torch
from collections import deque
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

from utils.plotter import BenchmarkPlotter
from algorithms.get_utility import get_utility
from algorithms.learn_from_history import learn_from_history, reset_learning_models
from algorithms.scale_out import scale_out
from algorithms.scale_in import scale_in
from algorithms.test_algorithms import run_policy, train_DRL, reset_drl
from entities.fog_node import FogNode
from entities.edge_node import EdgeNode

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = None
_log_file_handle = None

def get_env_config(args):
    """Dynamically generates SLA bounds and physical constraints based on the test scenario."""
    # Baseline (Normal test)
    config = {
        "NUM_SLOTS": 500, "NUM_EDGES": 15, "MIN_FOGS": 5, "MAX_SDN_FOGS": 15,
        "MAX_DELAY": 100.0, "MAX_ENERGY": 30.0, "MAX_COST": 15.0
    }
    
    if getattr(args, 'real', False):
        config.update({
            'NUM_SLOTS': 5000,
            'NUM_EDGES': 500,
            'MIN_FOGS': 20,
            'MAX_SDN_FOGS': 100,
            'MAX_DELAY': 250.0,
            'MAX_ENERGY': 50.0,
            'MAX_COST': 25.0
        })
    elif args.stress:
        config.update({
            "NUM_SLOTS": 2000, "NUM_EDGES": 150, "MIN_FOGS": 20, "MAX_SDN_FOGS": 50,
            "MAX_DELAY": 250.0, "MAX_ENERGY": 50.0, "MAX_COST": 25.0
        })
    elif args.best:
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
    reset_drl(env_config["MAX_SDN_FOGS"])
    log_message(f"\n{'='*20} STARTING POLICY: {policy} {'='*20}")
    
    K_MAX_RETRIES = 3
    K_MAX_HOPS = 3
    MIN_FOGS = env_config["MIN_FOGS"]
    REJECT_THRESHOLD = 0.1
    UTIL_THRESHOLD = 0.3
    CLOUD_DELAY_PENALTY = env_config["MAX_DELAY"] * 2.0
    CLOUD_ENERGY_PENALTY = env_config["MAX_ENERGY"] * 1.5
    CLOUD_COST_PENALTY = env_config["MAX_COST"] * 1.5
    
    fogs = [FogNode(i, quality=quality) for i in range(MIN_FOGS)]
    edges = [EdgeNode(i, [random.randint(1,10) for _ in range(4)], fogs, env_config, quality=quality) for i in range(env_config["NUM_EDGES"])]

    metrics = {"acc_rate": [], "delay": [], "utility": [], "energy": [], "cost": [], "time": []}
    util_window = deque(maxlen=5)
    cum_utility = 0
    
    for t in range(1, env_config["NUM_SLOTS"] + 1):
        for i, fog in enumerate(fogs):
            if trace_df is not None and (t - 1) < len(trace_df):
                # ALIBABA MODE: Override random stochasticity with REAL cluster data
                row = trace_df.iloc[t - 1]
                cpu_load = row['cpu_util_percent'] / 100.0
                mem_load = row['mem_util_percent'] / 100.0
                
                # Task completion still happens naturally (hardware processes jobs)
                tasks_to_complete = np.random.binomial(fog.active_tasks, 0.4)
                fog.active_tasks -= tasks_to_complete
                
                # Fog capacity is driven by real Alibaba memory constraints
                # Direct mapping: mem% used → resources available (no rescaling)
                fog.resources_left = max(0.01, 1.0 - mem_load)
                # Energy penalty scales with real Alibaba CPU utilization
                fog.cost = 5.0 + (cpu_load * 20.0)
                
                # SDN Control Plane: propagate live cost telemetry to edge nodes
                live_norm_cost = max(0.0, 1.0 - (fog.cost / env_config['MAX_COST']))
                for edge in edges:
                    edge.fog_metrics[fog.id]["norm_cost"] = live_norm_cost
            else:
                # SYNTHETIC MODE (Original np.random.binomial logic)
                fog.update_state()
            
        timeslot_rejects, timeslot_delay, timeslot_energy, timeslot_cost = 0, 0, 0, 0
        epsilon = 1.0 / (t ** 0.6) if policy in ["AC_DL_MATCH", "ORIGINAL_DL_MATCH"] else 0
        
        active_edges = 0
        for edge in edges:
            # Poisson Burst Traffic: ~45% chance an edge is idle if lam=0.8
            if np.random.poisson(lam=0.8) == 0:
                continue
                
            active_edges += 1
            
            # Dynamic Task Heterogeneity: tasks behave differently
            edge.weights = [random.randint(1, 10) for _ in range(4)]
            
            matched = False
            
            available_fogs = [f for f in fogs if edge.fog_metrics[f.id]["hops"] <= K_MAX_HOPS]
            
            # K_MAX Retry Loop
            for stage in range(K_MAX_RETRIES):
                if not available_fogs:
                    break
                
                best_fog, best_utility, best_prob = run_policy(
                    policy, available_fogs, edge, t, edge.local_coeffs, epsilon, env_config["MAX_SDN_FOGS"]
                )

                if best_fog:
                    outcome = best_fog.simulate_real_outcome(edge.fog_metrics[best_fog.id]["reliability"])

                    # 1. Calculate True Utility (The Referee / Environment Reality)
                    m_ref = edge.fog_metrics[best_fog.id]
                    true_utility = get_utility(
                        m_ref["norm_delay"], m_ref["norm_energy"],
                        m_ref["norm_rel"], m_ref["norm_cost"],
                        edge.weights, m_ref["hops"]
                    )
                    
                    if policy in ["AC_DL_MATCH", "ORIGINAL_DL_MATCH"]:
                        edge.interaction_history[best_fog.id] = t
                        edge.local_db.append([[best_utility, edge.fog_metrics[best_fog.id]["last_pi"], best_fog.resources_left], outcome])
                        
                    if policy == "DRL":
                        train_DRL(outcome, true_utility)
                    
                    if outcome == 1: # SUCCESS
                        edge.fog_metrics[best_fog.id]["successes"] += 1
                        best_fog.active_tasks += 1
                        best_fog.resources_left = max(0.0, best_fog.resources_left - (1/best_fog.capacity))
                        edge.fog_metrics[best_fog.id]["last_pi"] = 0.9 * edge.fog_metrics[best_fog.id]["last_pi"] + 0.1 * 1.0
                        timeslot_delay += edge.fog_metrics[best_fog.id]["delay"]
                        timeslot_energy += edge.fog_metrics[best_fog.id]["energy"]
                        timeslot_cost += best_fog.cost
                        cum_utility += true_utility
                        
                        log_message(f"[{t:03}] | Stage {stage+1} | Task-{edge.id:02} -> Fog-{best_fog.id:02} | Result: SUCCESS")
                        matched = True
                        break 
                    else: # FAILED (Try next stage)
                        edge.fog_metrics[best_fog.id]["failures"] += 1
                        edge.fog_metrics[best_fog.id]["last_pi"] = 0.9 * edge.fog_metrics[best_fog.id]["last_pi"]
                        
                        # Rejection delay: scales with physical hop distance
                        hop_count = edge.fog_metrics[best_fog.id]["hops"]
                        timeslot_delay += (edge.fog_metrics[best_fog.id]["delay"] * (0.05 * hop_count))
                        
                        log_message(f"[{t:03}] | Stage {stage+1} | Task-{edge.id:02} -> Fog-{best_fog.id:02} | Result: REJECTED")
                        available_fogs.remove(best_fog)
                        
            if not matched:
                timeslot_rejects += 1
                timeslot_delay += CLOUD_DELAY_PENALTY
                timeslot_energy += CLOUD_ENERGY_PENALTY
                timeslot_cost += CLOUD_COST_PENALTY

                log_message(f"[{t:03}] | Stage - | Task-{edge.id:02} -> CLOUD   | Result: ESCALATED")
                
        # Distributed weight updates (each edge trains independently)
        # Learning happens every 5 timeslots...not every timeslot, just to give flexibility to each IoT device to configure this based on individual capacities.
        if policy in ["AC_DL_MATCH", "ORIGINAL_DL_MATCH"] and t % 5 == 0:
            for edge in edges:
                if len(edge.local_db) > 10:
                    edge.local_coeffs = learn_from_history(edge.local_db, node_id=edge.id)
            
        # Infrastructure elasticity (applied to ALL policies — environment feature, not algorithmic)
        active_count = active_edges if active_edges > 0 else 1
        p_reject = timeslot_rejects / active_count
        
        avg_timeslot_energy = timeslot_energy / active_count
        avg_timeslot_cost = timeslot_cost / active_count
        
        metrics["acc_rate"].append(1.0 - p_reject)
        metrics["delay"].append(timeslot_delay / active_count)
        metrics["utility"].append(cum_utility)
        metrics["energy"].append(avg_timeslot_energy)
        metrics["cost"].append(avg_timeslot_cost)
        metrics["time"].append(t)
        
        if scale_out(p_reject, REJECT_THRESHOLD) and len(fogs) < env_config["MAX_SDN_FOGS"]:
            active_ids = {f.id for f in fogs}
            available_ids = [i for i in range(env_config["MAX_SDN_FOGS"]) if i not in active_ids]
            
            if available_ids:
                new_id = available_ids[0]
                new_fog = FogNode(new_id, quality=quality)
                fogs.append(new_fog)
                for edge in edges:
                    edge.add_fog_profile(new_fog)
                log_message(f"[!] SCALE OUT at T={t}: Added Fog-{new_fog.id} (Total: {len(fogs)}/{env_config['MAX_SDN_FOGS']})")
        else:
            util_window.append(sum([f.active_tasks/f.capacity for f in fogs]) / len(fogs))
            if scale_in(util_window, UTIL_THRESHOLD) and len(fogs) > MIN_FOGS:
                node_to_remove = min(fogs, key=lambda f: f.active_tasks)
                fogs.remove(node_to_remove)
                log_message(f"[!] SCALE IN at T={t}: Removed Fog-{node_to_remove.id}")
                util_window.clear()
                    
    log_message(f"Final Active Fog Nodes: {len(fogs)}")
    return metrics

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="AC-DL Match Architecture Simulator")
    parser.add_argument("--tests", action="store_true", help="Run full benchmarking suite.")
    parser.add_argument("--best", action="store_true", help="Simulate highly resourceful systems.")
    parser.add_argument("--worst", action="store_true", help="Simulate tight resource constraints.")
    parser.add_argument("--average", action="store_true", help="Simulate standard baseline constraints (Default).")
    parser.add_argument("--stress", action="store_true", help="Scale up devices and slots massively.")
    parser.add_argument('--real', action='store_true', help='Inject real Alibaba Cluster Traces.')
    parser.add_argument("--run", type=int, default=1, help="Number of evaluation runs to average.")
    args = parser.parse_args()

    quality = "average"
    if args.best: quality = "best"
    elif args.worst: quality = "worst"

    trace_df = None
    if getattr(args, 'real', False):
        import pandas as pd
        try:
            trace_df = pd.read_csv('traces/alibaba_clean_trace.csv')
            print(f"[INFO] Alibaba Trace Loaded: {len(trace_df)} timesteps.")
        except FileNotFoundError:
            print("[ERROR] Run extract_alibaba.py first!")
            exit(1)

    TOTAL_MC_RUNS = args.run
    env_config = get_env_config(args)
    sim_slots = env_config["NUM_SLOTS"]
    
    # Only create log files for non-stress/real runs to avoid I/O bottleneck
    if not (args.stress or getattr(args, 'real', False)):
        init_logging()

    if args.tests:
        is_stress_display = args.stress or getattr(args, 'real', False)
        print(f"\n[INIT] Running Evaluation Suite | Runs: {TOTAL_MC_RUNS} | Slots: {sim_slots} | Quality: {quality} | Stress: {is_stress_display}")
        
        policies = ["RANDOM", "GREEDY", "BLM_TS", "MV_UCB", "DRL", "META_PSO", "ORIGINAL_DL_MATCH", "AC_DL_MATCH"]
        if args.stress or getattr(args, 'real', False):
            policies.remove("META_PSO")
            
        averaged_metrics = {p: {"acc_rate": np.zeros(sim_slots), "delay": np.zeros(sim_slots), "utility": np.zeros(sim_slots), "energy": np.zeros(sim_slots), "cost": np.zeros(sim_slots), "time": list(range(1, sim_slots + 1))} for p in policies}
        
        for run in range(TOTAL_MC_RUNS):
            print(f"\n{'='*20} STARTING SIMULATION EPOCH {run+1}/{TOTAL_MC_RUNS} {'='*20}")
            log_message(f"\n{'='*20} STARTING SIMULATION EPOCH {run+1}/{TOTAL_MC_RUNS} {'='*20}")
            
            # Same seed ensures identical topology across all policies
            seed = random.randint(0, 1000000)
            
            for p in policies:
                reset_drl(env_config["MAX_SDN_FOGS"])
                reset_learning_models()
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
        
        log_message("\n" + "="*40 + "\nFINAL BENCHMARKING RESULTS\n" + "="*40)
        log_message(winner_msg)
        
    else:
        is_stress_display = args.stress or getattr(args, 'real', False)
        print(f"Running core AC_DL_MATCH simulation (Production Mode | Runs: {TOTAL_MC_RUNS} | Quality: {quality} | Stress: {is_stress_display})...")
        for run in range(TOTAL_MC_RUNS):
            seed = random.randint(0, 1000000)
            random.seed(seed)
            np.random.seed(seed)
            reset_drl(env_config["MAX_SDN_FOGS"])
            run_simulation("AC_DL_MATCH", env_config, quality, trace_df)

    # Cleanup
    if _log_file_handle:
        _log_file_handle.close()

    if log_file_path:
        print(f"Simulation Execution Concluded. System Logs: '{log_file_path}'")
    else:
        print(f"Simulation Execution Concluded. (Stress mode: logging disabled for performance)")