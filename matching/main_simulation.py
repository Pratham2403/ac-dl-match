import os
import random
import argparse
import numpy as np
from collections import deque
from datetime import datetime
import plotly.graph_objects as go
import plotly.io as pio
import warnings

warnings.filterwarnings('ignore')
pio.templates.default = "plotly_white"

# Import Algorithms
from algorithms.get_utility import get_utility
from algorithms.get_acceptance import get_acceptance
from algorithms.learn_from_history import learn_from_history
from algorithms.scale_out import scale_out
from algorithms.scale_in import scale_in
from algorithms.test_algorithms import run_policy, train_DRL, reset_drl

# Import Entities
from entities.fog_node import FogNode
from entities.edge_node import EdgeNode

# 1. Setup Logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = None  # Set later only if logging is needed
_log_file_handle = None

def init_logging():
    global log_file_path, _log_file_handle
    os.makedirs("logs", exist_ok=True)
    log_file_path = f"logs/simulation_{timestamp}.log"
    _log_file_handle = open(log_file_path, "a")

def log_message(msg):
    if _log_file_handle:
        _log_file_handle.write(msg + "\n")

# 2. Main Simulation Loop
def run_simulation(policy, num_slots=100, num_fogs=3, num_edges=10, quality="average"):
    reset_drl()
    log_message(f"\n{'='*20} STARTING POLICY: {policy} {'='*20}")
    
    K_MAX_RETRIES = 3
    MIN_FOGS = num_fogs
    REJECT_THRESHOLD = 0.4
    UTIL_THRESHOLD = 0.2
    
    fogs = [FogNode(i, quality=quality) for i in range(MIN_FOGS)]
    edges = [EdgeNode(i, [random.randint(1,10) for _ in range(4)], fogs, quality=quality) for i in range(num_edges)]
    
    global_db = []
    current_coeffs = [1.0, 1.0, 1.0]
    next_fog_id = MIN_FOGS
    
    metrics = {"acc_rate": [], "delay": [], "utility": [], "time": []}
    util_window = deque(maxlen=5)
    cum_utility = 0
    
    for t in range(1, num_slots + 1):
        for fog in fogs:
            fog.update_state()
            
        timeslot_requests, timeslot_rejects, timeslot_delay = 0, 0, 0
        epsilon = 1.0 / (t ** 0.6) if policy == "AC_DL_MATCH" else 0
        
        for edge in edges:
            matched = False
            available_fogs = list(fogs)
            
            # K_MAX Retry Loop
            for stage in range(K_MAX_RETRIES):
                if not available_fogs:
                    break
                
                # Evaluate algorithm execution policy
                best_fog, best_utility, best_prob = run_policy(
                    policy, available_fogs, edge, t, current_coeffs, epsilon
                )
                
                # Evaluation
                if best_fog:
                    timeslot_requests += 1
                    outcome = best_fog.simulate_real_outcome(edge.fog_metrics[best_fog.id]["reliability"])
                    
                    if policy in ["AC_DL_MATCH", "ORIGINAL_DL_MATCH"]:
                        edge.interaction_history[best_fog.id] = t
                        global_db.append([[best_utility, edge.fog_metrics[best_fog.id]["last_pi"], best_fog.resources_left], outcome])
                        
                    if policy == "DRL":
                        train_DRL(outcome, best_utility)
                    
                    if outcome == 1: # SUCCESS
                        edge.fog_metrics[best_fog.id]["successes"] = edge.fog_metrics[best_fog.id].get("successes", 0) + 1
                        best_fog.active_tasks += 1
                        best_fog.resources_left = max(0.0, best_fog.resources_left - (1/best_fog.capacity))
                        edge.fog_metrics[best_fog.id]["last_pi"] = 0.9 * edge.fog_metrics[best_fog.id]["last_pi"] + 0.1 * 1.0
                        timeslot_delay += edge.fog_metrics[best_fog.id]["delay"]
                        cum_utility += best_utility
                        log_message(f"[{t:03}] | Stage {stage+1} | Task-{edge.id:02} -> Fog-{best_fog.id:02} | Result: SUCCESS")
                        matched = True
                        break 
                    else: # FAILED (Try next stage)
                        edge.fog_metrics[best_fog.id]["failures"] = edge.fog_metrics[best_fog.id].get("failures", 0) + 1
                        edge.fog_metrics[best_fog.id]["last_pi"] = 0.9 * edge.fog_metrics[best_fog.id]["last_pi"]
                        log_message(f"[{t:03}] | Stage {stage+1} | Task-{edge.id:02} -> Fog-{best_fog.id:02} | Result: REJECTED")
                        available_fogs.remove(best_fog)
                        
            if not matched:
                timeslot_rejects += 1
                timeslot_delay += 100 # Cloud penalty
                log_message(f"[{t:03}] | Stage - | Task-{edge.id:02} -> CLOUD   | Result: ESCALATED")
                
        # Distributed Learning Weight Updates
        if policy in ["AC_DL_MATCH", "ORIGINAL_DL_MATCH"] and len(global_db) > 10 and t % 5 == 0:
            current_coeffs = learn_from_history(global_db)
            
        # Infrastructure Elasticity
        p_reject = timeslot_rejects / len(edges)
        metrics["acc_rate"].append(1.0 - p_reject)
        metrics["delay"].append(timeslot_delay / len(edges))
        metrics["utility"].append(cum_utility)
        metrics["time"].append(t)
        
        if policy == "AC_DL_MATCH":
            if scale_out(p_reject, REJECT_THRESHOLD):
                new_fog = FogNode(next_fog_id, quality=quality)
                fogs.append(new_fog)
                for edge in edges:
                    edge.add_fog_profile(new_fog)
                log_message(f"[!] SCALE OUT at T={t}: Added Fog-{new_fog.id} (Rejection={p_reject*100:.1f}%)")
                next_fog_id += 1
            else:
                util_window.append(sum([f.active_tasks/f.capacity for f in fogs]) / len(fogs))
                if scale_in(util_window, UTIL_THRESHOLD) and len(fogs) > MIN_FOGS:
                    node_to_remove = min(fogs, key=lambda f: f.active_tasks)
                    fogs.remove(node_to_remove)
                    log_message(f"[!] SCALE IN at T={t}: Removed Fog-{node_to_remove.id}")
                    util_window.clear()
                    
    log_message(f"Final Active Fog Nodes: {len(fogs)}")
    return metrics

# 3. Execution Execution
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="AC-DL Match Architecture Simulator")
    parser.add_argument("--tests", action="store_true", help="Run full benchmarking suite.")
    parser.add_argument("--best", action="store_true", help="Simulate highly resourceful systems.")
    parser.add_argument("--worst", action="store_true", help="Simulate tight resource constraints.")
    parser.add_argument("--average", action="store_true", help="Simulate standard baseline constraints (Default).")
    parser.add_argument("--stress", action="store_true", help="Scale up devices and slots massively.")
    parser.add_argument("--run", type=int, default=1, help="Number of evaluation runs to average.")
    args = parser.parse_args()

    quality = "average"
    if args.best: quality = "best"
    elif args.worst: quality = "worst"

    TOTAL_MC_RUNS = args.run
    sim_slots = 2000 if args.stress else 100
    sim_fogs = 20 if args.stress else 5
    sim_edges = 150 if args.stress else 15
    # Only create log files for non-stress runs to avoid I/O bottleneck
    if not args.stress:
        init_logging()

    if args.tests:
        print(f"\n[INIT] Running Evaluation Suite | Runs: {TOTAL_MC_RUNS} | Slots: {sim_slots} | Quality: {quality} | Stress: {args.stress}")
        
        policies = ["RANDOM", "GREEDY", "BLM_TS", "DRL", "META_PSO", "ORIGINAL_DL_MATCH", "AC_DL_MATCH"]
        averaged_metrics = {p: {"acc_rate": np.zeros(sim_slots), "delay": np.zeros(sim_slots), "utility": np.zeros(sim_slots), "time": list(range(1, sim_slots + 1))} for p in policies}
        
        for run in range(TOTAL_MC_RUNS):
            print(f"\n{'='*20} STARTING SIMULATION EPOCH {run+1}/{TOTAL_MC_RUNS} {'='*20}")
            log_message(f"\n{'='*20} STARTING SIMULATION EPOCH {run+1}/{TOTAL_MC_RUNS} {'='*20}")
            
            # Lock the initial mathematical topology for all policies within this epoch run
            seed = random.randint(0, 1000000)
            
            for p in policies:
                reset_drl()
                random.seed(seed)
                np.random.seed(seed)
                
                run_data = run_simulation(p, sim_slots, sim_fogs, sim_edges, quality) 
                
                # Intermediate calculation to replicate per-run output
                run_acc = np.mean(run_data['acc_rate']) if run_data['acc_rate'] else 0
                run_del = np.mean(run_data['delay']) if run_data['delay'] else 1e-5
                run_util = run_data['utility'][-1] if run_data['utility'] else 0
                run_score = (run_acc * run_util) / run_del if run_del > 0 else 0
                print(f"[{p:<17}] | Util: {run_util:<7.2f} | Acc: {run_acc*100:<5.1f}% | Delay: {run_del:<6.2f}ms | Score: {run_score:.2f}")
                
                averaged_metrics[p]["acc_rate"] += np.array(run_data["acc_rate"])
                averaged_metrics[p]["delay"] += np.array(run_data["delay"])
                averaged_metrics[p]["utility"] += np.array(run_data["utility"])

        for p in policies:
            averaged_metrics[p]["acc_rate"] /= TOTAL_MC_RUNS
            averaged_metrics[p]["delay"] /= TOTAL_MC_RUNS
            averaged_metrics[p]["utility"] /= TOTAL_MC_RUNS

        try:
            from utils.plotter import BenchmarkPlotter
            plotter = BenchmarkPlotter(averaged_metrics, timestamp)
            plotter.generate_all_plots()
            print("\n[SUCCESS] Benchmarking complete. Visualizations exported.")
        except Exception as e:
            print(f"\n[WARNING] Plotting tools failed to load. Error: {e}")
            
        print("\n" + "="*60)
        print(f"{'FINAL BENCHMARKING RESULTS (AVERAGED OVER '+str(TOTAL_MC_RUNS)+' RUNS)':^60}")
        print("="*60)
        
        best_composite = -float('inf')
        overall_winner = None
        category_winners = {
            "Acc Rate": {"val": -1, "policy": ""},
            "Delay": {"val": float('inf'), "policy": ""},
            "Utility": {"val": -1, "policy": ""}
        }
        
        for p, m in averaged_metrics.items():
            avg_acc = np.mean(m['acc_rate'])
            avg_del = np.mean(m['delay'])
            final_util = m['utility'][-1] 
            
            composite_score = (avg_acc * final_util) / avg_del if avg_del > 0 else 0
            
            print(f"[{p:<17}] | Util: {final_util:<7.2f} | Acc: {avg_acc*100:<5.1f}% | Delay: {avg_del:<6.2f}ms | Score: {composite_score:.2f}")
            
            if avg_acc > category_winners["Acc Rate"]["val"]: category_winners["Acc Rate"] = {"val": avg_acc, "policy": p}
            if avg_del < category_winners["Delay"]["val"]: category_winners["Delay"] = {"val": avg_del, "policy": p}
            if final_util > category_winners["Utility"]["val"]: category_winners["Utility"] = {"val": final_util, "policy": p}
            if composite_score > best_composite: best_composite, overall_winner = composite_score, p

        print("-" * 60)
        print(f"🏆 Highest Acceptance Rate : {category_winners['Acc Rate']['policy']} ({category_winners['Acc Rate']['val']*100:.1f}%)")
        print(f"🏆 Lowest Average Delay    : {category_winners['Delay']['policy']} ({category_winners['Delay']['val']:.2f}ms)")
        print(f"🏆 Highest Total Utility   : {category_winners['Utility']['policy']} ({category_winners['Utility']['val']:.2f})")
        print("=" * 60)
        winner_msg = f"🌟 OVERALL COMPOSITE WINNER: {overall_winner} 🌟"
        print(f"{winner_msg:^60}")
        print("=" * 60 + "\n")
        
        log_message("\n" + "="*40 + "\nFINAL BENCHMARKING RESULTS\n" + "="*40)
        log_message(winner_msg)
        
    else:
        print(f"Running core AC_DL_MATCH simulation (Production Mode | Runs: {TOTAL_MC_RUNS} | Quality: {quality} | Stress: {args.stress})...")
        for run in range(TOTAL_MC_RUNS):
            seed = random.randint(0, 1000000)
            random.seed(seed)
            np.random.seed(seed)
            reset_drl()
            run_simulation("AC_DL_MATCH", sim_slots, sim_fogs, sim_edges, quality)

    # Cleanup
    if _log_file_handle:
        _log_file_handle.close()

    if log_file_path:
        print(f"Simulation Execution Concluded. System Logs: '{log_file_path}'")
    else:
        print(f"Simulation Execution Concluded. (Stress mode: logging disabled for performance)")