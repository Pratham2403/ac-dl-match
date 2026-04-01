import os
import random
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
from algorithms.test_algorithms import run_policy

# Import Entities
from entities.fog_node import FogNode
from entities.edge_node import EdgeNode

# 1. Setup Logging
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = f"logs/simulation_{timestamp}.log"

def log_message(msg):
    with open(log_file_path, "a") as f:
        f.write(msg + "\n")

# 2. Main Simulation Loop
def run_simulation(policy, num_slots=100):
    log_message(f"\n{'='*20} STARTING POLICY: {policy} {'='*20}")
    
    K_MAX_RETRIES = 3
    MIN_FOGS = 3
    REJECT_THRESHOLD = 0.4
    UTIL_THRESHOLD = 0.2
    
    fogs = [FogNode(i) for i in range(MIN_FOGS)]
    edges = [EdgeNode(i, [random.randint(1,10) for _ in range(4)], fogs) for i in range(10)]
    
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
                    outcome = best_fog.simulate_real_outcome()
                    
                    if policy in ["AC_DL_MATCH", "ORIGINAL_DL_MATCH"]:
                        edge.interaction_history[best_fog.id] = t
                        global_db.append([[best_utility, edge.fog_metrics[best_fog.id]["last_pi"], best_fog.resources_left], outcome])
                    
                    if outcome == 1: # SUCCESS
                        best_fog.active_tasks += 1
                        best_fog.resources_left = max(0.0, best_fog.resources_left - (1/best_fog.capacity))
                        edge.fog_metrics[best_fog.id]["last_pi"] = 0.9 * edge.fog_metrics[best_fog.id]["last_pi"] + 0.1 * 1.0
                        timeslot_delay += edge.fog_metrics[best_fog.id]["delay"]
                        cum_utility += best_utility
                        log_message(f"[{t:03}] | Stage {stage+1} | Task-{edge.id:02} -> Fog-{best_fog.id:02} | Result: SUCCESS")
                        matched = True
                        break 
                    else: # FAILED (Try next stage)
                        edge.fog_metrics[best_fog.id]["last_pi"] = 0.9 * edge.fog_metrics[best_fog.id]["last_pi"]
                        log_message(f"[{t:03}] | Stage {stage+1} | Task-{edge.id:02} -> Fog-{best_fog.id:02} | Result: REJECTED")
                        available_fogs.remove(best_fog)
                        
            if not matched:
                timeslot_rejects += 1
                timeslot_delay += 100 # Cloud penalty
                log_message(f"[{t:03}] | Stage - | Task-{edge.id:02} -> CLOUD   | Result: ESCALATED")
                
        # Distributed Learning Weight Updates
        if policy == "AC_DL_MATCH" and len(global_db) > 10 and t % 5 == 0:
            current_coeffs = learn_from_history(global_db)
            
        # Infrastructure Elasticity
        p_reject = timeslot_rejects / len(edges)
        metrics["acc_rate"].append(1.0 - p_reject)
        metrics["delay"].append(timeslot_delay / len(edges))
        metrics["utility"].append(cum_utility)
        metrics["time"].append(t)
        
        if policy == "AC_DL_MATCH":
            if scale_out(p_reject, REJECT_THRESHOLD):
                new_fog = FogNode(next_fog_id)
                fogs.append(new_fog)
                for edge in edges:
                    edge.add_fog_profile(new_fog)
                log_message(f"[!] SCALE OUT at T={t}: Added Fog-{new_fog.id} (Rejection={p_reject*100:.1f}%)")
                next_fog_id += 1
            else:
                util_window.append(sum([f.active_tasks/f.capacity for f in fogs]) / len(fogs))
                if len(util_window) == 5 and sum(util_window)/5 < UTIL_THRESHOLD and len(fogs) > MIN_FOGS:
                    node_to_remove = min(fogs, key=lambda f: f.active_tasks)
                    fogs.remove(node_to_remove)
                    log_message(f"[!] SCALE IN at T={t}: Removed Fog-{node_to_remove.id}")
                    util_window.clear()
                    
    log_message(f"Final Active Fog Nodes: {len(fogs)}")
    return metrics

# 3. Execution Execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AC-DL Match Architecture Simulator")
    parser.add_argument("--tests", action="store_true", help="Run full benchmarking suite and plot results.")
    args = parser.parse_args()

    if args.tests:
        print("Running full benchmarking suite across all algorithms (Testing Mode)...")
        policies = ["RANDOM", "GREEDY", "BLM_TS", "ORIGINAL_DL_MATCH", "AC_DL_MATCH"]
        all_metrics = {p: run_simulation(p, 100) for p in policies}
        
        try:
            from utils.plotter import BenchmarkPlotter
            plotter = BenchmarkPlotter(all_metrics, timestamp)
            plotter.generate_all_plots()
            print("Benchmarking complete. Visualizations exported to 'results/' directory.")
        except ImportError as e:
            print(f"Warning: Could not initiate plotting tools. Ensure 'plotly' and 'kaleido' are installed. Error: {e}")
            
        # Determine Winner
        print("\n" + "="*40)
        print("FINAL BENCHMARKING RESULTS")
        print("="*40)
        
        best_policy = None
        best_utility = -float('inf')
        
        for p, m in all_metrics.items():
            avg_acc = sum(m['acc_rate']) / len(m['acc_rate']) if m['acc_rate'] else 0
            avg_del = sum(m['delay']) / len(m['delay']) if m['delay'] else 0
            final_util = m['utility'][-1] if m['utility'] else 0
            
            print(f"Algorithm: {p:<20} | Utility: {final_util:^8.2f} | Acc Rate: {avg_acc*100:>5.1f}% | Avg Delay: {avg_del:>6.2f}ms")
            
            if final_util > best_utility:
                best_utility = final_util
                best_policy = p
                
        winner_msg = f"\n>>> OVERALL WINNER: {best_policy} (Utility: {best_utility:.2f}) <<<"
        print(winner_msg)
        print("="*40 + "\n")
        
        # Log it as well
        log_message("\n" + "="*40 + "\nFINAL BENCHMARKING RESULTS\n" + "="*40)
        log_message(winner_msg)
            
    else:
        print("Running core AC_DL_MATCH simulation (Production Mode)...")
        run_simulation("AC_DL_MATCH", 100)
    
    print(f"Simulation Execution Concluded. System Logs: '{log_file_path}'")