import random
from collections import deque

class EdgeNode:
    """Represents an IoT task node with per-fog routing metrics and local learning state."""
    def __init__(self, node_id, weights, fogs, env_config, quality="average"):
        self.bounds = env_config  # Store the SLA limits
        self.id = node_id
        self.weights = weights
        self.quality = quality
        self.interaction_history = {}
        self.fog_metrics = {}
        # Per-device distributed learning state
        self.local_db = deque(maxlen=500)
        self.local_coeffs = [1.0, 1.0, 1.0]
        
        for f in fogs:
            self.add_fog_profile(f)

    def add_fog_profile(self, fog):
        if self.quality == "best":
            delay = random.uniform(1, 20)
            energy = random.uniform(2, 10)
            reliability = random.uniform(0.95, 0.99)
        elif self.quality == "worst":
            delay = random.uniform(20, 100)
            energy = random.uniform(10, 30)
            reliability = random.uniform(0.6, 0.8)
        else: # average
            delay = random.uniform(5, 50)
            energy = random.uniform(5, 20)
            reliability = random.uniform(0.8, 0.99)
            
        # The Edge generates real-world broadcast distances
        hops = random.randint(1, 4)

        # NEW: O(1) SLA-Driven Bounding. Clamped at 0.0 for anomalies.
        norm_delay = max(0.0, 1.0 - (delay / self.bounds['MAX_DELAY']))
        norm_energy = max(0.0, 1.0 - (energy / self.bounds['MAX_ENERGY']))
        norm_cost = max(0.0, 1.0 - (fog.cost / self.bounds['MAX_COST']))
        norm_rel = min(reliability, 1.0)

        self.fog_metrics[fog.id] = {
            "delay": delay,
            "energy": energy,
            "reliability": reliability,
            "norm_delay": norm_delay,
            "norm_energy": norm_energy,
            "norm_cost": norm_cost,
            "norm_rel": norm_rel,
            "last_pi": 0.5,
            "successes": 0,
            "failures": 0,
            "hops": hops
        }