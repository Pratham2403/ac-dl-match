from collections import deque
from sklearn.linear_model import LogisticRegression
import numpy as np
import random

class SDNController:
    """Represents a regional control plane managing local IoT task generators and Fog Computing nodes."""
    def __init__(self, sdn_id, env_config, quality="average"):
        self.id = sdn_id
        self.bounds = env_config
        self.quality = quality
        
        self.local_fogs = []
        self.local_edges = []
        self.neighbor_sdns = [] # 1-hop away SDNs for federated East-West scaling
        
        # Centralized Domain memory replacing Edge device memory (resolving memory bottleneck)
        self.domain_db = deque(maxlen=2000)
        self.util_window = deque(maxlen=5)
        self.domain_coeffs = ([1.0, 1.0, 1.0], 0.0) # (feature_coefficients, intercept)
        self.model = LogisticRegression(warm_start=True, max_iter=200, C=10.0)
        
        # The SDN holds the Global Topology Network delays & SLA metrics
        self.network_topology = {} # Maps: (edge_id, fog_id) -> {metrics_dict}
        
    def generate_network_metrics(self, edge_id, fog):
        """Intellectual Property: The SDN generates and knows the network path parameters, NOT the IoT device or Fog."""
        if (edge_id, fog.id) in self.network_topology:
            return self.network_topology[(edge_id, fog.id)]
            
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
            
        # Physical SLA-Driven Bounding
        norm_delay = max(0.0, 1.0 - (delay / self.bounds['MAX_DELAY']))
        norm_energy = max(0.0, 1.0 - (energy / self.bounds['MAX_ENERGY']))
        norm_rel = min(reliability, 1.0)
        
        metrics = {
            "delay": delay,
            "energy": energy,
            "reliability": reliability,
            "norm_delay": norm_delay,
            "norm_energy": norm_energy,
            "norm_rel": norm_rel,
            "successes": 0,
            "failures": 0,
            "last_pi": 0.5,
            "last_time": 0
        }
        self.network_topology[(edge_id, fog.id)] = metrics
        return metrics
        
    def get_metrics(self, edge_id, fog):
        """Lookup or compute topological metrics for a flow path. Updates dynamic cost/resources."""
        metrics = self.generate_network_metrics(edge_id, fog)
        # Update transient physical metrics that the SDN queries dynamically from the Fog Data Plane
        metrics["norm_cost"] = max(0.0, 1.0 - (fog.cost / self.bounds['MAX_COST']))
        return metrics

    def learn_from_domain(self):
        """Train domain logistic regression if enough heterogeneous data is gathered."""
        if len(self.domain_db) > 20:
            X = [record[0] for record in self.domain_db]
            y = [record[1] for record in self.domain_db]
            if len(set(y)) >= 2:
                self.model.fit(np.array(X), np.array(y))
                self.domain_coeffs = (self.model.coef_[0], self.model.intercept_[0])
