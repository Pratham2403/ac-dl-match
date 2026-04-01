import random

class EdgeNode:
    def __init__(self, node_id, weights, fogs):
        self.id = node_id
        self.weights = weights
        self.interaction_history = {}
        self.fog_metrics = {}
        for f in fogs:
            self.add_fog_profile(f)

    def add_fog_profile(self, fog):
        self.fog_metrics[fog.id] = {
            "delay": random.uniform(5, 50),
            "energy": random.uniform(5, 20),
            "reliability": random.uniform(0.8, 0.99),
            "last_pi": 0.5,
            "hops": random.randint(1, 3)
        }