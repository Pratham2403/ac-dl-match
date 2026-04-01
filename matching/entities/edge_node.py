import random

class EdgeNode:
    def __init__(self, node_id, weights, fogs, quality="average"):
        self.id = node_id
        self.weights = weights
        self.quality = quality
        self.interaction_history = {}
        self.fog_metrics = {}
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

        self.fog_metrics[fog.id] = {
            "delay": delay,
            "energy": energy,
            "reliability": reliability,
            "last_pi": 0.5,
            "successes": 0,
            "failures": 0,
            "hops": random.randint(1, 3)
        }