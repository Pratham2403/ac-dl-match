import random
import numpy as np

class FogNode:
    """Represents a fog computing server with dynamic capacity and stochastic task completion."""
    def __init__(self, node_id, quality="average"):
        self.id = node_id
        if quality == "best":
            self.cost = random.uniform(1, 5)
            self.resources_left = random.uniform(0.9, 1.0)
            self.capacity = 10
        elif quality == "worst":
            self.cost = random.uniform(5, 15)
            self.resources_left = random.uniform(0.4, 0.7)
            self.capacity = 3
        else: # average
            self.cost = random.uniform(1, 10)
            self.resources_left = random.uniform(0.8, 1.0)
            self.capacity = 5
            
        self.active_tasks = 0

    def update_state(self):
        """Simulate stochastic task completion and resource fluctuation."""
        tasks_to_complete = np.random.binomial(self.active_tasks, 0.4)
                
        self.active_tasks -= tasks_to_complete
        self.resources_left = min(1.0, self.resources_left + (tasks_to_complete * (1/self.capacity)))
        self.resources_left = max(0.0, min(1.0, self.resources_left + random.uniform(-0.05, 0.05)))

    def simulate_real_outcome(self, task_reliability=0.9):
        """Returns 1 (accepted) or 0 (rejected) based on resources and link reliability."""
        if self.active_tasks < self.capacity and self.resources_left >= 0.1:
            if random.random() < self.resources_left and random.random() < task_reliability:
                return 1
        return 0