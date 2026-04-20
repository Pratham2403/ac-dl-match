import random
import numpy as np

class FogNode:
    """Represents a fog computing server with dynamic capacity and stochastic task completion."""
    def __init__(self, node_id, quality="average", sdn_id=None, capacity=8):
        self.id = node_id
        self.sdn_id = sdn_id
        self.capacity = capacity
        
        # Quality controls physical characteristics (cost, initial resources), NOT capacity
        if quality == "best":
            self.cost = random.uniform(1, 5)
            self.resources_left = random.uniform(0.9, 1.0)
        elif quality == "worst":
            self.cost = random.uniform(5, 15)
            self.resources_left = random.uniform(0.4, 0.7)
        else: # average
            self.cost = random.uniform(1, 10)
            self.resources_left = random.uniform(0.8, 1.0)
            
        self.active_tasks = 0

    def update_state(self):
        """
        Simulate stochastic task completion and resource fluctuation. 
        Telemetery Gather in terms of real world deployment.
        """
        # Assumes each task has 40% chance of completion in this cycle.
        tasks_to_complete = np.random.binomial(self.active_tasks, 0.4)
                
        self.active_tasks -= tasks_to_complete
        # True physical state based on active tasks
        true_resources = max(0.0, 1.0 - (self.active_tasks / self.capacity))
        
        # Mean reversion (OU process) pulling the noisy state back to reality
        self.resources_left += 0.2 * (true_resources - self.resources_left)
        
        # Random fluctuation in resources (OS noise)
        self.resources_left = max(0.0, min(1.0, self.resources_left + random.uniform(-0.05, 0.05)))

    def simulate_real_outcome(self, task_reliability=0.9):
        """Returns 1 (accepted) or 0 (rejected) based on resources and link reliability."""
        if self.active_tasks < self.capacity and self.resources_left >= 0.1:
            if random.random() < self.resources_left and random.random() < task_reliability:
                return 1
        return 0