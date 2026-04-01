import random

class FogNode:
    def __init__(self, node_id):
        self.id = node_id
        self.cost = random.uniform(1, 10)
        self.resources_left = random.uniform(0.8, 1.0)
        self.active_tasks = 0
        self.capacity = 5

    def update_state(self):
        tasks_to_complete = 0
        for _ in range(self.active_tasks):
            if random.random() < 0.4:  
                tasks_to_complete += 1
                
        self.active_tasks -= tasks_to_complete
        self.resources_left = min(1.0, self.resources_left + (tasks_to_complete * (1/self.capacity)))
        self.resources_left = max(0.0, min(1.0, self.resources_left + random.uniform(-0.05, 0.05)))

    def simulate_real_outcome(self, task_reliability=0.9):
        # Improved: Outcome depends on node resources AND task connection reliability
        if self.active_tasks < self.capacity and self.resources_left >= 0.1:
            if random.random() < self.resources_left and random.random() < task_reliability:
                return 1
        return 0