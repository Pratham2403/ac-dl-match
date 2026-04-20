import random
from collections import deque

class EdgeNode:
    """Represents a lightweight IoT task generator without routing overhead."""
    def __init__(self, node_id, weights, bounds, quality="average"):
        self.id = node_id
        self.weights = weights
        self.bounds = bounds
        self.quality = quality
        
        # Pointer to the managing SDN domain
        self.sdn = None 