import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return []
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)