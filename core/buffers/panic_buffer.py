from .replay_buffer import ReplayBuffer

class PanicBuffer(ReplayBuffer):
    """ 
    PanicBuffer is ReplayBuffer with the following distinctions:
    - sample() removes transitions from memory.
    - fill() fills the buffer with transitions having surprising (far off) rewards, namely:
      * `panic_value` for negative rewards.
      * `euphoria_value` for positive rewards.
    """
    def sample(self, batch_size):
        """ get a sample and remove items from memory """
        if len(self.memory) < batch_size:
            return None
        sample = [self.memory.pop(random.randrange(len(self.memory))) for _ in range(batch_size)]
        self.position = len(self.memory)
        return sample

    def fill(self, transitions, panic_value, euphoria_value):
        """ fill buffer with transitions that has reward < panic_value
        and reward > euphoria_value """
        panic_generator = filter(lambda t: t.reward < panic_value or t.reward > euphoria_value, transitions)
        for t in panic_generator:
            self.push(*t)