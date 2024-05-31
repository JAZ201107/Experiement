import random
from collections import namedtuple

Transition = namedtuple(
    "transition", ("state", "action", "reward", "state_", "done", "raw_state")
)


class ReplayBuffer:
    def __init__(self, size=1e6):
        self.buffer = []
        self.max_size = size
        self.pointer = 0

    def add_transition(self, *args):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)

        self.buffer[self.pointer] = Transition(*args)
        self.pointer = int((self.pointer + 1) % self.max_size)

    def sample_batch(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*batch))

        return batch

    def __len__(self):
        return len(self.buffer)
