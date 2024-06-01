class RunningMemory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.obs_ = []
        self.values = []
        self.values_ = []
        self.gae_ = []
