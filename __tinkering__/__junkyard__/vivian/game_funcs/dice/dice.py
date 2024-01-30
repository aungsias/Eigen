import numpy as np

class TwoDice:

    def __init__(self):
        self.dist = {"fair": 0.49, "biased": 0.51}

    def choose_roll_type(self):
        r, c = np.random.rand(), 0

        for t, p in self.dist.items():
            c += p
            if c >= r:
                return t

    def roll(self):
        type_of_roll = self.choose_roll_type()
        return [np.random.randint(1, 7) for _ in range(2)] if type_of_roll == "fair" else [np.random.randint(1, 7)] * 2