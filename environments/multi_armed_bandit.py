"""
Multi-armed bandit environment

Sam Connolly 2019
"""
import numpy as np


class MultiArmedBandit:
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.chances = np.random.random(3)

    def pull_arm(self, n: int):
        p = np.random.random()
        if p < self.chances[n]:
            return True
        else:
            return False
