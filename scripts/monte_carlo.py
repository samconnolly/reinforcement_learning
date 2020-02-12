"""
Monte Carlo approach (to grid world)

Sam Connolly 2019
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings
from environments.grid_world import GridWorld, Policy


class Agent:
    def __init__(self, grid_world: GridWorld, policy: Policy):
        self.grid_world = grid_world
        self.policy = policy

    def play_game(self):
        state = self.grid_world.all_states[np.random.randint(0, len(self.grid_world.all_states))]
        terminal = False

        print(state)
        self.grid_world.state = state
        while not terminal:
            action = self.policy.action(state)
            reward, terminal = self.grid_world.move(action)
            print(state)



if __name__ == '__main__':
    grid_world = GridWorld()
    policy = Policy("random_choice", grid_world)
    agent = Agent(grid_world, policy)

    agent.play_game()
