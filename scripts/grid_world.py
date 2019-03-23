"""
Tic-tac-toe agent

Sam Connolly 2019
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union
import os


class GridWorld:
    def __init__(self, grid_map: np.ndarray = None, penalty=-0.1):
        if grid_map is None:
            self.grid_map = np.array([[0, 0, 0, -1],
                                      [0, 1, 0, -1],
                                      [2, 0, 0, 0]])
        else:
            self.grid_map = grid_map

        self.state = self.grid_map
        self.rewards = self.get_rewards(self.grid_map, penalty)
        self.actions = self.get_actions(self.grid_map)

    @staticmethod
    def get_rewards(grid_map, penalty=-0.1):
        rewards = grid_map.copy().astype(float)
        if penalty != 0:
            rewards[rewards == 0] = penalty
        rewards[(rewards == 1) | (rewards == 2)] = 0

        return rewards

    @staticmethod
    def get_actions(grid_map):
        all_actions = np.empty(grid_map.shape, dtype=list)
        for i in range(grid_map.shape[0]):
            for j in range(grid_map.shape[1]):
                all_actions[i, j] = []
                if grid_map[i, j] != 1:
                    if i != 0:
                        if grid_map[i - 1, j] != 1:
                            all_actions[i, j].append("L")
                    if i != grid_map.shape[0] - 1:
                        if grid_map[i + 1, j] != 1:
                            all_actions[i, j].append("R")
                    if j != 0:
                        if grid_map[i, j - 1] != 1:
                            all_actions[i, j].append("D")
                    if j != grid_map.shape[1] - 1:
                        if grid_map[i, j + 1] != 1:
                            all_actions[i, j].append("U")

        return all_actions

    def plot_state(self, state=None):
        if state is None:
            state = self.state

        self.draw_board(state)

        plt.show()

    @staticmethod
    def draw_player(pos, axes):
        x, y = pos
        circle = plt.Circle((x + 0.5, y + 0.5), 0.4, color='b', fill=True)
        axes.add_artist(circle)

    @staticmethod
    def draw_value_state(x, y, v, axes):
        if v != -1:
            axes.text(x + 0.5, y + 0.5, "{:.2f}".format(v), color='k',
                      horizontalalignment='center', verticalalignment='center')

            colour = plt.cm.RdYlGn(v)
            rect = plt.Rectangle((x, y), 1, 1, color=colour, fill=True, alpha=0.5)
            axes.add_artist(rect)

    def draw_actions(self, axes, actions: np.ndarray = None):
        if actions is None:
            actions = self.actions

        for i in range(actions.shape[0]):
            for j in range(actions.shape[1]):
                for a in actions[i, j]:
                    self.draw_action(axes, a, i, j)

    @staticmethod
    def draw_action(axes, action, i, j):
        if action == "U":
            axes.arrow(i + 0.5, j + 0.75, 0, 0.15, width=0.02, color="g")
        elif action == "D":
            axes.arrow(i + 0.5, j + 0.25, 0, -0.15, width=0.02, color="g")
        elif action == "R":
            axes.arrow(i + 0.75, j + 0.5, 0.15, 0, width=0.02, color="g")
        elif action == "L":
            axes.arrow(i + 0.25, j + 0.5, -0.15, 0, width=0.02, color="g")

    def draw_board(self, state=None, axes=None, value_state=None, draw_actions=True):
        if state is None:
            state = self.state

        if axes is None:
            _, axes = plt.subplots(figsize=(10, 8))

        # lines
        for i in range(state.shape[0] + 1):
            axes.plot([i, i], [0, state.shape[1]], color='k')
        for j in range(state.shape[1] + 1):
            axes.plot([0, state.shape[0]], [j, j], color='k')

        self.draw_player(np.where(state == 2), axes)

        # areas & value states
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j] == 1 or state[i, j] == -1:
                    if state[i, j] == 1:
                        colour = "grey"
                    elif state[i, j] == -1:
                        colour = "r"
                    rect = plt.Rectangle((i, j), 1, 1, color=colour, fill=True)
                    axes.add_artist(rect)
                if value_state is not None:
                    self.draw_value_state(i, j, value_state[i, j], axes)
                if draw_actions:
                    self.draw_actions(axes)
        #
        # # state outcome
        # w = self.check_state(state)
        # if w == 1:
        #     axes.set_title("X wins")
        # elif w == 2:
        #     axes.set_title("O wins")
        # elif w == -1:
        #     axes.set_title("draw")
        # else:
        #     axes.set_title("No win/draw")


# class Agent
#
#
# class Policy




if __name__ == '__main__':
    gridworld = GridWorld()

    print(gridworld.grid_map)
    print(gridworld.rewards)
    print(gridworld.actions)
    gridworld.plot_state()
