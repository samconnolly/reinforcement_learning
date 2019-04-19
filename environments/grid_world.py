"""
Grid world environment

Sam Connolly 2019
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class GridWorld:
    def __init__(self, grid_map: np.ndarray = None, penalty=-0.1, start_pos: Tuple[int] = (0, 0),
                 windy: bool = False):
        if grid_map is None:
            grid_map = np.array([[0, 0, 0, 1],
                                 [0, -2, 0, -1],
                                 [0, 0, 0, 0]])

        self.grid_map = grid_map.T[:, ::-1]

        self._all_states = [(x, y) for y in range(grid_map.shape[0]) for x in range(grid_map.shape[1])]

        self.terminal = ((self.grid_map == 1) | (self.grid_map == -1))

        self.start_pos = start_pos
        self.state = start_pos
        self.penalty = penalty
        self.rewards = self.get_rewards(self.grid_map, penalty)
        self.actions = self.get_actions(self.grid_map)
        self.last_move = None
        self.windy = windy

    @staticmethod
    def get_rewards(grid_map, penalty = -0.1):
        rewards = grid_map.copy().astype(float)
        if penalty != 0:
            rewards[rewards == 0] = penalty
        rewards[(rewards == -2)] = 0

        return rewards

    @staticmethod
    def get_actions(grid_map):
        all_actions = np.empty(grid_map.shape, dtype=list)
        for i in range(grid_map.shape[0]):
            for j in range(grid_map.shape[1]):
                all_actions[i, j] = []
                if grid_map[i, j] == 0:
                    if i != 0:
                        if grid_map[i - 1, j] != -2:
                            all_actions[i, j].append("L")
                    if i != grid_map.shape[0] - 1:
                        if grid_map[i + 1, j] != -2:
                            all_actions[i, j].append("R")
                    if j != 0:
                        if grid_map[i, j - 1] != -2:
                            all_actions[i, j].append("D")
                    if j != grid_map.shape[1] - 1:
                        if grid_map[i, j + 1] != -2:
                            all_actions[i, j].append("U")

        return all_actions

    def move(self, m):
        if m in self.actions[self.state]:
            if m == "U":
                self.state = (self.state[0], self.state[1] + 1)
            elif m == "D":
                self.state = (self.state[0], self.state[1] - 1)
            elif m == "L":
                self.state = (self.state[0] - 1, self.state[1])
            elif m == "R":
                self.state = (self.state[0] + 1, self.state[1])

        return self.check_state(self.state), self.is_terminal(self.state)

    def undo_move(self):
        if self.last_move == "U":
            move = "D"
        elif self.last_move == "D":
            move = "U"
        elif self.last_move == "R":
            move = "L"
        elif self.last_move == "L":
            move = "R"

        if move in self.actions[self.state]:
            if move == "U":
                self.state = (self.state[0], self.state[1] + 1)
            elif move == "D":
                self.state = (self.state[0], self.state[1] - 1)
            elif move == "L":
                self.state = (self.state[0] - 1, self.state[1])
            elif move == "R":
                self.state = (self.state[0] + 1, self.state[1])
        else:
            raise AttributeError("Bad undo.")

        return self.check_state(self.state), self.is_terminal(self.state)

    def check_move(self, state, m):
        actions = self.actions[state]
        n_actions = len(actions)
        if self.windy:
            #probs = np.ones(n_actions) * 0.5 / (n_actions - 1)
            probs = np.ones(n_actions) * 0.5 / 3
            probs[np.where(np.array(actions) == m)] = 0.5
        else:
            probs = np.zeros(n_actions)
            probs[np.where(np.array(actions) == m)] = 1.0

        next_states = []
        rewards = []
        terminal = []
        for action in actions:
            if action == "U":
                next_states.append((state[0], state[1] + 1))
            elif action == "D":
                next_states.append((state[0], state[1] - 1))
            elif action == "L":
                next_states.append((state[0] - 1, state[1]))
            elif action == "R":
                next_states.append((state[0] + 1, state[1]))

            rewards.append(self.check_state(next_states[-1]))
            terminal.append(self.is_terminal(next_states[-1]))

        return next_states, rewards, terminal, probs

    def check_state(self, position):
        return self.rewards[position]

    def is_terminal(self, position):
        return self.terminal[position]

    def reset_state(self):
        self.state = self.start_pos

    @property
    def all_states(self):
        return self._all_states

    # plotting
    def plot_states(self, states=None, value_states=None, policies=None):
        if states is None:
            states = [self.state]
        if value_states is None:
            value_states = [None for _ in states]
        if policies is None:
            policies = [None for _ in states]

        ns = len(states)
        nx = int(np.sqrt(ns))
        ny = ns // nx
        if ns % nx != 0:
            nx += 1

        _, axes = plt.subplots(nx, ny, figsize=(10, 8))
        try:
            if len(axes.shape) == 1:
                axes = np.array([axes])
        except AttributeError:
            axes = np.array([[axes]])

        n = 0
        for i in range(nx):
            for j in range(ny):
                self.plot_state(states[n], axes[i, j], value_state=value_states[n], policy=policies[n])
                n += 1
                if n == len(states):
                    break
            if n == len(states):
                break

        plt.show()

    def plot_state(self, position=None, axes=None, grid_map=None, value_state=None, policy=None):
        if grid_map is None:
            grid_map = self.grid_map

        if position is None:
            position = self.state

        self.draw_board(grid_map=grid_map, position=position, axes=axes, value_state=value_state, policy=policy)

    @staticmethod
    def draw_player(pos, axes):
        x, y = pos
        circle = plt.Circle((x + 0.5, y + 0.5), 0.4, color='b', fill=True, zorder=5, alpha=0.3)
        axes.add_artist(circle)

    @staticmethod
    def draw_value_state(x, y, v, axes):
        axes.text(x + 0.5, y + 0.5, "{:.2f}".format(v), color='k',
                  horizontalalignment='center', verticalalignment='center')

        colour = plt.cm.RdYlGn((v + 1) / 2)
        rect = plt.Rectangle((x, y), 1, 1, color=colour, fill=True, alpha=1)
        axes.add_artist(rect)

    def draw_actions(self, axes, actions: np.ndarray = None, policy = None):
        if actions is None:
            actions = self.actions

        for i in range(actions.shape[0]):
            for j in range(actions.shape[1]):
                policy_actions, probs = policy[i, j]
                for a in actions[i, j]:
                    if policy is not None:
                        p = probs[np.where(np.array(policy_actions) == a)[0][0]]
                    else:
                        p = None
                    self.draw_action(axes, a, i, j, p)

    @staticmethod
    def draw_action(axes, action, i, j, p: float = None):

        if p is not None:
            colour = plt.cm.PRGn(p)
        else:
            colour = "purple"

        if action == "U":
            axes.arrow(i + 0.5, j + 0.75, 0, 0.15, width=0.02, color=colour, zorder=6)
        elif action == "D":
            axes.arrow(i + 0.5, j + 0.25, 0, -0.15, width=0.02, color=colour, zorder=6)
        elif action == "R":
            axes.arrow(i + 0.75, j + 0.5, 0.15, 0, width=0.02, color=colour, zorder=6)
        elif action == "L":
            axes.arrow(i + 0.25, j + 0.5, -0.15, 0, width=0.02, color=colour, zorder=6)

    def draw_board(self,  grid_map=None, axes=None, value_state=None, draw_actions=True, position=None,
                   policy=None):
        if grid_map is None:
            grid_map = self.grid_map

        if position is None:
            position = self.state
        else:
            position = tuple(position)

        if axes is None:
            _, axes = plt.subplots(figsize=(10, 8))

        # lines
        for i in range(grid_map.shape[0] + 1):
            axes.plot([i, i], [0, grid_map.shape[1]], color='k')
        for j in range(grid_map.shape[1] + 1):
            axes.plot([0, grid_map.shape[0]], [j, j], color='k')

        self.draw_player(position, axes)

        # areas & value states
        for i in range(grid_map.shape[0]):
            for j in range(grid_map.shape[1]):
                if grid_map[i, j] != 0:
                    if grid_map[i, j] == -2:
                        colour = "grey"
                    elif grid_map[i, j] == -1 and value_state is None:
                        colour = "r"
                    elif grid_map[i, j] == 1 and value_state is None:
                        colour = "g"
                    rect = plt.Rectangle((i, j), 1, 1, color=colour, fill=True)
                    axes.add_artist(rect)
                if value_state is not None:
                    self.draw_value_state(i, j, value_state[i, j], axes)
                if draw_actions:
                    self.draw_actions(axes, policy=policy)

        # state outcome
        r = self.check_state(position)
        t = self.is_terminal(position)
        if t and r == 1:
            axes.set_title("Player wins")
        elif t and r == -1:
            axes.set_title("Player loses")
        else:
            axes.set_title("Still playing")
