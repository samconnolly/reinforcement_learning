"""
Tic-tac-toe environment

Sam Connolly 2019
"""
import numpy as np
import matplotlib.pyplot as plt


class TicTacToe:
    def __init__(self, rand_init: bool = False):
        self.state = np.zeros((3, 3))
        self.piece_dict = {"x": 1, "o": 2}

        # for testing
        if rand_init:
            for i in range(3):
                for j in range(3):
                    self.state[i, j] = np.random.randint(0, 3)

    def place(self, x: int, y: int, c: str):
        # place piece
        self.state[x, y] = self.piece_dict[c]

        # check for win/draw
        return self.check_state()

    def check_state(self, test_state: np.array = None):
        if test_state is None:
            test_state = self.state

        # check for win
        for d in range(1, 3):
            for i in range(3):
                # rows, columns, tl-br diagonal, bl-tr diagonal
                if np.all(test_state[i] == d)or \
                        np.all(test_state[:, i] == d) or \
                        np.all(test_state[(np.array([0, 1, 2]), np.array([0, 1, 2]))] == d) or \
                        np.all(test_state[(np.array([0, 1, 2]), np.array([2, 1, 0]))] == d):
                    return d
        else:
            # check for draw
            if np.all(test_state != 0):
                return -1
            # no win/draw
            else:
                return 0

    def clear_board(self):
        self.state = np.zeros((3, 3))

    @staticmethod
    def draw_piece(x, y, c, axes):
        if c == "x" or c == 1:
            axes.plot([0.1 + x, 0.9 + x], [0.1 + y, 0.9 + y], color='b')
            axes.plot([0.1 + x, 0.9 + x], [0.9 + y, 0.1 + y], color='b')
        elif c == "o" or c == 2:
            circle = plt.Circle((x + 0.5, y + 0.5), 0.4, color='r', fill=False)
            axes.add_artist(circle)

    @staticmethod
    def draw_value_state(x, y, v, axes):
        if v != -1:
            axes.text(x + 0.5, y + 0.5, "{:.2f}".format(v), color='k',
                      horizontalalignment='center', verticalalignment='center')

            colour = plt.cm.RdYlGn(v)
            rect = plt.Rectangle((x, y), 1, 1, color=colour, fill=True, alpha=0.5)
            axes.add_artist(rect)

    def draw_board(self, state=None, axes=None, value_state=None):
        if state is None:
            state = self.state
        if axes is None:
            _, axes = plt.subplots(figsize=(10, 8))

        # lines
        axes.plot([0, 3], [1, 1], color='k')
        axes.plot([0, 3], [2, 2], color='k')
        axes.plot([1, 1], [0, 3], color='k')
        axes.plot([2, 2], [0, 3], color='k')

        # pieces & value states
        for i in range(3):
            for j in range(3):
                if state[i, j] > 0:
                    self.draw_piece(i, j, state[i, j], axes)
                if value_state is not None:
                    self.draw_value_state(i, j, value_state[i, j], axes)

        # state outcome
        w = self.check_state(state)
        if w == 1:
            axes.set_title("X wins")
        elif w == 2:
            axes.set_title("O wins")
        elif w == -1:
            axes.set_title("draw")
        else:
            axes.set_title("No win/draw")

    def plot_states(self, states=None, value_states=None):
        if states is None:
            states = [self.state]

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
                self.draw_board(states[n], axes[i, j], value_state=value_states[n])
                n += 1
                if n == len(states):
                    break
            if n == len(states):
                break

        plt.show()

