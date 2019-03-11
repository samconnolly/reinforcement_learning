"""
Tic-tac-toe agent

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
                if np.all(test_state[i] == d )or \
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

    def draw_board(self, state=None, axes=None):
        if state is None:
            state = self.state
        if axes is None:
            _, axes = plt.subplots(figsize=(10, 8))

        # lines
        axes.plot([0, 3], [1, 1], color='k')
        axes.plot([0, 3], [2, 2], color='k')
        axes.plot([1, 1], [0, 3], color='k')
        axes.plot([2, 2], [0, 3], color='k')

        # pieces
        for i in range(3):
            for j in range(3):
                if state[i, j] > 0:
                    self.draw_piece(i, j, state[i, j], axes)

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

    def plot_states(self, states=None):
        if states is None:
            states = [self.state]

        ns = len(states)
        nx = int(np.sqrt(ns))
        ny = ns // nx
        if ns % nx != 0:
            ny += 1

        _, axes = plt.subplots(nx, ny, figsize=(10, 8))
        n = 0
        for i in range(nx):
            for j in range(ny):
                ttt.draw_board(states[n], axes[i, j])
                n += 1

        plt.show()


class Agent:
    def __init__(self, ttt: TicTacToe, player="x"):
        self.player = player
        self.ttt = ttt
        self.seen_states = np.array([])
        self.values = np.array([])
        self.last_run = np.array([])

    def action(self):
        v_max = 0
        a = [-1, -1]
        best_state = None
        for i in range(3):
            for j in range(3):
                if self.ttt.state[i, j] == 0:
                    test_state = self.ttt.state.copy()
                    test_state[i, j] = self.ttt.piece_dict[self.player]

                    if self.seen_states.shape[0] > 0:
                        index = np.argwhere(((self.seen_states == test_state).all(axis=1)).all(axis=1))
                    else:
                        index = np.array([])

                    # if state is seen, initialise state value
                    if index.shape[0] == 0:
                        outcome = self.ttt.check_state(test_state)
                        if self.seen_states.shape[0] == 0:
                            self.seen_states = test_state.reshape(1, 3, 3)
                        else:
                            self.seen_states = np.vstack((self.seen_states, test_state.reshape(1, 3, 3)))

                        if outcome == 0:
                            self.values = np.append(self.values, 0.5)
                        elif outcome == self.ttt.piece_dict[self.player]:
                            self.values = np.append(self.values, 1.0)
                        else:
                            self.values = np.append(self.values, 0.0)

                        index = -1
                    else:
                        index = index[0][0]

                    v = self.values[index]
                    if v > v_max:
                        v_max = v
                        a = [i, j]
                        best_state = test_state

        # record new state chosen
        self.last_run = np.vstack((self.last_run, best_state.reshape(1, 3, 3)))

        # make best move
        return self.ttt.place(*a, self.player)

    def run_episode(self):
        outcome = 0
        self.last_run = self.ttt.state.reshape(1, 3, 3)  # record initial state
        while outcome == 0:
            outcome = agent.action()
            if agent.player == "x":
                agent.player = "o"
            else:
                agent.player = "x"

        self.ttt.clear_board()


if __name__ == '__main__':
    ttt = TicTacToe()
    agent = Agent(ttt)

    agent.run_episode()
    ttt.plot_states(agent.last_run)
