"""
Tic-tac-toe agent

Sam Connolly 2019
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import os
from environments.tic_tac_toe import TicTacToe


class Agent:
    def __init__(self, ttt: TicTacToe, player: str = "x", learning_rate: float = 0.1, eta0: float = 1.0,
                 random_moves: bool = False, explore: bool = True):
        self.player = player
        self.ttt = ttt
        self.seen_states = np.array([[[0., 0., 0.],
                                      [0., 0., 0.],
                                      [0., 0., 0.]]])
        self.values = np.array([0.5])
        self.last_run = np.array([])
        self.alpha = learning_rate
        self.eta0 = eta0
        self.runs = 1
        self.random_moves = random_moves
        self.last_state_values = np.ones((3, 3)) * -1
        self.explore = explore

    @staticmethod
    def hash_state(state: np.array):
        h = np.sum(np.power(10, np.arange(9, 0, -1)) * state.flatten())
        return h

    def record_state(self, state):

        index = np.argwhere(((self.seen_states == state).all(axis=1)).all(axis=1))

        # if state is unseen, initialise state value
        if index.shape[0] == 0:
            outcome = self.ttt.check_state(state)
            self.seen_states = np.vstack((self.seen_states, state.reshape(1, 3, 3)))

            if outcome == 0:
                self.values = np.append(self.values, 0.5)
            elif outcome == self.ttt.piece_dict[self.player]:
                self.values = np.append(self.values, 1.0)
            else:
                self.values = np.append(self.values, 0.0)

            index = -1

        else:
            index = index[0][0]

        return index

    def action(self):
        eta = 0.1  # self.eta0 / self.runs
        choice_states = []
        coord_sets = []
        values = []

        # current state
        current_state = self.ttt.state.copy()

        _ = self.record_state(current_state)

        # next state
        self.last_state_values = np.ones((3, 3)) * -1
        for i in range(3):
            for j in range(3):
                if self.ttt.state[i, j] == 0:
                    test_state = self.ttt.state.copy()
                    test_state[i, j] = self.ttt.piece_dict[self.player]

                    index = self.record_state(test_state)

                    choice_states.append(test_state)
                    coord_sets.append([i, j])
                    values.append(self.values[index])
                    self.last_state_values[i, j] = self.values[index]

        if len(choice_states) == 0:
            print("whoops")

        # explore randomly
        if np.random.random() < eta and self.explore:
            r = np.random.randint(0, len(choice_states))
            best_state = choice_states[r]
            a = coord_sets[r]
        # otherwise set best state
        else:
            v_max = np.max(values)
            indices = np.where(values == v_max)[0]
            if len(indices) == 1 and not self.random_moves:
                a = coord_sets[indices[0]]
                best_state = choice_states[indices[0]]
            else:
                # pick randomly if both states are equally good
                r = np.random.randint(0, len(indices))
                best_state = choice_states[indices[r]]
                a = coord_sets[indices[r]]

        # record new state chosen
        if self.last_run.shape[0] == 0:
            self.last_run = best_state.reshape(1, 3, 3)
        else:
            self.last_run = np.vstack((self.last_run, best_state.reshape(1, 3, 3)))

        # make best move
        return self.ttt.place(*a, self.player)

    def update_value_function(self, states: List):
        # run backwards through states
        v_next = 1.0
        for n, state in enumerate(states[::-1]):
            index = np.argwhere(((self.seen_states == state).all(axis=1)).all(axis=1))[0, 0]
            self.values[index] = self.values[index] + self.alpha * (v_next - self.values[index])
            v_next = self.values[index]

    def save_value_function(self, filename_root):
        np.save(os.path.join("../models", filename_root + "_states.npy"), self.seen_states)
        np.save(os.path.join("../models", filename_root + "_values.npy"), self.values)

    def load_value_function(self, filename_root):
        self.seen_states = np.load(os.path.join("../models", filename_root + "_states.npy"))
        self.values = np.load(os.path.join("../models", filename_root + "_values.npy"))


def run_training_episode(agent1: Agent, agent2: Agent):
    agents = [agent1, agent2]
    outcome = 0
    runs = [agent1.ttt.state.reshape(3, 3).copy()]  # record initial state
    up = np.random.randint(0, 2)
    while outcome == 0:
        outcome = agents[up].action()
        runs.append(agent1.ttt.state.reshape(3, 3).copy())
        up = 1 - up

    agents[up].record_state(agent1.ttt.state)  # show last agent the final state

    agent1.update_value_function(runs)
    agent2.update_value_function(runs)
    agent1.ttt.clear_board()


def run_training_episodes(agent1: Agent, agent2: Agent, n_runs: int, verbose: bool = True):
    for n in range(n_runs):
        run_training_episode(agent1, agent2)
        if verbose and n % int(n_runs // 10) == 0:
            print((n // int(n_runs // 10)) * 10, "%")


def play_match(agent1: Agent, agent2: Agent, first: int = None):
    agents = [agent1, agent2]
    if first is None:
        index = np.random.randint(0, 2)
    else:
        index = first
    first_agent = agents[index]
    second_agent = agents[1 - index]
    states = [first_agent.ttt.state.copy()]
    value_states = []
    while True:
        outcome = first_agent.action()
        states.append(first_agent.ttt.state.copy())
        value_states.append(first_agent.last_state_values)
        if outcome != 0:
            break

        outcome = second_agent.action()
        states.append(second_agent.ttt.state.copy())
        value_states.append(second_agent.last_state_values)
        if outcome != 0:
            break

    value_states.append(None)
    second_agent.ttt.clear_board()

    return outcome, states, value_states


def plot_match(agent1: Agent, agent2: Agent, first: int = None):
    _, states, value_states = play_match(agent1, agent2, first)
    agent1.ttt.plot_states(states, value_states=value_states)


class PlayerMatch:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.outcome = 0
        self.ax = None
        self.fig = None
        self.cid = None
        self.agent = agent

    def step(self, x: int, y: int):
        if self.agent.ttt.state[x, y] == 0:
            self.outcome = self.agent.ttt.place(x, y, "o")
            if self.outcome == 0:
                self.outcome = self.agent.action()
            self.agent.ttt.draw_board(axes=self.ax)
            if self.outcome != 0:
                self.fig.canvas.mpl_disconnect(self.cid)
                self.agent.ttt.clear_board()
                print("game end")
        else:
            print("Position occupied!")

    def onclick(self, event):
        x, y = int(event.xdata), int(event.ydata)
        print('x=%d, y=%d' % (x, y))
        self.step(x, y)
        plt.show()

    def play(self):
        self.fig, self.ax = plt.subplots()
        self.agent.ttt.draw_board(axes=self.ax)

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

    def play_series(self):
        while True:
            self.play()


def play_agent_matches(agent1: Agent, agent2: Agent, n_games: int):
    outcomes = np.zeros(4)
    for n in range(n_games):
        outcomes[play_match(agent1, agent2)[0]] += 1

    return outcomes


if __name__ == '__main__':
    n_runs = 10000
    train = False
    play = True

    ttt = TicTacToe()
    x_agent = Agent(ttt, player='x')
    if train:
        o_agent = Agent(ttt, player='o')

        # train
        print("Training...")
        run_training_episodes(x_agent, o_agent, n_runs)
        # ttt.plot_states(agent.last_run)

        x_agent.save_value_function(str(n_runs))

        # play against untrained agent
        x_agent.player = "x"
        x_agent.explore = False
        untrained_agent = Agent(ttt, player="o", random_moves=True, explore=False)
        random_agent = Agent(ttt, player="o", random_moves=False, explore=False)

        print("Test matches...")
        outcomes = play_agent_matches(x_agent, random_agent, 100)
        print(outcomes / np.sum(outcomes))

        plot_match(x_agent, random_agent, first=0)
    else:
        x_agent.load_value_function(str(n_runs))

    if play:
        match = PlayerMatch(x_agent)
        match.play_series()
