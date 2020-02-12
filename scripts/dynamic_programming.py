"""
dynamic programming approach (to grid world)

Sam Connolly 2019
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings
from environments.grid_world import GridWorld, Policy



class Agent:
    def __init__(self, grid_world: GridWorld, policy: Policy, gamma: float = 0.9, binary_probs: bool = True,
                 action_gamma: float = 0.1):
        self.grid_world = grid_world
        self.policy = policy
        self.gamma = gamma
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.value_function = np.zeros_like(self.grid_world.grid_map).astype(float)
        self.action_values = np.zeros_like(self.grid_world.grid_map).astype(float)
        self.binary_probs = binary_probs
        self.action_gamma = action_gamma

    def action(self, state):
        return self.policy.action(state)

    def run_episode(self, plot=True):
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        terminal = False
        self.episode_states.append(self.grid_world.state)
        while not terminal:
            action = self.action(self.grid_world.state)
            reward, terminal = self.grid_world.move(action)
            self.episode_states.append(self.grid_world.state)
            self.episode_actions.append(action)
            self.episode_rewards.append(reward)

        # self.update_value_function()
        # self.grid_world.reset_state()

        if plot:
            self.grid_world.plot_states(self.episode_states)

    def update_value_function(self):
        for state in self.grid_world.all_states:
            old_value = self.value_function[state]
            actions, probs = self.policy.policy[state]
            value = 0
            for action, prob in zip(actions, probs):
                next_states, rewards, _, outcome_probs = self.grid_world.check_move(state, action)
                for next_state, reward, outcome_prob in zip(next_states, rewards, outcome_probs):
                    next_state_value = self.value_function[next_state]
                    value += prob * (reward + self.gamma * next_state_value) * outcome_prob
            self.value_function[state] = value
            delta = np.abs(value - old_value)
        return delta

    def solve_value_function(self, threshold: float = 1e-6, iter_limit: int = 100):
        delta = 0
        i = 0
        while (delta > threshold or i == 0) and i < iter_limit:
            new_delta = self.update_value_function()
            delta = np.max([delta, new_delta])
            i += 1

        if i == iter_limit:
            warnings.warn("Reached max iterations without value function convergence")

    def optimise_policy(self, iter_limit: int = 100):
        policy_changed = True
        i = 0
        while policy_changed and i < iter_limit:
            policy_changed = False
            old_policy = self.policy.policy.copy()
            for state in self.grid_world.all_states:
                actions, action_probs = self.policy.policy[state]
                if len(actions) > 0:
                    rewards = []
                    for action in actions:
                        new_states, new_rewards, _, probs = self.grid_world.check_move(state, action)
                        outcome = 0
                        for new_state, new_reward, prob in zip(new_states, new_rewards, probs):
                            outcome += (new_reward + self.gamma * self.value_function[new_state]) * prob
                        rewards.append(outcome)

                    # prescribed method
                    if self.binary_probs:
                        self.policy.policy[state] = (actions,
                                                     tuple((np.arange(len(rewards)) ==
                                                            np.argmax(rewards)).astype(float)))
                    # adjust probs instead
                    else:
                        new_action_probs = np.array(action_probs)
                        good = (np.arange(len(rewards)) == np.argmax(rewards))
                        new_action_probs[good] += self.action_gamma
                        new_action_probs[~good] -= self.action_gamma / (len(actions) - 1)

                        # check
                        if np.any(new_action_probs < 0) or np.any(new_action_probs > 1):
                            if np.any(new_action_probs < 0):
                                new_action_probs -= np.min(new_action_probs)

                        new_action_probs /= np.sum(new_action_probs)

                        self.policy.policy[state] = (actions, tuple(new_action_probs))
                    self.action_values[state] = np.max(rewards)

            if old_policy != self.policy.policy:
                policy_changed = True

            i += 1

        if i == iter_limit:
            warnings.warn("Reached max iterations without policy function convergence")


if __name__ == '__main__':
    gridworld = GridWorld(windy=True)

    # print(gridworld.grid_map)
    # print(gridworld.rewards)
    # print(gridworld.actions)
    # gridworld.plot_state()

    # even chance of all available options
    uniform_policy = Policy("uniform", gridworld)

    # up if available, otherwise right
    set_policy = Policy("set", gridworld)

    # random
    random_policy = Policy("random", gridworld)

    max_runs = 100
    max_same = 3
    # policy = uniform_policy
    # policy = set_policy
    policy = random_policy
    agent = Agent(gridworld, policy, binary_probs=False)

    states = [agent.grid_world.state]
    value_functions = [agent.value_function.copy()]
    policies = [agent.policy.policy.copy()]

    converged = False
    i = 0
    while not converged and i < max_runs:
        old_policy = agent.policy.policy.copy()

        agent.solve_value_function()
        agent.optimise_policy()

        states.append(agent.grid_world.state)
        value_functions.append(agent.value_function.copy())
        policies.append(agent.policy.policy.copy())

        if np.all(agent.action_values == agent.value_function):
            converged = True

        i += 1

    if i == max_runs:
        warnings.warn("Reached max runs without convergence")
    else:
        print(f"Converged in {i + 1} runs")

    # limit to 9 plots
    if len(states) > 9:
        p = np.append(np.arange(0, len(states), len(states) // 7), len(states) - 1)
    else:
        p = np.arange(len(states))

    gridworld.plot_states(np.array(states)[p],
                          value_states=np.array(value_functions)[p],
                          policies=np.array(policies)[p])
    plt.show()
