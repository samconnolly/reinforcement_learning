"""
Multi-armed bandit
"""
import numpy as np
import matplotlib.pyplot as plt


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


class Learn:
    def __init__(self, bandit: MultiArmedBandit):
        self.bandit = bandit
        self.n_arms = bandit.n_arms

    def get_stats(self, success_counts: np.ndarray, pull_counts: np.ndarray, initial_mean: float = 0.5):
        chance_estimates = success_counts / pull_counts
        chance_estimates[np.isnan(chance_estimates)] = initial_mean

        chance_errors = np.sqrt((chance_estimates * (1 - chance_estimates) * 2) / pull_counts)
        chance_errors[pull_counts == 0] = 1
        chance_errors[chance_errors == 0] = 1

        success_rate = np.sum(success_counts) / np.sum(pull_counts)

        return chance_estimates, chance_errors, success_rate

    def epsilon_greedy(self, n: int, eta: float = 0.1, stop_error: bool = False, initial_mean: float = 0.5,
                       optimistic=False, decay=False):
        current_best = 0
        if optimistic:
            pull_counts = np.ones(self.n_arms)
            success_counts = np.ones(self.n_arms)
        else:
            pull_counts = np.zeros(self.n_arms)
            success_counts = np.zeros(self.n_arms)
        diffs = np.array([-1, -1])
        for i in range(n):
            if decay:
                eta = 1 / (i + 1)

            if np.any(diffs > 0) or not stop_error:
                p = np.random.random()
                if p < eta:
                    a = np.random.randint(0, self.n_arms)
                else:
                    a = current_best
            else:
                a = current_best
            outcome = bandit.pull_arm(a)
            pull_counts[a] += 1
            success_counts[a] += int(outcome)

            chance_estimates, chance_errors, success_rate = self.get_stats(success_counts, pull_counts,
                                                                           initial_mean=initial_mean)
            current_best = np.argmax(chance_estimates)

            if stop_error:
                current_best_mask = (np.arange(len(chance_estimates)) == current_best)
                current_lower_lim = chance_estimates[current_best_mask] - chance_errors[current_best_mask]
                current_upper_lims = chance_estimates[~current_best_mask] + chance_errors[~current_best_mask]
                diffs = current_upper_lims - current_lower_lim

        return chance_estimates, chance_errors, success_rate

    def optimistic_init(self, n: int, initial_mean: float = 10):
        return self.epsilon_greedy(n, eta=0, stop_error=False, initial_mean=initial_mean, optimistic=True)

    def ucb1(self, n: int, initial_mean: float = 0.5):
        current_best = 0
        pull_counts = np.zeros(self.n_arms)
        success_counts = np.zeros(self.n_arms)
        for i in range(n):
            outcome = bandit.pull_arm(current_best)
            pull_counts[current_best] += 1
            success_counts[current_best] += int(outcome)

            chance_estimates, chance_errors, success_rate = self.get_stats(success_counts, pull_counts,
                                                                           initial_mean=initial_mean)
            epsilon = np.sqrt((2 * np.log(np.sum(pull_counts))) / pull_counts)
            epsilon[np.isnan(epsilon)] = 1
            upper_confidence_bounds = chance_estimates + epsilon
            current_best = np.argmax(upper_confidence_bounds)

        return chance_estimates, chance_errors, success_rate

    def hammer_time(self, n: int, initial_mean: float = 0.5):
        pull_counts = np.zeros(self.n_arms)
        success_counts = np.zeros(self.n_arms)
        a = 0
        for i in range(n):
            outcome = bandit.pull_arm(a)
            pull_counts[a] += 1
            success_counts[a] += int(outcome)

            a += 1
            if a == self.n_arms:
                a = 0

        chance_estimates, chance_errors, success_rate = self.get_stats(success_counts, pull_counts,
                                                                       initial_mean=initial_mean)
        return chance_estimates, chance_errors, success_rate

    def thompson_sampling(self, n: int, tau: float = 1, m0: float = 0, lambda0: float = 1):
        lambdas = np.ones(3) * lambda0
        pull_counts = np.zeros(self.n_arms)
        success_counts = np.zeros(self.n_arms)
        means = np.ones(self.n_arms) * m0

        for i in range(n):
            # sample means from estimated mean distribution
            samples = np.random.normal(means, lambdas)

            # pick the arm with the best mean, pull it
            a = np.argmax(samples)
            outcome = self.bandit.pull_arm(a)
            pull_counts[a] += 1
            success_counts[a] += int(outcome)

            # update estimated mean distribution
            lambdas += tau
            means = (tau * np.sum(success_counts) + m0 * lambda0) / lambdas

        chance_estimates, chance_errors, success_rate = self.get_stats(success_counts, pull_counts, initial_mean=m0)
        return chance_estimates, chance_errors, success_rate


if __name__ == '__main__':
    n_pulls = 1000

    bandit = MultiArmedBandit(3)
    learner = Learn(bandit)

    print("Hammer")
    h_chance_estimates, h_chance_errors, h_success_rate = learner.hammer_time(n_pulls)

    print("Epsilon Greedy")
    e_chance_estimates, e_chance_errors, e_success_rate = learner.epsilon_greedy(n_pulls, eta=0.1, stop_error=True,
                                                                                 decay=True)

    print("Optimistic Initial Values")
    o_chance_estimates, o_chance_errors, o_success_rate = learner.optimistic_init(n_pulls)

    print("UCB1")
    u_chance_estimates, u_chance_errors, u_success_rate = learner.ucb1(n_pulls)

    print("Thompson sampling")
    t_chance_estimates, t_chance_errors, t_success_rate = learner.thompson_sampling(n_pulls)

    print("True")
    true_chances = bandit.chances
    print(true_chances)

    # create plot
    labels = ['Hammer', 'Epsilon Greedy', 'Optimistic Initial Values', "UCB1", "Thompson", 'Truth']
    colours = ['r', 'b', 'purple', "orange", "pink", 'g']
    n_groups = 3  # 3 arms
    chances = np.array([h_chance_estimates, e_chance_estimates, o_chance_estimates, u_chance_estimates,
                        t_chance_estimates, true_chances])
    errors = np.array([h_chance_errors, e_chance_errors, o_chance_errors, u_chance_errors, t_chance_errors,
                       np.zeros(n_groups)])
    success_rates = [h_success_rate, e_success_rate, o_success_rate, u_success_rate, t_success_rate]
    n_methods = len(chances)

    index = np.arange(n_groups)
    bar_width = 0.8 / n_methods
    opacity = 0.8

    # limit errors to 0/1
    errors = np.tile(errors, (2, 1, 1))
    errors[0][errors[0] > chances] = chances[errors[0] > chances]
    errors[1][errors[1] + chances > 1] = \
        1 - chances[errors[1] + chances > 1]

    # plot
    fig, ax = plt.subplots()
    plt.subplot(1, 2, 1)
    for i in range(n_methods):
        plt.bar(index + bar_width * i, chances[i], bar_width,
                alpha=opacity, color=colours[i], label=labels[i])
        plt.errorbar(index + bar_width * i, chances[i], yerr=errors[:, i],
                     color='k', linestyle="none", capsize=5)

    plt.ylabel('Chance estimates')
    plt.xticks(index + bar_width, ('Arm 1', 'Arm 2', 'Arm 3'))
    plt.legend()

    plt.subplot(1, 2, 2)

    for i in range(n_methods - 1):
        plt.bar(bar_width * i, success_rates[i], bar_width,
                alpha=opacity, color=colours[i], label=labels[i])

    plt.ylabel('Success rates')
    plt.legend()

    plt.tight_layout()
    plt.show()
