import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
#%matplotlib inline

import check_test
from plot_utils import plot_values

env = gym.make('CliffWalking-v0')


def epsilon_greedy(state, Q, epsilon, nA):
    a_max = np.argmax(Q[state])
    probabilities = np.zeros(nA)
    probabilities[a_max] = 1 - epsilon
    probabilities = probabilities + epsilon / nA
    return probabilities

def get_action(probabilities):
    action = np.random.choice(env.action_space.n, 1, p=probabilities)[0]
    return action


def sarsa(env, num_episodes, alpha, gamma=1.0):
    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))
    # initialize performance monitor
    # loop over episodes
    epsilon_init = 1.0
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        epsilon = epsilon_init / i_episode
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        ## TODO: complete the function
        state = env.reset()
        probabilities = epsilon_greedy(state, Q, epsilon, env.action_space.n)
        action = get_action(probabilities)
        while True:

            next_state, reward, done, info = env.step(action)
            probabilities = epsilon_greedy(next_state, Q, epsilon, env.action_space.n)
            next_action = get_action(probabilities)
            Q[state][action] = Q[state][action] + alpha * (
                        reward + gamma * Q[next_state][next_action] - Q[state][action])
            state = next_state
            action = next_action
            if done:
                break

    return Q

# obtain the estimated optimal policy and corresponding action-value function
Q_sarsa = sarsa(env, 5000, .01)

# print the estimated optimal policy
policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)
check_test.run_check('td_control_check', policy_sarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsa)

# plot the estimated optimal state-value function
V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
plot_values(V_sarsa)