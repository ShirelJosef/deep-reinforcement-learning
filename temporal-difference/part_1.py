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

def q_learning(env, num_episodes, alpha, gamma=1.0):
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(env.nA))
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
        while True:
            probabilities = epsilon_greedy(state, Q, epsilon, env.action_space.n)
            action = get_action(probabilities)

            next_state, reward, done, info = env.step(action)
            Qmax = np.max(Q[next_state])
            Q[state][action] = Q[state][action] + alpha * (
                        reward + gamma * Qmax - Q[state][action])
            state = next_state
            if done:
                break

    return Q

# obtain the estimated optimal policy and corresponding action-value function
Q_sarsamax = q_learning(env, 5000, .01)

# print the estimated optimal policy
policy_sarsamax = np.array([np.argmax(Q_sarsamax[key]) if key in Q_sarsamax else -1 for key in np.arange(48)]).reshape((4,12))
check_test.run_check('td_control_check', policy_sarsamax)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsamax)

# plot the estimated optimal state-value function
plot_values([np.max(Q_sarsamax[key]) if key in Q_sarsamax else 0 for key in np.arange(48)])
