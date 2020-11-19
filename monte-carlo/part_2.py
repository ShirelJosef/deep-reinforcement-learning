import sys
import gym
import numpy as np
from collections import defaultdict

from plot_utils import plot_blackjack_values, plot_policy

env = gym.make('Blackjack-v0')


def epsilon_greedy(state, Q, epsilon, nA):
    a_max = np.argmax(Q[state])
    probability = np.zeros(nA)
    probability[a_max] = 1 - epsilon
    probability = probability + epsilon / nA
    return probability

def generate_episode(env, Q, epsilon):
    episode = []
    state = env.reset()
    while True:
        action_probabilities = epsilon_greedy(state, Q,epsilon, env.action_space.n)
        action = np.random.choice(env.action_space.n, 1, p=action_probabilities)[0]
        # print(action)
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

def mc_control(env, num_episodes, alpha, gamma=1.0):
    nA = env.action_space.n
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    # loop over episodes
    epsilon = 1.0
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        ## TODO: complete the function
        episode = generate_episode(env, Q, epsilon)
        visit = defaultdict(lambda: np.zeros(env.action_space.n))
        for i, (state, action, reward) in enumerate(episode):
            if visit[state][action] == 0:
                visit[state][action] = 1
                episode_reward = 0
                for j in range(len(episode) - i):
                    episode_reward = episode_reward + (gamma ** j) * episode[i + j][2]
                Q[state][action] = Q[state][action] + alpha * (episode_reward - Q[state][action])
        epsilon = epsilon / i_episode
        policy = defaultdict(lambda: 0)
        for k, v in Q.items():
            policy[k] = np.argmax(v)
    return policy, Q

# obtain the estimated optimal policy and action-value function
policy, Q = mc_control(env, 500000, 0.02)

# obtain the corresponding state-value function
V = dict((k,np.max(v)) for k, v in Q.items())

# plot the state-value function
plot_blackjack_values(V)

# plot the policy
plot_policy(policy)