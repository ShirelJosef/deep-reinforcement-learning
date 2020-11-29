import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon_init = 1.0
        self.epsilon = self.epsilon_init
        self.alpha = .15
        self.gamma = 1
        self.episode_counter = 1

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        probabilities = self.epsilon_greedy(state, self.Q, self.epsilon, self.nA)
        action = self.get_action(probabilities)
        return action

    def get_action(self, probabilities):
        action = np.random.choice(self.nA, 1, p=probabilities)[0]
        return action

    def epsilon_greedy(self, state, Q, epsilon, nA):
        a_max = np.argmax(Q[state])
        probabilities = np.zeros(nA)
        probabilities[a_max] = 1 - epsilon
        probabilities = probabilities + epsilon / nA
        return probabilities

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        Qmax = np.max(self.Q[next_state])
        self.Q[state][action] = self.Q[state][action] + self.alpha * (
                reward + self.gamma * Qmax - self.Q[state][action])
        if done == True:
            self.episode_counter = self.episode_counter + 1
        self.epsilon = self.epsilon_init / self.episode_counter