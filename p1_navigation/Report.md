# Report - Project 1: Navigation

[//]: # "Image References"

[image1]: banana.gif "Trained Agent"
[image2]: plot.png "Reward Plot"
[image3]: pseudo.png "Pseudo"

### Results

This is an example of the behavior achieved by the trained agent.  

![Trained Agent][image1]

### Learning Algorithm

The chosen algorithm for solving this environment is Deep Q Networks (DQN).

Deep Q Network is a deep learning variant of the classic Q learning algorithm.
We approximate the Q values of each state and derive our policy by taking the action which produce the maximum value for a given state.

In DQN we first initialize an empty replay memory D and the parameters or weights of the neural networks. In order to avoid the problem of moving target, we also use a fixed target network and initialize that network as well.

Then, we sample the environment by performing actions and store away the observed experienced tuples in a replay memory. Afterwards, we select the small batch of tuples from this memory, randomly, and learn from that batch using a gradient descent update step.

Finally, we update the target network every C steps. (In the implemented algorithm we use soft updates -> x~ = (1-alpha)*x~ + alpha * x)

The pseudo code for the algorithm can be found here:

![Pseudo][image3]

Taken from: Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." *nature* 518.7540 (2015): 529-533.

The chosen hyper-parameters are:

```python
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
eps_start=1.0           # Epsilon initial value for epsilon greedy exploration
eps_end=0.01            # Epsilon final value for epsilon greedy exploration
eps_decay=0.995         # Epsilon decay value for epsilon greedy exploration
```

The Q network architecture is:

```python
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.fc1 = nn.Linear(state_size,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

The inputs to the network has`37` dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. The input are inserted into a fully connected layer with 64 neurons with RELU activation, then the output is passed to another 64 neurons layer with RELU activation and then output to the action size which is 4 nodes which approximate the Q value of the corresponding action with the input state.

### Plot of Rewards



![Reward Plot][image2]

As can be seen in the Jupyter notebook the environment was solved in 397 episodes.

```
Episode 100	Average Score: 0.73
Episode 200	Average Score: 3.97
Episode 300	Average Score: 7.38
Episode 400	Average Score: 10.46
Episode 497	Average Score: 13.03
Environment solved in 397 episodes!	Average Score: 13.03
```

### Ideas for Future Work

Future ideas to improve the agent are to incorporate:

#### Double DQN

Q-learning is prone to overestimating the Q values as we always pick the maximum Q values which can be noisy especially in the beginning of the training. The solution is to select the best action using one set of parameters w, but evaluate it using a different set of parameters w'. To mitigate this issue can use the q-network to select the best action and the target network to evaluate the Q value for the target value.

#### Dueling DQN

Dueling DQN learn the value of states and one stream, and the advantage of each action in a different stream.
The intuition is that learning the value of state without the effect of each action is more robust, and we add the advantage stream in order to approximate the Q function which we need in order to derive the policy.