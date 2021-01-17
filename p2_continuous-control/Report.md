# Report - Project 2: Continuous Control

[//]: # "Image References"

[image1]: reward_graph.png "Reward Plot"
[image2]: pseudo.png "Pseudo"

### Learning Algorithm

The chosen algorithm for solving this environment is Deep Deterministic Policy Gradients (DDPG).

DDPG can be seen as a kind of actor-critic method, although I think it is more of a DQN variant. In DQN we can't apply the max operation for continuous actions which bounds us to problems with discrete action space only. DDPG solves this issue by using a critic network in order to approximate the maximizer over the Q values of the next state when calculating the target.

In DDPG, we use two deep neural networks, an actor and a ctrici. The actor is used to approximate the optimal action given a state i.e.  argmax a Q(S, a). The critic as explained before, learns to evaluate the optimal action value function and is used when calculating the target.

Lets go over the pseudo code:

In DDPG we first initialize an empty replay memory R and the parameters or weights of the neural networks for the actor and the critic. In order to avoid the problem of moving target, we also use a target network for the actor and the critic and initialize that networks as well.

Then, we sample the environment by performing actions and store away the observed experienced tuples in a replay memory. The actions are selected using the actor network with an added noise. In our case we used the Ornstein-Uhlenbeck process as noise. Afterwards, we select the small batch of N tuples from the memory, randomly. The actor is used to approximate the optimal action given a state i.e.  argmax a Q(S, a), so when used as an input the the critic target network we approximate the maximizer over the Q.
The critic is updated by minimizing the loss between the critic and the target.
The actor is updated by using the policy gradient theorem.

Finally, we update the target networks. (In the implemented algorithm we use soft updates -> x~ = (1-alpha)*x~ + alpha * x)

In my implementation I also decay the noise linearly over the run of the training.

The pseudo code for the algorithm can be found here:

![Pseudo][image2]

Taken from: Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." *arXiv preprint arXiv:1509.02971* (2015).

The chosen hyper-parameters are:

```python
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay
START_EPS = 1.0
END_EPS = 0.1
```

The Actor architecture is:

```python
class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        return F.tanh(self.fc2(x))

```

The actor network receive state as an input, and outputs actions between -1 to 1 in the dimension of action_size. 

The critic architecture:

```python
class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=256, fc3_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
```

The critic network receive state AND action as an input, and outputs a single number which corresponds to the state-action value. 

### Plot of Rewards



![Reward Plot][image1]

As can be seen in the Jupyter notebook the environment was solved in 70 rpisodes.

```
Episode 1	Average Score: 0.54
epsilon: 0.9819820000000375
Episode 2	Average Score: 0.62
epsilon: 0.9639640000000751
Episode 3	Average Score: 0.71
epsilon: 0.9459460000001126
Episode 4	Average Score: 0.78
epsilon: 0.9279280000001502
Episode 5	Average Score: 0.87
epsilon: 0.9099100000001877
Episode 6	Average Score: 0.96
epsilon: 0.8918920000002253
Episode 7	Average Score: 1.00
epsilon: 0.8738740000002628
Episode 8	Average Score: 1.03
epsilon: 0.8558560000003004
Episode 9	Average Score: 1.08
epsilon: 0.8378380000003379
Episode 10	Average Score: 1.21
epsilon: 0.8198200000003755
Episode 11	Average Score: 1.29
epsilon: 0.801802000000413
Episode 12	Average Score: 1.38
epsilon: 0.7837840000004506
Episode 13	Average Score: 1.48
epsilon: 0.7657660000004881
Episode 14	Average Score: 1.63
epsilon: 0.7477480000005257
Episode 15	Average Score: 1.77
epsilon: 0.7297300000005632
Episode 16	Average Score: 1.90
epsilon: 0.7117120000006008
Episode 17	Average Score: 2.03
epsilon: 0.6936940000006383
Episode 18	Average Score: 2.14
epsilon: 0.6756760000006758
Episode 19	Average Score: 2.28
epsilon: 0.6576580000007134
Episode 20	Average Score: 2.38
epsilon: 0.6396400000007509
Episode 21	Average Score: 2.53
epsilon: 0.6216220000007885
Episode 22	Average Score: 2.66
epsilon: 0.603604000000826
Episode 23	Average Score: 2.80
epsilon: 0.5855860000008636
Episode 24	Average Score: 2.93
epsilon: 0.5675680000009011
Episode 25	Average Score: 3.09
epsilon: 0.5495500000009387
Episode 26	Average Score: 3.26
epsilon: 0.5315320000009762
Episode 27	Average Score: 3.42
epsilon: 0.5135140000010138
Episode 28	Average Score: 3.59
epsilon: 0.4954960000010374
Episode 29	Average Score: 3.77
epsilon: 0.47747800000101936
Episode 30	Average Score: 3.96
epsilon: 0.45946000000100135
Episode 31	Average Score: 4.13
epsilon: 0.4414420000009833
Episode 32	Average Score: 4.30
epsilon: 0.4234240000009653
Episode 33	Average Score: 4.52
epsilon: 0.4054060000009473
Episode 34	Average Score: 4.70
epsilon: 0.38738800000092927
Episode 35	Average Score: 4.88
epsilon: 0.36937000000091125
Episode 36	Average Score: 5.05
epsilon: 0.3513520000008932
Episode 37	Average Score: 5.26
epsilon: 0.3333340000008752
Episode 38	Average Score: 5.50
epsilon: 0.3153160000008572
Episode 39	Average Score: 5.73
epsilon: 0.29729800000083917
Episode 40	Average Score: 5.92
epsilon: 0.27928000000082115
Episode 41	Average Score: 6.10
epsilon: 0.26126200000080313
Episode 42	Average Score: 6.30
epsilon: 0.24324400000079555
Episode 43	Average Score: 6.51
epsilon: 0.2252260000008053
Episode 44	Average Score: 6.72
epsilon: 0.20720800000081507
Episode 45	Average Score: 6.93
epsilon: 0.18919000000082484
Episode 46	Average Score: 7.15
epsilon: 0.1711720000008346
Episode 47	Average Score: 7.41
epsilon: 0.15315400000084436
Episode 48	Average Score: 7.62
epsilon: 0.13513600000085413
Episode 49	Average Score: 7.80
epsilon: 0.11711800000085781
Episode 50	Average Score: 7.98
epsilon: 0.1
Episode 51	Average Score: 8.18
epsilon: 0.1
Episode 52	Average Score: 8.41
epsilon: 0.1
Episode 53	Average Score: 8.60
epsilon: 0.1
Episode 54	Average Score: 8.82
epsilon: 0.1
Episode 55	Average Score: 9.03
epsilon: 0.1
Episode 56	Average Score: 9.24
epsilon: 0.1
Episode 57	Average Score: 9.40
epsilon: 0.1
Episode 58	Average Score: 9.58
epsilon: 0.1
Episode 59	Average Score: 9.76
epsilon: 0.1
Episode 60	Average Score: 9.97
epsilon: 0.1
Episode 61	Average Score: 10.17
epsilon: 0.1
Episode 62	Average Score: 10.38
epsilon: 0.1
Episode 63	Average Score: 10.56
epsilon: 0.1
Episode 64	Average Score: 10.72
epsilon: 0.1
Episode 65	Average Score: 10.91
epsilon: 0.1
Episode 66	Average Score: 11.09
epsilon: 0.1
Episode 67	Average Score: 11.28
epsilon: 0.1
Episode 68	Average Score: 11.44
epsilon: 0.1
Episode 69	Average Score: 11.62
epsilon: 0.1
Episode 70	Average Score: 11.77
epsilon: 0.1
Episode 71	Average Score: 11.97
epsilon: 0.1
Episode 72	Average Score: 12.16
epsilon: 0.1
Episode 73	Average Score: 12.33
epsilon: 0.1
Episode 74	Average Score: 12.49
epsilon: 0.1
Episode 75	Average Score: 12.64
epsilon: 0.1
Episode 76	Average Score: 12.79
epsilon: 0.1
Episode 77	Average Score: 12.96
epsilon: 0.1
Episode 78	Average Score: 13.11
epsilon: 0.1
Episode 79	Average Score: 13.28
epsilon: 0.1
Episode 80	Average Score: 13.44
epsilon: 0.1
Episode 81	Average Score: 13.61
epsilon: 0.1
Episode 82	Average Score: 13.77
epsilon: 0.1
Episode 83	Average Score: 13.93
epsilon: 0.1
Episode 84	Average Score: 14.08
epsilon: 0.1
Episode 85	Average Score: 14.22
epsilon: 0.1
Episode 86	Average Score: 14.38
epsilon: 0.1
Episode 87	Average Score: 14.54
epsilon: 0.1
Episode 88	Average Score: 14.68
epsilon: 0.1
Episode 89	Average Score: 14.83
epsilon: 0.1
Episode 90	Average Score: 14.98
epsilon: 0.1
Episode 91	Average Score: 15.12
epsilon: 0.1
Episode 92	Average Score: 15.29
epsilon: 0.1
Episode 93	Average Score: 15.44
epsilon: 0.1
Episode 94	Average Score: 15.62
epsilon: 0.1
Episode 95	Average Score: 15.78
epsilon: 0.1
Episode 96	Average Score: 15.94
epsilon: 0.1
Episode 97	Average Score: 16.08
epsilon: 0.1
Episode 98	Average Score: 16.21
epsilon: 0.1
Episode 99	Average Score: 16.35
epsilon: 0.1
Episode 100	Average Score: 16.48
epsilon: 0.1
Episode 101	Average Score: 16.78
epsilon: 0.1
Episode 102	Average Score: 17.09
epsilon: 0.1
Episode 103	Average Score: 17.39
epsilon: 0.1
Episode 104	Average Score: 17.68
epsilon: 0.1
Episode 105	Average Score: 17.96
epsilon: 0.1
Episode 106	Average Score: 18.24
epsilon: 0.1
Episode 107	Average Score: 18.52
epsilon: 0.1
Episode 108	Average Score: 18.80
epsilon: 0.1
Episode 109	Average Score: 19.07
epsilon: 0.1
Episode 110	Average Score: 19.36
epsilon: 0.1
Episode 111	Average Score: 19.62
epsilon: 0.1
Episode 112	Average Score: 19.87
epsilon: 0.1
Episode 113	Average Score: 20.12
epsilon: 0.1
Episode 114	Average Score: 20.37
epsilon: 0.1
Episode 115	Average Score: 20.62
epsilon: 0.1
Episode 116	Average Score: 20.90
epsilon: 0.1
Episode 117	Average Score: 21.17
epsilon: 0.1
Episode 118	Average Score: 21.43
epsilon: 0.1
Episode 119	Average Score: 21.70
epsilon: 0.1
Episode 120	Average Score: 21.97
epsilon: 0.1
Episode 121	Average Score: 22.22
epsilon: 0.1
Episode 122	Average Score: 22.47
epsilon: 0.1
Episode 123	Average Score: 22.73
epsilon: 0.1
Episode 124	Average Score: 22.96
epsilon: 0.1
Episode 125	Average Score: 23.22
epsilon: 0.1
Episode 126	Average Score: 23.46
epsilon: 0.1
Episode 127	Average Score: 23.71
epsilon: 0.1
Episode 128	Average Score: 23.94
epsilon: 0.1
Episode 129	Average Score: 24.18
epsilon: 0.1
Episode 130	Average Score: 24.40
epsilon: 0.1
Episode 131	Average Score: 24.60
epsilon: 0.1
Episode 132	Average Score: 24.84
epsilon: 0.1
Episode 133	Average Score: 25.04
epsilon: 0.1
Episode 134	Average Score: 25.25
epsilon: 0.1
Episode 135	Average Score: 25.46
epsilon: 0.1
Episode 136	Average Score: 25.67
epsilon: 0.1
Episode 137	Average Score: 25.85
epsilon: 0.1
Episode 138	Average Score: 26.01
epsilon: 0.1
Episode 139	Average Score: 26.18
epsilon: 0.1
Episode 140	Average Score: 26.37
epsilon: 0.1
Episode 141	Average Score: 26.54
epsilon: 0.1
Episode 142	Average Score: 26.69
epsilon: 0.1
Episode 143	Average Score: 26.87
epsilon: 0.1
Episode 144	Average Score: 27.03
epsilon: 0.1
Episode 145	Average Score: 27.18
epsilon: 0.1
Episode 146	Average Score: 27.33
epsilon: 0.1
Episode 147	Average Score: 27.44
epsilon: 0.1
Episode 148	Average Score: 27.58
epsilon: 0.1
Episode 149	Average Score: 27.73
epsilon: 0.1
Episode 150	Average Score: 27.88
epsilon: 0.1
Episode 151	Average Score: 28.02
epsilon: 0.1
Episode 152	Average Score: 28.13
epsilon: 0.1
Episode 153	Average Score: 28.26
epsilon: 0.1
Episode 154	Average Score: 28.37
epsilon: 0.1
Episode 155	Average Score: 28.49
epsilon: 0.1
Episode 156	Average Score: 28.62
epsilon: 0.1
Episode 157	Average Score: 28.77
epsilon: 0.1
Episode 158	Average Score: 28.90
epsilon: 0.1
Episode 159	Average Score: 29.01
epsilon: 0.1
Episode 160	Average Score: 29.05
epsilon: 0.1
Episode 161	Average Score: 29.13
epsilon: 0.1
Episode 162	Average Score: 29.21
epsilon: 0.1
Episode 163	Average Score: 29.33
epsilon: 0.1
Episode 164	Average Score: 29.46
epsilon: 0.1
Episode 165	Average Score: 29.54
epsilon: 0.1
Episode 166	Average Score: 29.65
epsilon: 0.1
Episode 167	Average Score: 29.74
epsilon: 0.1
Episode 168	Average Score: 29.83
epsilon: 0.1
Episode 169	Average Score: 29.94
epsilon: 0.1
Episode 170	Average Score: 30.07
epsilon: 0.1

Environment solved in 70 episodes!	Average Score: 30.07
```

### Ideas for Future Work

Future ideas to improve the agent are to incorporate:

#### Prioritize Experience Replay

In DDPG,the agent interacts with the environment,and a transition (s,a,r,s'), is inserted to an experience replay. To update the parameters of the neural network, a sample of transitions is drawn uniformly at random from the experience replay buffer. Some transitions might be more important to the learning process than others, however,uniform sampling policy treat transitions by the amount of their occurrences. Prioritized experience replay, aims to deal with this issue by assigning each transition a priority, P ,which is proportional to the TD-error. 

#### Delayed Policy Updates

In order to allow for better state-value estimation and have a more stable actor, we can update the policy less frequently compared to the critic.

### Learned Policy

The solution can be found in files checkpoint_actor.pth and checkpoint_critic.pth