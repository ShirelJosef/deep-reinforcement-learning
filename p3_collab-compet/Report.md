# Report - Project 2: Continuous Control

[//]: # "Image References"

[image1]: rewards.png "Reward Plot"
[image2]: pseudo.png "Pseudo"
[]: 
[image3]: ddpg_pseudo.png "Pseudo"

### Learning Algorithm

The chosen algorithm for solving this environment is Multi Agent Deep Deterministic Policy Gradients (MADDPG).

#### DDPG:

First, I will explain about DDPG:

DDPG can be seen as a kind of actor-critic method, although I think it is more of a DQN variant. In DQN we can't apply the max operation for continuous actions which bounds us to problems with discrete action space only. DDPG solves this issue by using a critic network in order to approximate the maximizer over the Q values of the next state when calculating the target.

In DDPG, we use two deep neural networks, an actor and a critic. The actor is used to approximate the optimal action given a state i.e.  argmax a Q(S, a). The critic as explained before, learns to evaluate the optimal action value function and is used when calculating the target.

Lets go over the pseudo code:

In DDPG we first initialize an empty replay memory R and the parameters or weights of the neural networks for the actor and the critic. In order to avoid the problem of moving target, we also use a target network for the actor and the critic and initialize that networks as well.

Then, we sample the environment by performing actions and store away the observed experienced tuples in a replay memory. The actions are selected using the actor network with an added noise. In our case we used the Ornstein-Uhlenbeck process as noise. Afterwards, we select the small batch of N tuples from the memory, randomly. The actor is used to approximate the optimal action given a state i.e.  argmax a Q(S, a), so when used as an input the the critic target network we approximate the maximizer over the Q.
The critic is updated by minimizing the loss between the critic and the target.
The actor is updated by using the policy gradient theorem.

Finally, we update the target networks. (In the implemented algorithm we use soft updates -> x~ = (1-alpha)*x~ + alpha * x)

In my implementation I also decay the noise linearly over the run of the training.

The pseudo code for the algorithm can be found here:

![Pseudo][image3]

Taken from: Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." *arXiv preprint arXiv:1509.02971* (2015).

#### MADDPG:

In MADDPG each agent’s critic is trained using the observations and actions from all the agents, whereas each agent’s actor is trained using just its own observations. This allows the agents to be effectively trained without requiring other agents’ observations during inference (because the actor is only dependent on its own observations). 

Using centralized learning and decentralized execution in multiagent environments, allowing agents to learn to collaborate and compete with each other.

The pseudo code for the algorithm can be found here:

![Pseudo][image2]

Taken from: Lowe, Ryan, et al. "Multi-agent actor-critic for mixed cooperative-competitive environments." *Advances in neural information processing systems*. 2017.

The chosen hyper-parameters are:

```python
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-2  # for soft update of target parameters
#GAMMA = 0.95  # discount factor
#TAU = 1e-2  # for soft update of target parameters
LR_ACTOR = 1e-3  # learning rate of the actor
LR_CRITIC = 3e-3  # learning rate of the critic
WEIGHT_DECAY = 0.0001  # L2 weight decay
START_EPS = 5.0
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
        xs = F.leaky_relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)
```

to initilaize the networks I use:

```python
# Actor Network (w/ Target Network)
self.actor_local = Actor(state_size, action_size, random_seed).to(device)
self.actor_target = Actor(state_size, action_size, random_seed).to(device)

# Critic Network (w/ Target Network)
self.critic_local = Critic(state_size * number_of_agents, action_size * number_of_agents, random_seed).to(device)
self.critic_target = Critic(state_size * number_of_agents, action_size * number_of_agents, random_seed).to(device)
```

The actor network receive its own local state as an input, and outputs actions between -1 to 1 in the dimension of action_size. 

The critic network receive state AND action from ALL agents as an input, and outputs a single number which corresponds to the state-action value. 

### Plot of Rewards



![Reward Plot][image1]

As can be seen in the Jupyter notebook the environment was solved in 5000 episodes.

```
Episode 1	Average Score: 0.00
epsilon: 4.999265000000004
Episode 2	Average Score: 0.00
epsilon: 4.998579000000007
Episode 3	Average Score: 0.00
epsilon: 4.997893000000011
Episode 4	Average Score: 0.00
epsilon: 4.997207000000015
Episode 5	Average Score: 0.00
epsilon: 4.996521000000018
Episode 6	Average Score: 0.00
epsilon: 4.995786000000022
Episode 7	Average Score: 0.00
epsilon: 4.995100000000026
Episode 8	Average Score: 0.00
epsilon: 4.994414000000029
Episode 9	Average Score: 0.00
epsilon: 4.993728000000033
Episode 10	Average Score: 0.00
epsilon: 4.993042000000036
Episode 11	Average Score: 0.00
epsilon: 4.99230700000004
Episode 12	Average Score: 0.00
epsilon: 4.991621000000044
Episode 13	Average Score: 0.01
epsilon: 4.990102000000052
Episode 14	Average Score: 0.01
epsilon: 4.989416000000055
Episode 15	Average Score: 0.01
epsilon: 4.988730000000059
Episode 16	Average Score: 0.01
epsilon: 4.987995000000063
Episode 17	Average Score: 0.01
epsilon: 4.987309000000066
Episode 18	Average Score: 0.01
epsilon: 4.98662300000007
Episode 19	Average Score: 0.01
epsilon: 4.985937000000074
Episode 20	Average Score: 0.01
epsilon: 4.985251000000077
Episode 21	Average Score: 0.00
epsilon: 4.984565000000081
Episode 22	Average Score: 0.00
epsilon: 4.983830000000085
Episode 23	Average Score: 0.01
epsilon: 4.982360000000092
Episode 24	Average Score: 0.01
epsilon: 4.9809390000001
Episode 25	Average Score: 0.01
epsilon: 4.9802040000001035
Episode 26	Average Score: 0.01
epsilon: 4.979518000000107
Episode 27	Average Score: 0.01
epsilon: 4.978832000000111
Episode 28	Average Score: 0.01
epsilon: 4.977362000000118
Episode 29	Average Score: 0.01
epsilon: 4.976676000000122
Episode 30	Average Score: 0.01
epsilon: 4.975990000000126
Episode 31	Average Score: 0.01
epsilon: 4.975304000000129
Episode 32	Average Score: 0.01
epsilon: 4.974618000000133
Episode 33	Average Score: 0.01
epsilon: 4.973932000000136
Episode 34	Average Score: 0.01
epsilon: 4.97324600000014
Episode 35	Average Score: 0.01
epsilon: 4.972511000000144
Episode 36	Average Score: 0.01
epsilon: 4.971825000000147
Episode 37	Average Score: 0.01
epsilon: 4.971090000000151
Episode 38	Average Score: 0.01
epsilon: 4.970404000000155
Episode 39	Average Score: 0.01
epsilon: 4.969718000000158
Episode 40	Average Score: 0.01
epsilon: 4.969032000000162
Episode 41	Average Score: 0.01
epsilon: 4.968297000000166
Episode 42	Average Score: 0.01
epsilon: 4.966680000000174
Episode 43	Average Score: 0.01
epsilon: 4.965994000000178
Episode 44	Average Score: 0.01
epsilon: 4.9653080000001815
Episode 45	Average Score: 0.01
epsilon: 4.964622000000185
Episode 46	Average Score: 0.01
epsilon: 4.963936000000189
Episode 47	Average Score: 0.01
epsilon: 4.9632010000001925
Episode 48	Average Score: 0.01
epsilon: 4.9617310000002
Episode 49	Average Score: 0.01
epsilon: 4.961045000000204
Episode 50	Average Score: 0.01
epsilon: 4.960359000000207
Episode 51	Average Score: 0.01
epsilon: 4.959673000000211
Episode 52	Average Score: 0.01
epsilon: 4.9589870000002145
Episode 53	Average Score: 0.01
epsilon: 4.958252000000218
Episode 54	Average Score: 0.01
epsilon: 4.957566000000222
Episode 55	Average Score: 0.01
epsilon: 4.955851000000231
Episode 56	Average Score: 0.01
epsilon: 4.955116000000235
Episode 57	Average Score: 0.01
epsilon: 4.954430000000238
Episode 58	Average Score: 0.01
epsilon: 4.953744000000242
Episode 59	Average Score: 0.01
epsilon: 4.9530580000002455
Episode 60	Average Score: 0.01
epsilon: 4.95227400000025
Episode 61	Average Score: 0.01
epsilon: 4.951588000000253
Episode 62	Average Score: 0.01
epsilon: 4.950902000000257
Episode 63	Average Score: 0.01
epsilon: 4.950167000000261
Episode 64	Average Score: 0.01
epsilon: 4.949481000000264
Episode 65	Average Score: 0.01
epsilon: 4.948795000000268
Episode 66	Average Score: 0.01
epsilon: 4.948060000000272
Episode 67	Average Score: 0.01
epsilon: 4.947374000000275
Episode 68	Average Score: 0.01
epsilon: 4.946688000000279
Episode 69	Average Score: 0.01
epsilon: 4.9460020000002825
Episode 70	Average Score: 0.01
epsilon: 4.945316000000286
Episode 71	Average Score: 0.01
epsilon: 4.94463000000029
Episode 72	Average Score: 0.01
epsilon: 4.9438950000002935
Episode 73	Average Score: 0.01
epsilon: 4.942376000000301
Episode 74	Average Score: 0.01
epsilon: 4.941690000000305
Episode 75	Average Score: 0.01
epsilon: 4.941004000000309
Episode 76	Average Score: 0.01
epsilon: 4.940318000000312
Episode 77	Average Score: 0.01
epsilon: 4.939632000000316
Episode 78	Average Score: 0.01
epsilon: 4.93889700000032
Episode 79	Average Score: 0.01
epsilon: 4.938211000000323
Episode 80	Average Score: 0.01
epsilon: 4.937476000000327
Episode 81	Average Score: 0.01
epsilon: 4.936790000000331
Episode 82	Average Score: 0.01
epsilon: 4.936104000000334
Episode 83	Average Score: 0.01
epsilon: 4.935418000000338
Episode 84	Average Score: 0.01
epsilon: 4.934683000000342
Episode 85	Average Score: 0.01
epsilon: 4.933997000000345
Episode 86	Average Score: 0.01
epsilon: 4.933311000000349
Episode 87	Average Score: 0.01
epsilon: 4.932625000000352
Episode 88	Average Score: 0.01
epsilon: 4.931939000000356
Episode 89	Average Score: 0.01
epsilon: 4.93125300000036
Episode 90	Average Score: 0.01
epsilon: 4.930567000000363
Episode 91	Average Score: 0.01
epsilon: 4.929832000000367
Episode 92	Average Score: 0.01
epsilon: 4.929146000000371
Episode 93	Average Score: 0.01
epsilon: 4.928460000000374
Episode 94	Average Score: 0.01
epsilon: 4.927774000000378
Episode 95	Average Score: 0.01
epsilon: 4.927088000000381
Episode 96	Average Score: 0.01
epsilon: 4.926402000000385
Episode 97	Average Score: 0.01
epsilon: 4.925716000000389
Episode 98	Average Score: 0.01
epsilon: 4.925030000000392
Episode 99	Average Score: 0.01
epsilon: 4.924344000000396
Episode 100	Average Score: 0.01
epsilon: 4.923658000000399
Episode 101	Average Score: 0.01
epsilon: 4.922972000000403
Episode 102	Average Score: 0.01
epsilon: 4.9222860000004065
Episode 103	Average Score: 0.01
epsilon: 4.92160000000041
Episode 104	Average Score: 0.01
epsilon: 4.920914000000414
Episode 105	Average Score: 0.01
epsilon: 4.920228000000417
Episode 106	Average Score: 0.01
epsilon: 4.919493000000421
Episode 107	Average Score: 0.01
epsilon: 4.918709000000425
Episode 108	Average Score: 0.01
epsilon: 4.917974000000429
Episode 109	Average Score: 0.01
epsilon: 4.917288000000433
Episode 110	Average Score: 0.01
epsilon: 4.916602000000436
Episode 111	Average Score: 0.01
epsilon: 4.91591600000044
Episode 112	Average Score: 0.01
epsilon: 4.915181000000444
Episode 113	Average Score: 0.01
epsilon: 4.914495000000447
Episode 114	Average Score: 0.01
epsilon: 4.913809000000451
Episode 115	Average Score: 0.01
epsilon: 4.913123000000454
Episode 116	Average Score: 0.01
epsilon: 4.911653000000462
Episode 117	Average Score: 0.01
epsilon: 4.910967000000466
Episode 118	Average Score: 0.01
epsilon: 4.910281000000469
Episode 119	Average Score: 0.01
epsilon: 4.908860000000477
Episode 120	Average Score: 0.01
epsilon: 4.90817400000048
Episode 121	Average Score: 0.01
epsilon: 4.907488000000484
Episode 122	Average Score: 0.01
epsilon: 4.9068020000004875
Episode 123	Average Score: 0.01
epsilon: 4.906067000000491
Episode 124	Average Score: 0.01
epsilon: 4.905381000000495
Episode 125	Average Score: 0.01
epsilon: 4.9046950000004985
Episode 126	Average Score: 0.01
epsilon: 4.903911000000503
Episode 127	Average Score: 0.01
epsilon: 4.903225000000506
Episode 128	Average Score: 0.01
epsilon: 4.90253900000051
Episode 129	Average Score: 0.01
epsilon: 4.901853000000513
Episode 130	Average Score: 0.01
epsilon: 4.901118000000517
Episode 131	Average Score: 0.01
epsilon: 4.900432000000521
Episode 132	Average Score: 0.01
epsilon: 4.899746000000524
Episode 133	Average Score: 0.01
epsilon: 4.899060000000528
Episode 134	Average Score: 0.01
epsilon: 4.8974430000005365
Episode 135	Average Score: 0.01
epsilon: 4.89675700000054
Episode 136	Average Score: 0.01
epsilon: 4.896071000000544
Episode 137	Average Score: 0.01
epsilon: 4.894552000000552
Episode 138	Average Score: 0.01
epsilon: 4.893866000000555
Episode 139	Average Score: 0.01
epsilon: 4.893180000000559
Episode 140	Average Score: 0.01
epsilon: 4.892445000000563
Episode 141	Average Score: 0.01
epsilon: 4.891808000000566
Episode 142	Average Score: 0.01
epsilon: 4.8911220000005695
Episode 143	Average Score: 0.01
epsilon: 4.890387000000573
Episode 144	Average Score: 0.01
epsilon: 4.889701000000577
Episode 145	Average Score: 0.01
epsilon: 4.888084000000585
Episode 146	Average Score: 0.01
epsilon: 4.887447000000589
Episode 147	Average Score: 0.01
epsilon: 4.886712000000593
Episode 148	Average Score: 0.01
epsilon: 4.886026000000596
Episode 149	Average Score: 0.01
epsilon: 4.8853400000006
Episode 150	Average Score: 0.01
epsilon: 4.884654000000603
Episode 151	Average Score: 0.01
epsilon: 4.883968000000607
Episode 152	Average Score: 0.01
epsilon: 4.882400000000615
Episode 153	Average Score: 0.01
epsilon: 4.881714000000619
Episode 154	Average Score: 0.01
epsilon: 4.881028000000622
Episode 155	Average Score: 0.01
epsilon: 4.880293000000626
Episode 156	Average Score: 0.01
epsilon: 4.87960700000063
Episode 157	Average Score: 0.01
epsilon: 4.878921000000633
Episode 158	Average Score: 0.01
epsilon: 4.878235000000637
Episode 159	Average Score: 0.01
epsilon: 4.8775490000006405
Episode 160	Average Score: 0.01
epsilon: 4.876863000000644
Episode 161	Average Score: 0.01
epsilon: 4.876177000000648
Episode 162	Average Score: 0.01
epsilon: 4.875491000000651
Episode 163	Average Score: 0.01
epsilon: 4.874805000000655
Episode 164	Average Score: 0.01
epsilon: 4.874070000000659
Episode 165	Average Score: 0.01
epsilon: 4.873384000000662
Episode 166	Average Score: 0.01
epsilon: 4.872698000000666
Episode 167	Average Score: 0.01
epsilon: 4.8720120000006695
Episode 168	Average Score: 0.01
epsilon: 4.871326000000673
Episode 169	Average Score: 0.01
epsilon: 4.869856000000681
Episode 170	Average Score: 0.01
epsilon: 4.869170000000684
Episode 171	Average Score: 0.01
epsilon: 4.867651000000692
Episode 172	Average Score: 0.01
epsilon: 4.867014000000696
Episode 173	Average Score: 0.01
epsilon: 4.8662790000006995
Episode 174	Average Score: 0.01
epsilon: 4.865593000000703
Episode 175	Average Score: 0.01
epsilon: 4.864907000000707
Episode 176	Average Score: 0.01
epsilon: 4.86422100000071
Episode 177	Average Score: 0.01
epsilon: 4.863535000000714
Episode 178	Average Score: 0.01
epsilon: 4.8620650000007215
Episode 179	Average Score: 0.01
epsilon: 4.861379000000725
Episode 180	Average Score: 0.01
epsilon: 4.860693000000729
Episode 181	Average Score: 0.01
epsilon: 4.8599580000007325
Episode 182	Average Score: 0.01
epsilon: 4.859272000000736
Episode 183	Average Score: 0.01
epsilon: 4.85858600000074
Episode 184	Average Score: 0.01
epsilon: 4.857900000000743
Episode 185	Average Score: 0.01
epsilon: 4.857214000000747
Episode 186	Average Score: 0.01
epsilon: 4.8565280000007505
Episode 187	Average Score: 0.01
epsilon: 4.853539000000766
Episode 188	Average Score: 0.01
epsilon: 4.85285300000077
Episode 189	Average Score: 0.01
epsilon: 4.8521180000007735
Episode 190	Average Score: 0.01
epsilon: 4.851432000000777
Episode 191	Average Score: 0.01
epsilon: 4.850746000000781
Episode 192	Average Score: 0.01
epsilon: 4.850060000000784
Episode 193	Average Score: 0.01
epsilon: 4.8484920000007925
Episode 194	Average Score: 0.01
epsilon: 4.847806000000796
Episode 195	Average Score: 0.01
epsilon: 4.8471200000008
Episode 196	Average Score: 0.01
epsilon: 4.846434000000803
Episode 197	Average Score: 0.01
epsilon: 4.845699000000807
Episode 198	Average Score: 0.01
epsilon: 4.845013000000811
Episode 199	Average Score: 0.01
epsilon: 4.844327000000814
Episode 200	Average Score: 0.01
epsilon: 4.843641000000818
Episode 201	Average Score: 0.01
epsilon: 4.8429550000008215
Episode 202	Average Score: 0.01
epsilon: 4.842220000000825
Episode 203	Average Score: 0.01
epsilon: 4.841534000000829
Episode 204	Average Score: 0.01
epsilon: 4.8408480000008325
Episode 205	Average Score: 0.01
epsilon: 4.840162000000836
Episode 206	Average Score: 0.01
epsilon: 4.83947600000084
Episode 207	Average Score: 0.01
epsilon: 4.837320000000851
Episode 208	Average Score: 0.01
epsilon: 4.836585000000855
Episode 209	Average Score: 0.01
epsilon: 4.835948000000858
Episode 210	Average Score: 0.01
epsilon: 4.835213000000862
Episode 211	Average Score: 0.01
epsilon: 4.8345270000008655
Episode 212	Average Score: 0.01
epsilon: 4.833890000000869
Episode 213	Average Score: 0.01
epsilon: 4.833155000000873
Episode 214	Average Score: 0.01
epsilon: 4.832469000000876
Episode 215	Average Score: 0.01
epsilon: 4.83178300000088
Episode 216	Average Score: 0.01
epsilon: 4.8310970000008835
Episode 217	Average Score: 0.01
epsilon: 4.829578000000891
Episode 218	Average Score: 0.01
epsilon: 4.8268830000009055
Episode 219	Average Score: 0.01
epsilon: 4.82609900000091
Episode 220	Average Score: 0.01
epsilon: 4.825413000000913
Episode 221	Average Score: 0.01
epsilon: 4.824727000000917
Episode 222	Average Score: 0.01
epsilon: 4.82404100000092
Episode 223	Average Score: 0.01
epsilon: 4.823355000000924
Episode 224	Average Score: 0.01
epsilon: 4.82222800000093
Episode 225	Average Score: 0.01
epsilon: 4.8215420000009335
Episode 226	Average Score: 0.01
epsilon: 4.820072000000941
Episode 227	Average Score: 0.01
epsilon: 4.819141000000946
Episode 228	Average Score: 0.01
epsilon: 4.81845500000095
Episode 229	Average Score: 0.01
epsilon: 4.817769000000953
Episode 230	Average Score: 0.01
epsilon: 4.817034000000957
Episode 231	Average Score: 0.01
epsilon: 4.816348000000961
Episode 232	Average Score: 0.01
epsilon: 4.815662000000964
Episode 233	Average Score: 0.01
epsilon: 4.814927000000968
Episode 234	Average Score: 0.01
epsilon: 4.814192000000972
Episode 235	Average Score: 0.01
epsilon: 4.8135060000009755
Episode 236	Average Score: 0.01
epsilon: 4.812820000000979
Episode 237	Average Score: 0.01
epsilon: 4.812134000000983
Episode 238	Average Score: 0.01
epsilon: 4.811448000000986
Episode 239	Average Score: 0.01
epsilon: 4.81076200000099
Episode 240	Average Score: 0.01
epsilon: 4.809145000000998
Episode 241	Average Score: 0.01
epsilon: 4.808459000001002
Episode 242	Average Score: 0.01
epsilon: 4.8077730000010055
Episode 243	Average Score: 0.01
epsilon: 4.807087000001009
Episode 244	Average Score: 0.01
epsilon: 4.806352000001013
Episode 245	Average Score: 0.01
epsilon: 4.805617000001017
Episode 246	Average Score: 0.01
epsilon: 4.80493100000102
Episode 247	Average Score: 0.01
epsilon: 4.804245000001024
Episode 248	Average Score: 0.01
epsilon: 4.802726000001032
Episode 249	Average Score: 0.01
epsilon: 4.8020400000010355
Episode 250	Average Score: 0.01
epsilon: 4.801354000001039
Episode 251	Average Score: 0.01
epsilon: 4.800619000001043
Episode 252	Average Score: 0.01
epsilon: 4.7999330000010465
Episode 253	Average Score: 0.01
epsilon: 4.79924700000105
Episode 254	Average Score: 0.01
epsilon: 4.798561000001054
Episode 255	Average Score: 0.01
epsilon: 4.797091000001061
Episode 256	Average Score: 0.01
epsilon: 4.796258000001066
Episode 257	Average Score: 0.01
epsilon: 4.79552300000107
Episode 258	Average Score: 0.01
epsilon: 4.794837000001073
Episode 259	Average Score: 0.01
epsilon: 4.794151000001077
Episode 260	Average Score: 0.01
epsilon: 4.793171000001082
Episode 261	Average Score: 0.01
epsilon: 4.7924850000010855
Episode 262	Average Score: 0.01
epsilon: 4.791799000001089
Episode 263	Average Score: 0.01
epsilon: 4.791064000001093
Episode 264	Average Score: 0.01
epsilon: 4.7903780000010965
Episode 265	Average Score: 0.01
epsilon: 4.7896920000011
Episode 266	Average Score: 0.01
epsilon: 4.789006000001104
Episode 267	Average Score: 0.01
epsilon: 4.788320000001107
Episode 268	Average Score: 0.01
epsilon: 4.7867520000011154
Episode 269	Average Score: 0.01
epsilon: 4.786066000001119
Episode 270	Average Score: 0.01
epsilon: 4.784498000001127
Episode 271	Average Score: 0.01
epsilon: 4.783861000001131
Episode 272	Average Score: 0.01
epsilon: 4.783175000001134
Episode 273	Average Score: 0.01
epsilon: 4.782391000001138
Episode 274	Average Score: 0.01
epsilon: 4.781705000001142
Episode 275	Average Score: 0.01
epsilon: 4.781019000001145
Episode 276	Average Score: 0.01
epsilon: 4.780333000001149
Episode 277	Average Score: 0.01
epsilon: 4.778569000001158
Episode 278	Average Score: 0.01
epsilon: 4.777883000001162
Episode 279	Average Score: 0.01
epsilon: 4.777197000001165
Episode 280	Average Score: 0.01
epsilon: 4.776511000001169
Episode 281	Average Score: 0.01
epsilon: 4.775776000001173
Episode 282	Average Score: 0.01
epsilon: 4.775139000001176
Episode 283	Average Score: 0.01
epsilon: 4.77445300000118
Episode 284	Average Score: 0.01
epsilon: 4.773767000001183
Episode 285	Average Score: 0.01
epsilon: 4.773081000001187
Episode 286	Average Score: 0.01
epsilon: 4.772346000001191
Episode 287	Average Score: 0.01
epsilon: 4.771709000001194
Episode 288	Average Score: 0.01
epsilon: 4.770974000001198
Episode 289	Average Score: 0.01
epsilon: 4.770337000001201
Episode 290	Average Score: 0.01
epsilon: 4.769602000001205
Episode 291	Average Score: 0.01
epsilon: 4.768916000001209
Episode 292	Average Score: 0.01
epsilon: 4.768230000001212
Episode 293	Average Score: 0.01
epsilon: 4.767544000001216
Episode 294	Average Score: 0.01
epsilon: 4.7668580000012195
Episode 295	Average Score: 0.01
epsilon: 4.766172000001223
Episode 296	Average Score: 0.01
epsilon: 4.765437000001227
Episode 297	Average Score: 0.01
epsilon: 4.7647510000012305
Episode 298	Average Score: 0.01
epsilon: 4.764065000001234
Episode 299	Average Score: 0.01
epsilon: 4.763379000001238
Episode 300	Average Score: 0.01
epsilon: 4.762693000001241
Episode 301	Average Score: 0.01
epsilon: 4.761958000001245
Episode 302	Average Score: 0.01
epsilon: 4.7590180000012605
Episode 303	Average Score: 0.01
epsilon: 4.758332000001264
Episode 304	Average Score: 0.01
epsilon: 4.757597000001268
Episode 305	Average Score: 0.01
epsilon: 4.7569110000012715
Episode 306	Average Score: 0.01
epsilon: 4.756225000001275
Episode 307	Average Score: 0.01
epsilon: 4.755539000001279
Episode 308	Average Score: 0.01
epsilon: 4.754853000001282
Episode 309	Average Score: 0.01
epsilon: 4.754020000001287
Episode 310	Average Score: 0.01
epsilon: 4.75333400000129
Episode 311	Average Score: 0.01
epsilon: 4.752648000001294
Episode 312	Average Score: 0.01
epsilon: 4.751913000001298
Episode 313	Average Score: 0.01
epsilon: 4.751227000001301
Episode 314	Average Score: 0.01
epsilon: 4.749757000001309
Episode 315	Average Score: 0.01
epsilon: 4.7490710000013125
Episode 316	Average Score: 0.01
epsilon: 4.748385000001316
Episode 317	Average Score: 0.01
epsilon: 4.74769900000132
Episode 318	Average Score: 0.01
epsilon: 4.746915000001324
Episode 319	Average Score: 0.01
epsilon: 4.746229000001327
Episode 320	Average Score: 0.01
epsilon: 4.745543000001331
Episode 321	Average Score: 0.01
epsilon: 4.744857000001335
Episode 322	Average Score: 0.01
epsilon: 4.744171000001338
Episode 323	Average Score: 0.01
epsilon: 4.743485000001342
Episode 324	Average Score: 0.01
epsilon: 4.742799000001345
Episode 325	Average Score: 0.01
epsilon: 4.742064000001349
Episode 326	Average Score: 0.01
epsilon: 4.741378000001353
Episode 327	Average Score: 0.01
epsilon: 4.7399080000013605
Episode 328	Average Score: 0.01
epsilon: 4.738389000001368
Episode 329	Average Score: 0.01
epsilon: 4.737507000001373
Episode 330	Average Score: 0.01
epsilon: 4.736821000001377
Episode 331	Average Score: 0.01
epsilon: 4.73613500000138
Episode 332	Average Score: 0.01
epsilon: 4.735400000001384
Episode 333	Average Score: 0.01
epsilon: 4.733881000001392
Episode 334	Average Score: 0.01
epsilon: 4.733195000001396
Episode 335	Average Score: 0.01
epsilon: 4.732509000001399
Episode 336	Average Score: 0.01
epsilon: 4.731823000001403
Episode 337	Average Score: 0.01
epsilon: 4.731137000001406
Episode 338	Average Score: 0.01
epsilon: 4.73040200000141
Episode 339	Average Score: 0.01
epsilon: 4.729716000001414
Episode 340	Average Score: 0.01
epsilon: 4.729030000001417
Episode 341	Average Score: 0.01
epsilon: 4.728344000001421
Episode 342	Average Score: 0.01
epsilon: 4.7276580000014246
Episode 343	Average Score: 0.01
epsilon: 4.726923000001428
Episode 344	Average Score: 0.01
epsilon: 4.726237000001432
Episode 345	Average Score: 0.01
epsilon: 4.724963000001439
Episode 346	Average Score: 0.01
epsilon: 4.724277000001442
Episode 347	Average Score: 0.01
epsilon: 4.723591000001446
Episode 348	Average Score: 0.01
epsilon: 4.722905000001449
Episode 349	Average Score: 0.01
epsilon: 4.722170000001453
Episode 350	Average Score: 0.01
epsilon: 4.721484000001457
Episode 351	Average Score: 0.01
epsilon: 4.72079800000146
Episode 352	Average Score: 0.01
epsilon: 4.720112000001464
Episode 353	Average Score: 0.01
epsilon: 4.719328000001468
Episode 354	Average Score: 0.01
epsilon: 4.718544000001472
Episode 355	Average Score: 0.01
epsilon: 4.717760000001476
Episode 356	Average Score: 0.01
epsilon: 4.71707400000148
Episode 357	Average Score: 0.01
epsilon: 4.7163880000014835
Episode 358	Average Score: 0.01
epsilon: 4.715702000001487
Episode 359	Average Score: 0.01
epsilon: 4.714183000001495
Episode 360	Average Score: 0.01
epsilon: 4.713497000001499
Episode 361	Average Score: 0.01
epsilon: 4.712811000001502
Episode 362	Average Score: 0.01
epsilon: 4.71129200000151
Episode 363	Average Score: 0.01
epsilon: 4.710606000001514
Episode 364	Average Score: 0.01
epsilon: 4.709871000001518
Episode 365	Average Score: 0.01
epsilon: 4.709185000001521
Episode 366	Average Score: 0.01
epsilon: 4.708499000001525
Episode 367	Average Score: 0.01
epsilon: 4.707813000001528
Episode 368	Average Score: 0.01
epsilon: 4.707127000001532
Episode 369	Average Score: 0.01
epsilon: 4.7064410000015355
Episode 370	Average Score: 0.01
epsilon: 4.705755000001539
Episode 371	Average Score: 0.01
epsilon: 4.705069000001543
Episode 372	Average Score: 0.01
epsilon: 4.704334000001547
Episode 373	Average Score: 0.01
epsilon: 4.70359900000155
Episode 374	Average Score: 0.01
epsilon: 4.702913000001554
Episode 375	Average Score: 0.01
epsilon: 4.702227000001558
Episode 376	Average Score: 0.01
epsilon: 4.701541000001561
Episode 377	Average Score: 0.01
epsilon: 4.700806000001565
Episode 378	Average Score: 0.01
epsilon: 4.700120000001569
Episode 379	Average Score: 0.01
epsilon: 4.699434000001572
Episode 380	Average Score: 0.01
epsilon: 4.698748000001576
Episode 381	Average Score: 0.01
epsilon: 4.698062000001579
Episode 382	Average Score: 0.01
epsilon: 4.697376000001583
Episode 383	Average Score: 0.01
epsilon: 4.6966900000015865
Episode 384	Average Score: 0.01
epsilon: 4.69595500000159
Episode 385	Average Score: 0.01
epsilon: 4.695269000001594
Episode 386	Average Score: 0.01
epsilon: 4.694632000001597
Episode 387	Average Score: 0.01
epsilon: 4.693897000001601
Episode 388	Average Score: 0.01
epsilon: 4.693211000001605
Episode 389	Average Score: 0.01
epsilon: 4.692525000001608
Episode 390	Average Score: 0.01
epsilon: 4.691839000001612
Episode 391	Average Score: 0.01
epsilon: 4.6911530000016155
Episode 392	Average Score: 0.01
epsilon: 4.690467000001619
Episode 393	Average Score: 0.01
epsilon: 4.689781000001623
Episode 394	Average Score: 0.01
epsilon: 4.689095000001626
Episode 395	Average Score: 0.01
epsilon: 4.68840900000163
Episode 396	Average Score: 0.01
epsilon: 4.6877230000016334
Episode 397	Average Score: 0.01
epsilon: 4.687037000001637
Episode 398	Average Score: 0.01
epsilon: 4.686351000001641
Episode 399	Average Score: 0.01
epsilon: 4.685567000001645
Episode 400	Average Score: 0.01
epsilon: 4.684881000001648
Episode 401	Average Score: 0.01
epsilon: 4.684195000001652
Episode 402	Average Score: 0.01
epsilon: 4.6835090000016555
Episode 403	Average Score: 0.01
epsilon: 4.682774000001659
Episode 404	Average Score: 0.01
epsilon: 4.682088000001663
Episode 405	Average Score: 0.01
epsilon: 4.6814020000016665
Episode 406	Average Score: 0.01
epsilon: 4.68066700000167
Episode 407	Average Score: 0.01
epsilon: 4.679981000001674
Episode 408	Average Score: 0.01
epsilon: 4.6792950000016775
Episode 409	Average Score: 0.01
epsilon: 4.678609000001681
Episode 410	Average Score: 0.01
epsilon: 4.677874000001685
Episode 411	Average Score: 0.01
epsilon: 4.6771880000016886
Episode 412	Average Score: 0.01
epsilon: 4.676502000001692
Episode 413	Average Score: 0.01
epsilon: 4.675816000001696
Episode 414	Average Score: 0.01
epsilon: 4.675130000001699
Episode 415	Average Score: 0.01
epsilon: 4.674395000001703
Episode 416	Average Score: 0.01
epsilon: 4.673709000001707
Episode 417	Average Score: 0.01
epsilon: 4.672190000001715
Episode 418	Average Score: 0.01
epsilon: 4.670524000001723
Episode 419	Average Score: 0.01
epsilon: 4.669005000001731
Episode 420	Average Score: 0.01
epsilon: 4.668319000001735
Episode 421	Average Score: 0.01
epsilon: 4.667584000001739
Episode 422	Average Score: 0.01
epsilon: 4.666898000001742
Episode 423	Average Score: 0.01
epsilon: 4.666212000001746
Episode 424	Average Score: 0.01
epsilon: 4.6655260000017496
Episode 425	Average Score: 0.01
epsilon: 4.664840000001753
Episode 426	Average Score: 0.01
epsilon: 4.664154000001757
Episode 427	Average Score: 0.01
epsilon: 4.663419000001761
Episode 428	Average Score: 0.01
epsilon: 4.662733000001764
Episode 429	Average Score: 0.01
epsilon: 4.662047000001768
Episode 430	Average Score: 0.01
epsilon: 4.661361000001771
Episode 431	Average Score: 0.01
epsilon: 4.660675000001775
Episode 432	Average Score: 0.01
epsilon: 4.659940000001779
Episode 433	Average Score: 0.01
epsilon: 4.659254000001782
Episode 434	Average Score: 0.01
epsilon: 4.658568000001786
Episode 435	Average Score: 0.01
epsilon: 4.65783300000179
Episode 436	Average Score: 0.01
epsilon: 4.657147000001793
Episode 437	Average Score: 0.01
epsilon: 4.656461000001797
Episode 438	Average Score: 0.01
epsilon: 4.655775000001801
Episode 439	Average Score: 0.01
epsilon: 4.654501000001807
Episode 440	Average Score: 0.01
epsilon: 4.653815000001811
Episode 441	Average Score: 0.01
epsilon: 4.653080000001815
Episode 442	Average Score: 0.01
epsilon: 4.652394000001818
Episode 443	Average Score: 0.01
epsilon: 4.651708000001822
Episode 444	Average Score: 0.01
epsilon: 4.6506300000018275
Episode 445	Average Score: 0.01
epsilon: 4.649944000001831
Episode 446	Average Score: 0.01
epsilon: 4.649209000001835
Episode 447	Average Score: 0.01
epsilon: 4.6485230000018385
Episode 448	Average Score: 0.01
epsilon: 4.647837000001842
Episode 449	Average Score: 0.01
epsilon: 4.647151000001846
Episode 450	Average Score: 0.01
epsilon: 4.646465000001849
Episode 451	Average Score: 0.01
epsilon: 4.645779000001853
Episode 452	Average Score: 0.01
epsilon: 4.645093000001856
Episode 453	Average Score: 0.01
epsilon: 4.64440700000186
Episode 454	Average Score: 0.01
epsilon: 4.643672000001864
Episode 455	Average Score: 0.01
epsilon: 4.6422020000018716
Episode 456	Average Score: 0.01
epsilon: 4.641516000001875
Episode 457	Average Score: 0.01
epsilon: 4.640095000001883
Episode 458	Average Score: 0.01
epsilon: 4.639360000001886
Episode 459	Average Score: 0.01
epsilon: 4.63867400000189
Episode 460	Average Score: 0.01
epsilon: 4.637939000001894
Episode 461	Average Score: 0.01
epsilon: 4.637253000001897
Episode 462	Average Score: 0.01
epsilon: 4.636567000001901
Episode 463	Average Score: 0.01
epsilon: 4.635881000001905
Episode 464	Average Score: 0.01
epsilon: 4.635195000001908
Episode 465	Average Score: 0.01
epsilon: 4.634460000001912
Episode 466	Average Score: 0.01
epsilon: 4.633774000001916
Episode 467	Average Score: 0.01
epsilon: 4.633088000001919
Episode 468	Average Score: 0.01
epsilon: 4.632402000001923
Episode 469	Average Score: 0.01
epsilon: 4.631716000001926
Episode 470	Average Score: 0.01
epsilon: 4.63098100000193
Episode 471	Average Score: 0.01
epsilon: 4.630295000001934
Episode 472	Average Score: 0.01
epsilon: 4.629609000001937
Episode 473	Average Score: 0.01
epsilon: 4.628923000001941
Episode 474	Average Score: 0.01
epsilon: 4.628237000001945
Episode 475	Average Score: 0.01
epsilon: 4.6275020000019484
Episode 476	Average Score: 0.01
epsilon: 4.626816000001952
Episode 477	Average Score: 0.01
epsilon: 4.626130000001956
Episode 478	Average Score: 0.01
epsilon: 4.625444000001959
Episode 479	Average Score: 0.01
epsilon: 4.624758000001963
Episode 480	Average Score: 0.01
epsilon: 4.624072000001966
Episode 481	Average Score: 0.01
epsilon: 4.62338600000197
Episode 482	Average Score: 0.01
epsilon: 4.622651000001974
Episode 483	Average Score: 0.01
epsilon: 4.621965000001977
Episode 484	Average Score: 0.01
epsilon: 4.621279000001981
Episode 485	Average Score: 0.01
epsilon: 4.620054000001987
Episode 486	Average Score: 0.01
epsilon: 4.619368000001991
Episode 487	Average Score: 0.01
epsilon: 4.618682000001995
Episode 488	Average Score: 0.01
epsilon: 4.617996000001998
Episode 489	Average Score: 0.01
epsilon: 4.617261000002002
Episode 490	Average Score: 0.01
epsilon: 4.616575000002006
Episode 491	Average Score: 0.01
epsilon: 4.615007000002014
Episode 492	Average Score: 0.01
epsilon: 4.614272000002018
Episode 493	Average Score: 0.01
epsilon: 4.613586000002021
Episode 494	Average Score: 0.01
epsilon: 4.612900000002025
Episode 495	Average Score: 0.01
epsilon: 4.611381000002033
Episode 496	Average Score: 0.01
epsilon: 4.610695000002036
Episode 497	Average Score: 0.01
epsilon: 4.61000900000204
Episode 498	Average Score: 0.01
epsilon: 4.6093230000020435
Episode 499	Average Score: 0.01
epsilon: 4.608637000002047
Episode 500	Average Score: 0.01
epsilon: 4.607951000002051
Episode 501	Average Score: 0.01
epsilon: 4.607265000002054
Episode 502	Average Score: 0.01
epsilon: 4.606579000002058
Episode 503	Average Score: 0.01
epsilon: 4.6058930000020615
Episode 504	Average Score: 0.01
epsilon: 4.605158000002065
Episode 505	Average Score: 0.01
epsilon: 4.604472000002069
Episode 506	Average Score: 0.01
epsilon: 4.602855000002077
Episode 507	Average Score: 0.01
epsilon: 4.602169000002081
Episode 508	Average Score: 0.01
epsilon: 4.601434000002085
Episode 509	Average Score: 0.01
epsilon: 4.600797000002088
Episode 510	Average Score: 0.01
epsilon: 4.600111000002092
Episode 511	Average Score: 0.01
epsilon: 4.599376000002096
Episode 512	Average Score: 0.01
epsilon: 4.598690000002099
Episode 513	Average Score: 0.01
epsilon: 4.598004000002103
Episode 514	Average Score: 0.01
epsilon: 4.597318000002106
Episode 515	Average Score: 0.01
epsilon: 4.59663200000211
Episode 516	Average Score: 0.01
epsilon: 4.595897000002114
Episode 517	Average Score: 0.01
epsilon: 4.595211000002117
Episode 518	Average Score: 0.01
epsilon: 4.594525000002121
Episode 519	Average Score: 0.01
epsilon: 4.5938390000021245
Episode 520	Average Score: 0.01
epsilon: 4.593153000002128
Episode 521	Average Score: 0.01
epsilon: 4.592418000002132
Episode 522	Average Score: 0.01
epsilon: 4.5917320000021355
Episode 523	Average Score: 0.01
epsilon: 4.591046000002139
Episode 524	Average Score: 0.01
epsilon: 4.590360000002143
Episode 525	Average Score: 0.01
epsilon: 4.589674000002146
Episode 526	Average Score: 0.01
epsilon: 4.58893900000215
Episode 527	Average Score: 0.01
epsilon: 4.588253000002154
Episode 528	Average Score: 0.01
epsilon: 4.587567000002157
Episode 529	Average Score: 0.01
epsilon: 4.586783000002161
Episode 530	Average Score: 0.01
epsilon: 4.586097000002165
Episode 531	Average Score: 0.01
epsilon: 4.5844800000021735
Episode 532	Average Score: 0.01
epsilon: 4.583794000002177
Episode 533	Average Score: 0.01
epsilon: 4.582324000002185
Episode 534	Average Score: 0.01
epsilon: 4.581638000002188
Episode 535	Average Score: 0.01
epsilon: 4.580952000002192
Episode 536	Average Score: 0.01
epsilon: 4.5802660000021955
Episode 537	Average Score: 0.01
epsilon: 4.579531000002199
Episode 538	Average Score: 0.01
epsilon: 4.578845000002203
Episode 539	Average Score: 0.01
epsilon: 4.5781590000022065
Episode 540	Average Score: 0.01
epsilon: 4.57747300000221
Episode 541	Average Score: 0.01
epsilon: 4.576787000002214
Episode 542	Average Score: 0.01
epsilon: 4.576101000002217
Episode 543	Average Score: 0.01
epsilon: 4.575366000002221
Episode 544	Average Score: 0.01
epsilon: 4.5747290000022245
Episode 545	Average Score: 0.01
epsilon: 4.574043000002228
Episode 546	Average Score: 0.01
epsilon: 4.573357000002232
Episode 547	Average Score: 0.01
epsilon: 4.572671000002235
Episode 548	Average Score: 0.01
epsilon: 4.571642000002241
Episode 549	Average Score: 0.01
epsilon: 4.570956000002244
Episode 550	Average Score: 0.01
epsilon: 4.570221000002248
Episode 551	Average Score: 0.01
epsilon: 4.569535000002252
Episode 552	Average Score: 0.01
epsilon: 4.5688000000022555
Episode 553	Average Score: 0.01
epsilon: 4.568114000002259
Episode 554	Average Score: 0.01
epsilon: 4.567428000002263
Episode 555	Average Score: 0.01
epsilon: 4.566742000002266
Episode 556	Average Score: 0.01
epsilon: 4.56605600000227
Episode 557	Average Score: 0.00
epsilon: 4.565321000002274
Episode 558	Average Score: 0.00
epsilon: 4.564635000002277
Episode 559	Average Score: 0.00
epsilon: 4.563949000002281
Episode 560	Average Score: 0.00
epsilon: 4.5632630000022845
Episode 561	Average Score: 0.00
epsilon: 4.562577000002288
Episode 562	Average Score: 0.00
epsilon: 4.561891000002292
Episode 563	Average Score: 0.00
epsilon: 4.561205000002295
Episode 564	Average Score: 0.00
epsilon: 4.560470000002299
Episode 565	Average Score: 0.00
epsilon: 4.559784000002303
Episode 566	Average Score: 0.00
epsilon: 4.559098000002306
Episode 567	Average Score: 0.00
epsilon: 4.55841200000231
Episode 568	Average Score: 0.00
epsilon: 4.557726000002313
Episode 569	Average Score: 0.00
epsilon: 4.556991000002317
Episode 570	Average Score: 0.00
epsilon: 4.556305000002321
Episode 571	Average Score: 0.00
epsilon: 4.555472000002325
Episode 572	Average Score: 0.00
epsilon: 4.554786000002329
Episode 573	Average Score: 0.00
epsilon: 4.554100000002332
Episode 574	Average Score: 0.00
epsilon: 4.553463000002336
Episode 575	Average Score: 0.00
epsilon: 4.55272800000234
Episode 576	Average Score: 0.00
epsilon: 4.552042000002343
Episode 577	Average Score: 0.00
epsilon: 4.551356000002347
Episode 578	Average Score: 0.00
epsilon: 4.55067000000235
Episode 579	Average Score: 0.00
epsilon: 4.549984000002354
Episode 580	Average Score: 0.00
epsilon: 4.549249000002358
Episode 581	Average Score: 0.01
epsilon: 4.547681000002366
Episode 582	Average Score: 0.01
epsilon: 4.54694600000237
Episode 583	Average Score: 0.01
epsilon: 4.546309000002373
Episode 584	Average Score: 0.01
epsilon: 4.545623000002377
Episode 585	Average Score: 0.01
epsilon: 4.544888000002381
Episode 586	Average Score: 0.01
epsilon: 4.544202000002384
Episode 587	Average Score: 0.01
epsilon: 4.5411640000024
Episode 588	Average Score: 0.01
epsilon: 4.540478000002404
Episode 589	Average Score: 0.01
epsilon: 4.539792000002407
Episode 590	Average Score: 0.01
epsilon: 4.539106000002411
Episode 591	Average Score: 0.01
epsilon: 4.538420000002414
Episode 592	Average Score: 0.01
epsilon: 4.537685000002418
Episode 593	Average Score: 0.01
epsilon: 4.536999000002422
Episode 594	Average Score: 0.01
epsilon: 4.53548000000243
Episode 595	Average Score: 0.01
epsilon: 4.534794000002433
Episode 596	Average Score: 0.01
epsilon: 4.534108000002437
Episode 597	Average Score: 0.01
epsilon: 4.5334220000024406
Episode 598	Average Score: 0.01
epsilon: 4.532736000002444
Episode 599	Average Score: 0.01
epsilon: 4.532001000002448
Episode 600	Average Score: 0.01
epsilon: 4.531315000002452
Episode 601	Average Score: 0.01
epsilon: 4.530629000002455
Episode 602	Average Score: 0.01
epsilon: 4.529943000002459
Episode 603	Average Score: 0.01
epsilon: 4.529257000002462
Episode 604	Average Score: 0.01
epsilon: 4.528522000002466
Episode 605	Average Score: 0.01
epsilon: 4.52783600000247
Episode 606	Average Score: 0.00
epsilon: 4.527150000002473
Episode 607	Average Score: 0.00
epsilon: 4.526121000002479
Episode 608	Average Score: 0.00
epsilon: 4.525435000002482
Episode 609	Average Score: 0.01
epsilon: 4.524308000002488
Episode 610	Average Score: 0.01
epsilon: 4.523622000002492
Episode 611	Average Score: 0.01
epsilon: 4.522936000002495
Episode 612	Average Score: 0.01
epsilon: 4.522299000002499
Episode 613	Average Score: 0.01
epsilon: 4.521564000002503
Episode 614	Average Score: 0.01
epsilon: 4.5200450000025105
Episode 615	Average Score: 0.01
epsilon: 4.519359000002514
Episode 616	Average Score: 0.01
epsilon: 4.518673000002518
Episode 617	Average Score: 0.01
epsilon: 4.517987000002521
Episode 618	Average Score: 0.01
epsilon: 4.517301000002525
Episode 619	Average Score: 0.01
epsilon: 4.515782000002533
Episode 620	Average Score: 0.01
epsilon: 4.515047000002537
Episode 621	Average Score: 0.01
epsilon: 4.51436100000254
Episode 622	Average Score: 0.01
epsilon: 4.513675000002544
Episode 623	Average Score: 0.01
epsilon: 4.512989000002547
Episode 624	Average Score: 0.01
epsilon: 4.512303000002551
Episode 625	Average Score: 0.01
epsilon: 4.511568000002555
Episode 626	Average Score: 0.01
epsilon: 4.510098000002563
Episode 627	Average Score: 0.01
epsilon: 4.509412000002566
Episode 628	Average Score: 0.01
epsilon: 4.50867700000257
Episode 629	Average Score: 0.01
epsilon: 4.507991000002574
Episode 630	Average Score: 0.01
epsilon: 4.507305000002577
Episode 631	Average Score: 0.01
epsilon: 4.506619000002581
Episode 632	Average Score: 0.01
epsilon: 4.505933000002584
Episode 633	Average Score: 0.01
epsilon: 4.505198000002588
Episode 634	Average Score: 0.01
epsilon: 4.504512000002592
Episode 635	Average Score: 0.01
epsilon: 4.503826000002595
Episode 636	Average Score: 0.01
epsilon: 4.502356000002603
Episode 637	Average Score: 0.01
epsilon: 4.501670000002607
Episode 638	Average Score: 0.01
epsilon: 4.500151000002615
Episode 639	Average Score: 0.01
epsilon: 4.499465000002618
Episode 640	Average Score: 0.01
epsilon: 4.498779000002622
Episode 641	Average Score: 0.01
epsilon: 4.498093000002625
Episode 642	Average Score: 0.01
epsilon: 4.497407000002629
Episode 643	Average Score: 0.01
epsilon: 4.4967210000026325
Episode 644	Average Score: 0.01
epsilon: 4.496035000002636
Episode 645	Average Score: 0.01
epsilon: 4.494516000002644
Episode 646	Average Score: 0.01
epsilon: 4.493781000002648
Episode 647	Average Score: 0.01
epsilon: 4.4930950000026515
Episode 648	Average Score: 0.01
epsilon: 4.492409000002655
Episode 649	Average Score: 0.01
epsilon: 4.491723000002659
Episode 650	Average Score: 0.01
epsilon: 4.491037000002662
Episode 651	Average Score: 0.01
epsilon: 4.490351000002666
Episode 652	Average Score: 0.01
epsilon: 4.489665000002669
Episode 653	Average Score: 0.01
epsilon: 4.488979000002673
Episode 654	Average Score: 0.01
epsilon: 4.488293000002677
Episode 655	Average Score: 0.01
epsilon: 4.48760700000268
Episode 656	Average Score: 0.01
epsilon: 4.486921000002684
Episode 657	Average Score: 0.01
epsilon: 4.486235000002687
Episode 658	Average Score: 0.01
epsilon: 4.485500000002691
Episode 659	Average Score: 0.01
epsilon: 4.484814000002695
Episode 660	Average Score: 0.01
epsilon: 4.484128000002698
Episode 661	Average Score: 0.01
epsilon: 4.483442000002702
Episode 662	Average Score: 0.01
epsilon: 4.482756000002706
Episode 663	Average Score: 0.01
epsilon: 4.482070000002709
Episode 664	Average Score: 0.01
epsilon: 4.481384000002713
Episode 665	Average Score: 0.01
epsilon: 4.480698000002716
Episode 666	Average Score: 0.01
epsilon: 4.48001200000272
Episode 667	Average Score: 0.01
epsilon: 4.478591000002727
Episode 668	Average Score: 0.01
epsilon: 4.477905000002731
Episode 669	Average Score: 0.01
epsilon: 4.4772190000027345
Episode 670	Average Score: 0.01
epsilon: 4.475749000002742
Episode 671	Average Score: 0.01
epsilon: 4.475063000002746
Episode 672	Average Score: 0.01
epsilon: 4.474377000002749
Episode 673	Average Score: 0.01
epsilon: 4.473642000002753
Episode 674	Average Score: 0.01
epsilon: 4.472956000002757
Episode 675	Average Score: 0.01
epsilon: 4.47227000000276
Episode 676	Average Score: 0.01
epsilon: 4.4714860000027645
Episode 677	Average Score: 0.01
epsilon: 4.470800000002768
Episode 678	Average Score: 0.01
epsilon: 4.470114000002772
Episode 679	Average Score: 0.01
epsilon: 4.469428000002775
Episode 680	Average Score: 0.01
epsilon: 4.468742000002779
Episode 681	Average Score: 0.01
epsilon: 4.4680560000027825
Episode 682	Average Score: 0.01
epsilon: 4.467370000002786
Episode 683	Average Score: 0.01
epsilon: 4.46668400000279
Episode 684	Average Score: 0.01
epsilon: 4.465998000002793
Episode 685	Average Score: 0.01
epsilon: 4.465263000002797
Episode 686	Average Score: 0.01
epsilon: 4.464577000002801
Episode 687	Average Score: 0.01
epsilon: 4.463744000002805
Episode 688	Average Score: 0.01
epsilon: 4.463009000002809
Episode 689	Average Score: 0.01
epsilon: 4.4623230000028125
Episode 690	Average Score: 0.01
epsilon: 4.461637000002816
Episode 691	Average Score: 0.01
epsilon: 4.46095100000282
Episode 692	Average Score: 0.01
epsilon: 4.4602160000028235
Episode 693	Average Score: 0.01
epsilon: 4.459530000002827
Episode 694	Average Score: 0.01
epsilon: 4.458844000002831
Episode 695	Average Score: 0.01
epsilon: 4.458158000002834
Episode 696	Average Score: 0.01
epsilon: 4.457472000002838
Episode 697	Average Score: 0.01
epsilon: 4.456737000002842
Episode 698	Average Score: 0.01
epsilon: 4.456051000002845
Episode 699	Average Score: 0.01
epsilon: 4.455365000002849
Episode 700	Average Score: 0.01
epsilon: 4.454679000002852
Episode 701	Average Score: 0.01
epsilon: 4.4538950000028565
Episode 702	Average Score: 0.01
epsilon: 4.45320900000286
Episode 703	Average Score: 0.01
epsilon: 4.4525720000028635
Episode 704	Average Score: 0.01
epsilon: 4.451837000002867
Episode 705	Average Score: 0.01
epsilon: 4.451151000002871
Episode 706	Average Score: 0.01
epsilon: 4.4504650000028745
Episode 707	Average Score: 0.01
epsilon: 4.449779000002878
Episode 708	Average Score: 0.01
epsilon: 4.449093000002882
Episode 709	Average Score: 0.01
epsilon: 4.448407000002885
Episode 710	Average Score: 0.01
epsilon: 4.447721000002889
Episode 711	Average Score: 0.01
epsilon: 4.447035000002892
Episode 712	Average Score: 0.01
epsilon: 4.446349000002896
Episode 713	Average Score: 0.01
epsilon: 4.443556000002911
Episode 714	Average Score: 0.01
epsilon: 4.442870000002914
Episode 715	Average Score: 0.01
epsilon: 4.442184000002918
Episode 716	Average Score: 0.01
epsilon: 4.441498000002921
Episode 717	Average Score: 0.01
epsilon: 4.440812000002925
Episode 718	Average Score: 0.01
epsilon: 4.440126000002929
Episode 719	Average Score: 0.01
epsilon: 4.439440000002932
Episode 720	Average Score: 0.01
epsilon: 4.438754000002936
Episode 721	Average Score: 0.01
epsilon: 4.438068000002939
Episode 722	Average Score: 0.01
epsilon: 4.436549000002947
Episode 723	Average Score: 0.01
epsilon: 4.435079000002955
Episode 724	Average Score: 0.01
epsilon: 4.4343930000029586
Episode 725	Average Score: 0.01
epsilon: 4.433707000002962
Episode 726	Average Score: 0.01
epsilon: 4.43218800000297
Episode 727	Average Score: 0.01
epsilon: 4.431502000002974
Episode 728	Average Score: 0.01
epsilon: 4.430816000002977
Episode 729	Average Score: 0.01
epsilon: 4.430130000002981
Episode 730	Average Score: 0.01
epsilon: 4.429395000002985
Episode 731	Average Score: 0.01
epsilon: 4.428709000002988
Episode 732	Average Score: 0.01
epsilon: 4.428023000002992
Episode 733	Average Score: 0.01
epsilon: 4.426553000003
Episode 734	Average Score: 0.01
epsilon: 4.425867000003003
Episode 735	Average Score: 0.01
epsilon: 4.425132000003007
Episode 736	Average Score: 0.01
epsilon: 4.424446000003011
Episode 737	Average Score: 0.01
epsilon: 4.423760000003014
Episode 738	Average Score: 0.01
epsilon: 4.423074000003018
Episode 739	Average Score: 0.01
epsilon: 4.422388000003021
Episode 740	Average Score: 0.01
epsilon: 4.421653000003025
Episode 741	Average Score: 0.01
epsilon: 4.420967000003029
Episode 742	Average Score: 0.01
epsilon: 4.420281000003032
Episode 743	Average Score: 0.01
epsilon: 4.419595000003036
Episode 744	Average Score: 0.01
epsilon: 4.41886000000304
Episode 745	Average Score: 0.01
epsilon: 4.418125000003044
Episode 746	Average Score: 0.01
epsilon: 4.417439000003047
Episode 747	Average Score: 0.01
epsilon: 4.416753000003051
Episode 748	Average Score: 0.01
epsilon: 4.416067000003054
Episode 749	Average Score: 0.01
epsilon: 4.415381000003058
Episode 750	Average Score: 0.01
epsilon: 4.414646000003062
Episode 751	Average Score: 0.01
epsilon: 4.413960000003065
Episode 752	Average Score: 0.01
epsilon: 4.413274000003069
Episode 753	Average Score: 0.01
epsilon: 4.412588000003073
Episode 754	Average Score: 0.01
epsilon: 4.411902000003076
Episode 755	Average Score: 0.01
epsilon: 4.41116700000308
Episode 756	Average Score: 0.01
epsilon: 4.410481000003084
Episode 757	Average Score: 0.01
epsilon: 4.409795000003087
Episode 758	Average Score: 0.01
epsilon: 4.4091580000030905
Episode 759	Average Score: 0.01
epsilon: 4.408423000003094
Episode 760	Average Score: 0.01
epsilon: 4.407737000003098
Episode 761	Average Score: 0.01
epsilon: 4.407051000003102
Episode 762	Average Score: 0.01
epsilon: 4.406365000003105
Episode 763	Average Score: 0.01
epsilon: 4.405679000003109
Episode 764	Average Score: 0.01
epsilon: 4.404993000003112
Episode 765	Average Score: 0.01
epsilon: 4.404307000003116
Episode 766	Average Score: 0.01
epsilon: 4.4036210000031195
Episode 767	Average Score: 0.01
epsilon: 4.402886000003123
Episode 768	Average Score: 0.01
epsilon: 4.402200000003127
Episode 769	Average Score: 0.01
epsilon: 4.4015140000031305
Episode 770	Average Score: 0.00
epsilon: 4.400828000003134
Episode 771	Average Score: 0.00
epsilon: 4.400142000003138
Episode 772	Average Score: 0.00
epsilon: 4.399456000003141
Episode 773	Average Score: 0.00
epsilon: 4.398770000003145
Episode 774	Average Score: 0.00
epsilon: 4.398035000003149
Episode 775	Average Score: 0.00
epsilon: 4.397349000003152
Episode 776	Average Score: 0.00
epsilon: 4.396663000003156
Episode 777	Average Score: 0.01
epsilon: 4.394115000003169
Episode 778	Average Score: 0.01
epsilon: 4.393429000003173
Episode 779	Average Score: 0.01
epsilon: 4.391077000003185
Episode 780	Average Score: 0.01
epsilon: 4.390391000003189
Episode 781	Average Score: 0.01
epsilon: 4.389705000003192
Episode 782	Average Score: 0.01
epsilon: 4.389019000003196
Episode 783	Average Score: 0.01
epsilon: 4.3883330000031995
Episode 784	Average Score: 0.01
epsilon: 4.387598000003203
Episode 785	Average Score: 0.01
epsilon: 4.386177000003211
Episode 786	Average Score: 0.01
epsilon: 4.385491000003214
Episode 787	Average Score: 0.01
epsilon: 4.384756000003218
Episode 788	Average Score: 0.01
epsilon: 4.384070000003222
Episode 789	Average Score: 0.01
epsilon: 4.383384000003225
Episode 790	Average Score: 0.01
epsilon: 4.382698000003229
Episode 791	Average Score: 0.01
epsilon: 4.3820120000032325
Episode 792	Average Score: 0.01
epsilon: 4.38059100000324
Episode 793	Average Score: 0.01
epsilon: 4.379954000003243
Episode 794	Average Score: 0.01
epsilon: 4.379219000003247
Episode 795	Average Score: 0.01
epsilon: 4.378533000003251
Episode 796	Average Score: 0.01
epsilon: 4.377847000003254
Episode 797	Average Score: 0.01
epsilon: 4.377161000003258
Episode 798	Average Score: 0.01
epsilon: 4.376377000003262
Episode 799	Average Score: 0.01
epsilon: 4.375691000003266
Episode 800	Average Score: 0.01
epsilon: 4.375005000003269
Episode 801	Average Score: 0.01
epsilon: 4.374270000003273
Episode 802	Average Score: 0.01
epsilon: 4.373584000003277
Episode 803	Average Score: 0.01
epsilon: 4.37289800000328
Episode 804	Average Score: 0.01
epsilon: 4.372114000003284
Episode 805	Average Score: 0.01
epsilon: 4.371428000003288
Episode 806	Average Score: 0.01
epsilon: 4.3707420000032915
Episode 807	Average Score: 0.01
epsilon: 4.369223000003299
Episode 808	Average Score: 0.01
epsilon: 4.366626000003313
Episode 809	Average Score: 0.01
epsilon: 4.365989000003316
Episode 810	Average Score: 0.01
epsilon: 4.36525400000332
Episode 811	Average Score: 0.01
epsilon: 4.364568000003324
Episode 812	Average Score: 0.01
epsilon: 4.363882000003327
Episode 813	Average Score: 0.01
epsilon: 4.363196000003331
Episode 814	Average Score: 0.01
epsilon: 4.362461000003335
Episode 815	Average Score: 0.01
epsilon: 4.361775000003338
Episode 816	Average Score: 0.01
epsilon: 4.360305000003346
Episode 817	Average Score: 0.01
epsilon: 4.35961900000335
Episode 818	Average Score: 0.01
epsilon: 4.358835000003354
Episode 819	Average Score: 0.01
epsilon: 4.358100000003358
Episode 820	Average Score: 0.01
epsilon: 4.357414000003361
Episode 821	Average Score: 0.01
epsilon: 4.356728000003365
Episode 822	Average Score: 0.01
epsilon: 4.356042000003368
Episode 823	Average Score: 0.01
epsilon: 4.3531510000033835
Episode 824	Average Score: 0.01
epsilon: 4.352465000003387
Episode 825	Average Score: 0.01
epsilon: 4.351779000003391
Episode 826	Average Score: 0.01
epsilon: 4.351093000003394
Episode 827	Average Score: 0.01
epsilon: 4.350358000003398
Episode 828	Average Score: 0.01
epsilon: 4.3497210000034014
Episode 829	Average Score: 0.01
epsilon: 4.348986000003405
Episode 830	Average Score: 0.01
epsilon: 4.348300000003409
Episode 831	Average Score: 0.01
epsilon: 4.3476140000034125
Episode 832	Average Score: 0.01
epsilon: 4.346928000003416
Episode 833	Average Score: 0.01
epsilon: 4.34624200000342
Episode 834	Average Score: 0.01
epsilon: 4.3455070000034235
Episode 835	Average Score: 0.01
epsilon: 4.344821000003427
Episode 836	Average Score: 0.01
epsilon: 4.344135000003431
Episode 837	Average Score: 0.01
epsilon: 4.343449000003434
Episode 838	Average Score: 0.01
epsilon: 4.342763000003438
Episode 839	Average Score: 0.01
epsilon: 4.342028000003442
Episode 840	Average Score: 0.01
epsilon: 4.340607000003449
Episode 841	Average Score: 0.01
epsilon: 4.339872000003453
Episode 842	Average Score: 0.01
epsilon: 4.339137000003457
Episode 843	Average Score: 0.01
epsilon: 4.338157000003462
Episode 844	Average Score: 0.01
epsilon: 4.3374710000034655
Episode 845	Average Score: 0.01
epsilon: 4.336785000003469
Episode 846	Average Score: 0.01
epsilon: 4.336050000003473
Episode 847	Average Score: 0.01
epsilon: 4.3353640000034765
Episode 848	Average Score: 0.01
epsilon: 4.33467800000348
Episode 849	Average Score: 0.01
epsilon: 4.332179000003493
Episode 850	Average Score: 0.01
epsilon: 4.331493000003497
Episode 851	Average Score: 0.01
epsilon: 4.3308070000035
Episode 852	Average Score: 0.01
epsilon: 4.330072000003504
Episode 853	Average Score: 0.01
epsilon: 4.329435000003508
Episode 854	Average Score: 0.01
epsilon: 4.328700000003511
Episode 855	Average Score: 0.01
epsilon: 4.328014000003515
Episode 856	Average Score: 0.01
epsilon: 4.327328000003519
Episode 857	Average Score: 0.01
epsilon: 4.325711000003527
Episode 858	Average Score: 0.01
epsilon: 4.325025000003531
Episode 859	Average Score: 0.01
epsilon: 4.323457000003539
Episode 860	Average Score: 0.01
epsilon: 4.322771000003542
Episode 861	Average Score: 0.01
epsilon: 4.322085000003546
Episode 862	Average Score: 0.01
epsilon: 4.32139900000355
Episode 863	Average Score: 0.01
epsilon: 4.320664000003553
Episode 864	Average Score: 0.01
epsilon: 4.319145000003561
Episode 865	Average Score: 0.01
epsilon: 4.318410000003565
Episode 866	Average Score: 0.01
epsilon: 4.317724000003569
Episode 867	Average Score: 0.01
epsilon: 4.317038000003572
Episode 868	Average Score: 0.01
epsilon: 4.316352000003576
Episode 869	Average Score: 0.01
epsilon: 4.31566600000358
Episode 870	Average Score: 0.01
epsilon: 4.314931000003583
Episode 871	Average Score: 0.01
epsilon: 4.314294000003587
Episode 872	Average Score: 0.01
epsilon: 4.31360800000359
Episode 873	Average Score: 0.01
epsilon: 4.312873000003594
Episode 874	Average Score: 0.01
epsilon: 4.312187000003598
Episode 875	Average Score: 0.01
epsilon: 4.311501000003601
Episode 876	Average Score: 0.01
epsilon: 4.310815000003605
Episode 877	Average Score: 0.01
epsilon: 4.3101290000036085
Episode 878	Average Score: 0.01
epsilon: 4.309394000003612
Episode 879	Average Score: 0.01
epsilon: 4.308708000003616
Episode 880	Average Score: 0.01
epsilon: 4.30802200000362
Episode 881	Average Score: 0.01
epsilon: 4.307140000003624
Episode 882	Average Score: 0.01
epsilon: 4.305670000003632
Episode 883	Average Score: 0.01
epsilon: 4.3049840000036355
Episode 884	Average Score: 0.01
epsilon: 4.304249000003639
Episode 885	Average Score: 0.01
epsilon: 4.3026810000036475
Episode 886	Average Score: 0.01
epsilon: 4.301995000003651
Episode 887	Average Score: 0.01
epsilon: 4.301309000003655
Episode 888	Average Score: 0.01
epsilon: 4.300623000003658
Episode 889	Average Score: 0.01
epsilon: 4.299937000003662
Episode 890	Average Score: 0.01
epsilon: 4.2992510000036654
Episode 891	Average Score: 0.01
epsilon: 4.298565000003669
Episode 892	Average Score: 0.01
epsilon: 4.297830000003673
Episode 893	Average Score: 0.01
epsilon: 4.2971440000036765
Episode 894	Average Score: 0.01
epsilon: 4.29645800000368
Episode 895	Average Score: 0.01
epsilon: 4.295772000003684
Episode 896	Average Score: 0.01
epsilon: 4.295086000003687
Episode 897	Average Score: 0.01
epsilon: 4.294351000003691
Episode 898	Average Score: 0.01
epsilon: 4.293665000003695
Episode 899	Average Score: 0.01
epsilon: 4.293028000003698
Episode 900	Average Score: 0.01
epsilon: 4.292342000003702
Episode 901	Average Score: 0.01
epsilon: 4.291656000003705
Episode 902	Average Score: 0.01
epsilon: 4.290970000003709
Episode 903	Average Score: 0.01
epsilon: 4.290284000003712
Episode 904	Average Score: 0.01
epsilon: 4.289598000003716
Episode 905	Average Score: 0.01
epsilon: 4.28886300000372
Episode 906	Average Score: 0.01
epsilon: 4.288177000003723
Episode 907	Average Score: 0.01
epsilon: 4.287491000003727
Episode 908	Average Score: 0.01
epsilon: 4.2868050000037305
Episode 909	Average Score: 0.01
epsilon: 4.286119000003734
Episode 910	Average Score: 0.01
epsilon: 4.285384000003738
Episode 911	Average Score: 0.01
epsilon: 4.283963000003745
Episode 912	Average Score: 0.01
epsilon: 4.282346000003754
Episode 913	Average Score: 0.01
epsilon: 4.280729000003762
Episode 914	Average Score: 0.01
epsilon: 4.27921000000377
Episode 915	Average Score: 0.01
epsilon: 4.278524000003774
Episode 916	Average Score: 0.01
epsilon: 4.2778380000037775
Episode 917	Average Score: 0.01
epsilon: 4.277152000003781
Episode 918	Average Score: 0.01
epsilon: 4.275584000003789
Episode 919	Average Score: 0.01
epsilon: 4.274898000003793
Episode 920	Average Score: 0.01
epsilon: 4.274212000003796
Episode 921	Average Score: 0.01
epsilon: 4.2734770000038
Episode 922	Average Score: 0.01
epsilon: 4.272840000003804
Episode 923	Average Score: 0.01
epsilon: 4.272105000003807
Episode 924	Average Score: 0.01
epsilon: 4.271419000003811
Episode 925	Average Score: 0.01
epsilon: 4.270733000003815
Episode 926	Average Score: 0.01
epsilon: 4.270047000003818
Episode 927	Average Score: 0.01
epsilon: 4.269361000003822
Episode 928	Average Score: 0.01
epsilon: 4.268626000003826
Episode 929	Average Score: 0.01
epsilon: 4.267989000003829
Episode 930	Average Score: 0.01
epsilon: 4.266323000003838
Episode 931	Average Score: 0.01
epsilon: 4.265637000003841
Episode 932	Average Score: 0.01
epsilon: 4.264951000003845
Episode 933	Average Score: 0.01
epsilon: 4.2642650000038484
Episode 934	Average Score: 0.01
epsilon: 4.263579000003852
Episode 935	Average Score: 0.01
epsilon: 4.262893000003856
Episode 936	Average Score: 0.01
epsilon: 4.261374000003864
Episode 937	Average Score: 0.01
epsilon: 4.260688000003867
Episode 938	Average Score: 0.01
epsilon: 4.260002000003871
Episode 939	Average Score: 0.01
epsilon: 4.259267000003875
Episode 940	Average Score: 0.01
epsilon: 4.258581000003878
Episode 941	Average Score: 0.01
epsilon: 4.257895000003882
Episode 942	Average Score: 0.01
epsilon: 4.257209000003885
Episode 943	Average Score: 0.01
epsilon: 4.256523000003889
Episode 944	Average Score: 0.01
epsilon: 4.2558370000038925
Episode 945	Average Score: 0.01
epsilon: 4.255151000003896
Episode 946	Average Score: 0.01
epsilon: 4.2544650000039
Episode 947	Average Score: 0.01
epsilon: 4.2537300000039036
Episode 948	Average Score: 0.01
epsilon: 4.253044000003907
Episode 949	Average Score: 0.01
epsilon: 4.252358000003911
Episode 950	Average Score: 0.01
epsilon: 4.251672000003914
Episode 951	Average Score: 0.01
epsilon: 4.250986000003918
Episode 952	Average Score: 0.01
epsilon: 4.250251000003922
Episode 953	Average Score: 0.01
epsilon: 4.249565000003925
Episode 954	Average Score: 0.01
epsilon: 4.248879000003929
Episode 955	Average Score: 0.01
epsilon: 4.2481930000039325
Episode 956	Average Score: 0.01
epsilon: 4.247507000003936
Episode 957	Average Score: 0.01
epsilon: 4.24682100000394
Episode 958	Average Score: 0.01
epsilon: 4.246135000003943
Episode 959	Average Score: 0.01
epsilon: 4.244616000003951
Episode 960	Average Score: 0.01
epsilon: 4.243930000003955
Episode 961	Average Score: 0.01
epsilon: 4.243244000003958
Episode 962	Average Score: 0.01
epsilon: 4.242558000003962
Episode 963	Average Score: 0.01
epsilon: 4.241823000003966
Episode 964	Average Score: 0.01
epsilon: 4.241186000003969
Episode 965	Average Score: 0.01
epsilon: 4.240451000003973
Episode 966	Average Score: 0.01
epsilon: 4.239765000003977
Episode 967	Average Score: 0.01
epsilon: 4.23907900000398
Episode 968	Average Score: 0.01
epsilon: 4.238393000003984
Episode 969	Average Score: 0.01
epsilon: 4.237707000003987
Episode 970	Average Score: 0.01
epsilon: 4.236825000003992
Episode 971	Average Score: 0.01
epsilon: 4.236139000003996
Episode 972	Average Score: 0.01
epsilon: 4.235453000003999
Episode 973	Average Score: 0.01
epsilon: 4.234767000004003
Episode 974	Average Score: 0.01
epsilon: 4.23329700000401
Episode 975	Average Score: 0.01
epsilon: 4.232611000004014
Episode 976	Average Score: 0.01
epsilon: 4.231925000004018
Episode 977	Average Score: 0.01
epsilon: 4.2311900000040215
Episode 978	Average Score: 0.01
epsilon: 4.230553000004025
Episode 979	Average Score: 0.01
epsilon: 4.229818000004029
Episode 980	Average Score: 0.01
epsilon: 4.229132000004032
Episode 981	Average Score: 0.01
epsilon: 4.227515000004041
Episode 982	Average Score: 0.01
epsilon: 4.226143000004048
Episode 983	Average Score: 0.01
epsilon: 4.2254570000040514
Episode 984	Average Score: 0.01
epsilon: 4.224771000004055
Episode 985	Average Score: 0.01
epsilon: 4.224085000004059
Episode 986	Average Score: 0.01
epsilon: 4.223399000004062
Episode 987	Average Score: 0.01
epsilon: 4.222664000004066
Episode 988	Average Score: 0.01
epsilon: 4.22197800000407
Episode 989	Average Score: 0.01
epsilon: 4.220508000004077
Episode 990	Average Score: 0.01
epsilon: 4.217519000004093
Episode 991	Average Score: 0.01
epsilon: 4.2168330000040966
Episode 992	Average Score: 0.01
epsilon: 4.215265000004105
Episode 993	Average Score: 0.01
epsilon: 4.214579000004108
Episode 994	Average Score: 0.01
epsilon: 4.2116880000041235
Episode 995	Average Score: 0.02
epsilon: 4.210218000004131
Episode 996	Average Score: 0.02
epsilon: 4.208650000004139
Episode 997	Average Score: 0.02
epsilon: 4.206102000004153
Episode 998	Average Score: 0.02
epsilon: 4.205416000004156
Episode 999	Average Score: 0.02
epsilon: 4.20473000000416
Episode 1000	Average Score: 0.02
epsilon: 4.2040440000041635
Episode 1001	Average Score: 0.02
epsilon: 4.203358000004167
Episode 1002	Average Score: 0.02
epsilon: 4.202427000004172
Episode 1003	Average Score: 0.02
epsilon: 4.2017410000041755
Episode 1004	Average Score: 0.02
epsilon: 4.201055000004179
Episode 1005	Average Score: 0.02
epsilon: 4.198311000004193
Episode 1006	Average Score: 0.02
epsilon: 4.197625000004197
Episode 1007	Average Score: 0.02
epsilon: 4.196890000004201
Episode 1008	Average Score: 0.02
epsilon: 4.1962040000042045
Episode 1009	Average Score: 0.02
epsilon: 4.194636000004213
Episode 1010	Average Score: 0.02
epsilon: 4.193117000004221
Episode 1011	Average Score: 0.02
epsilon: 4.1923820000042245
Episode 1012	Average Score: 0.02
epsilon: 4.191696000004228
Episode 1013	Average Score: 0.02
epsilon: 4.191010000004232
Episode 1014	Average Score: 0.02
epsilon: 4.190324000004235
Episode 1015	Average Score: 0.02
epsilon: 4.189638000004239
Episode 1016	Average Score: 0.02
epsilon: 4.188952000004242
Episode 1017	Average Score: 0.02
epsilon: 4.1881680000042465
Episode 1018	Average Score: 0.02
epsilon: 4.186649000004254
Episode 1019	Average Score: 0.02
epsilon: 4.185914000004258
Episode 1020	Average Score: 0.02
epsilon: 4.185228000004262
Episode 1021	Average Score: 0.02
epsilon: 4.182582000004276
Episode 1022	Average Score: 0.02
epsilon: 4.181896000004279
Episode 1023	Average Score: 0.02
epsilon: 4.181210000004283
Episode 1024	Average Score: 0.02
epsilon: 4.1805240000042865
Episode 1025	Average Score: 0.02
epsilon: 4.179740000004291
Episode 1026	Average Score: 0.02
epsilon: 4.179054000004294
Episode 1027	Average Score: 0.02
epsilon: 4.1782210000042985
Episode 1028	Average Score: 0.02
epsilon: 4.177535000004302
Episode 1029	Average Score: 0.02
epsilon: 4.176849000004306
Episode 1030	Average Score: 0.02
epsilon: 4.1761140000043095
Episode 1031	Average Score: 0.02
epsilon: 4.175428000004313
Episode 1032	Average Score: 0.02
epsilon: 4.1745950000043175
Episode 1033	Average Score: 0.02
epsilon: 4.173909000004321
Episode 1034	Average Score: 0.02
epsilon: 4.173223000004325
Episode 1035	Average Score: 0.02
epsilon: 4.1724880000043285
Episode 1036	Average Score: 0.02
epsilon: 4.171802000004332
Episode 1037	Average Score: 0.02
epsilon: 4.171116000004336
Episode 1038	Average Score: 0.02
epsilon: 4.170430000004339
Episode 1039	Average Score: 0.02
epsilon: 4.169744000004343
Episode 1040	Average Score: 0.02
epsilon: 4.1690580000043465
Episode 1041	Average Score: 0.02
epsilon: 4.16837200000435
Episode 1042	Average Score: 0.02
epsilon: 4.167686000004354
Episode 1043	Average Score: 0.02
epsilon: 4.167000000004357
Episode 1044	Average Score: 0.02
epsilon: 4.165530000004365
Episode 1045	Average Score: 0.02
epsilon: 4.16263900000438
Episode 1046	Average Score: 0.02
epsilon: 4.162002000004383
Episode 1047	Average Score: 0.02
epsilon: 4.161267000004387
Episode 1048	Average Score: 0.02
epsilon: 4.160581000004391
Episode 1049	Average Score: 0.02
epsilon: 4.159944000004394
Episode 1050	Average Score: 0.02
epsilon: 4.159209000004398
Episode 1051	Average Score: 0.02
epsilon: 4.158523000004402
Episode 1052	Average Score: 0.02
epsilon: 4.157837000004405
Episode 1053	Average Score: 0.02
epsilon: 4.156367000004413
Episode 1054	Average Score: 0.02
epsilon: 4.155681000004416
Episode 1055	Average Score: 0.02
epsilon: 4.152888000004431
Episode 1056	Average Score: 0.02
epsilon: 4.1514670000044385
Episode 1057	Average Score: 0.02
epsilon: 4.150781000004442
Episode 1058	Average Score: 0.02
epsilon: 4.150095000004446
Episode 1059	Average Score: 0.02
epsilon: 4.149409000004449
Episode 1060	Average Score: 0.02
epsilon: 4.148723000004453
Episode 1061	Average Score: 0.02
epsilon: 4.147988000004457
Episode 1062	Average Score: 0.02
epsilon: 4.14730200000446
Episode 1063	Average Score: 0.02
epsilon: 4.146616000004464
Episode 1064	Average Score: 0.02
epsilon: 4.145930000004467
Episode 1065	Average Score: 0.02
epsilon: 4.145244000004471
Episode 1066	Average Score: 0.02
epsilon: 4.144558000004475
Episode 1067	Average Score: 0.02
epsilon: 4.143774000004479
Episode 1068	Average Score: 0.02
epsilon: 4.143088000004482
Episode 1069	Average Score: 0.02
epsilon: 4.142402000004486
Episode 1070	Average Score: 0.02
epsilon: 4.1417160000044895
Episode 1071	Average Score: 0.02
epsilon: 4.141030000004493
Episode 1072	Average Score: 0.02
epsilon: 4.140295000004497
Episode 1073	Average Score: 0.02
epsilon: 4.139462000004501
Episode 1074	Average Score: 0.02
epsilon: 4.138776000004505
Episode 1075	Average Score: 0.02
epsilon: 4.138090000004508
Episode 1076	Average Score: 0.02
epsilon: 4.137355000004512
Episode 1077	Average Score: 0.02
epsilon: 4.136669000004516
Episode 1078	Average Score: 0.02
epsilon: 4.1359830000045195
Episode 1079	Average Score: 0.02
epsilon: 4.135297000004523
Episode 1080	Average Score: 0.02
epsilon: 4.134611000004527
Episode 1081	Average Score: 0.02
epsilon: 4.132798000004536
Episode 1082	Average Score: 0.02
epsilon: 4.13211200000454
Episode 1083	Average Score: 0.02
epsilon: 4.130593000004548
Episode 1084	Average Score: 0.02
epsilon: 4.129907000004551
Episode 1085	Average Score: 0.02
epsilon: 4.128388000004559
Episode 1086	Average Score: 0.02
epsilon: 4.127702000004563
Episode 1087	Average Score: 0.02
epsilon: 4.1262320000045705
Episode 1088	Average Score: 0.02
epsilon: 4.125497000004574
Episode 1089	Average Score: 0.02
epsilon: 4.124811000004578
Episode 1090	Average Score: 0.02
epsilon: 4.123194000004586
Episode 1091	Average Score: 0.02
epsilon: 4.12250800000459
Episode 1092	Average Score: 0.02
epsilon: 4.121871000004593
Episode 1093	Average Score: 0.02
epsilon: 4.121185000004597
Episode 1094	Average Score: 0.02
epsilon: 4.120450000004601
Episode 1095	Average Score: 0.02
epsilon: 4.119764000004604
Episode 1096	Average Score: 0.02
epsilon: 4.119078000004608
Episode 1097	Average Score: 0.01
epsilon: 4.1183920000046115
Episode 1098	Average Score: 0.01
epsilon: 4.117706000004615
Episode 1099	Average Score: 0.01
epsilon: 4.116971000004619
Episode 1100	Average Score: 0.01
epsilon: 4.116138000004623
Episode 1101	Average Score: 0.01
epsilon: 4.115452000004627
Episode 1102	Average Score: 0.02
epsilon: 4.114080000004634
Episode 1103	Average Score: 0.02
epsilon: 4.113394000004638
Episode 1104	Average Score: 0.02
epsilon: 4.1126590000046415
Episode 1105	Average Score: 0.02
epsilon: 4.111483000004648
Episode 1106	Average Score: 0.02
epsilon: 4.110797000004651
Episode 1107	Average Score: 0.02
epsilon: 4.110111000004655
Episode 1108	Average Score: 0.02
epsilon: 4.109425000004658
Episode 1109	Average Score: 0.01
epsilon: 4.108739000004662
Episode 1110	Average Score: 0.01
epsilon: 4.1080530000046656
Episode 1111	Average Score: 0.01
epsilon: 4.107367000004669
Episode 1112	Average Score: 0.01
epsilon: 4.106632000004673
Episode 1113	Average Score: 0.01
epsilon: 4.105946000004677
Episode 1114	Average Score: 0.01
epsilon: 4.1044270000046845
Episode 1115	Average Score: 0.01
epsilon: 4.103741000004688
Episode 1116	Average Score: 0.01
epsilon: 4.103006000004692
Episode 1117	Average Score: 0.01
epsilon: 4.1023200000046955
Episode 1118	Average Score: 0.01
epsilon: 4.101634000004699
Episode 1119	Average Score: 0.01
epsilon: 4.100948000004703
Episode 1120	Average Score: 0.01
epsilon: 4.100262000004706
Episode 1121	Average Score: 0.01
epsilon: 4.098792000004714
Episode 1122	Average Score: 0.01
epsilon: 4.098106000004718
Episode 1123	Average Score: 0.01
epsilon: 4.097420000004721
Episode 1124	Average Score: 0.01
epsilon: 4.096685000004725
Episode 1125	Average Score: 0.01
epsilon: 4.095999000004729
Episode 1126	Average Score: 0.01
epsilon: 4.095313000004732
Episode 1127	Average Score: 0.01
epsilon: 4.094627000004736
Episode 1128	Average Score: 0.01
epsilon: 4.093941000004739
Episode 1129	Average Score: 0.01
epsilon: 4.093255000004743
Episode 1130	Average Score: 0.01
epsilon: 4.0925690000047465
Episode 1131	Average Score: 0.01
epsilon: 4.0910500000047545
Episode 1132	Average Score: 0.01
epsilon: 4.090364000004758
Episode 1133	Average Score: 0.01
epsilon: 4.089678000004762
Episode 1134	Average Score: 0.01
epsilon: 4.088992000004765
Episode 1135	Average Score: 0.01
epsilon: 4.088306000004769
Episode 1136	Average Score: 0.01
epsilon: 4.087620000004772
Episode 1137	Average Score: 0.01
epsilon: 4.086934000004776
Episode 1138	Average Score: 0.01
epsilon: 4.08619900000478
Episode 1139	Average Score: 0.01
epsilon: 4.085562000004783
Episode 1140	Average Score: 0.01
epsilon: 4.084827000004787
Episode 1141	Average Score: 0.01
epsilon: 4.084141000004791
Episode 1142	Average Score: 0.02
epsilon: 4.082475000004799
Episode 1143	Average Score: 0.02
epsilon: 4.081789000004803
Episode 1144	Average Score: 0.01
epsilon: 4.079633000004814
Episode 1145	Average Score: 0.01
epsilon: 4.078065000004822
Episode 1146	Average Score: 0.01
epsilon: 4.077379000004826
Episode 1147	Average Score: 0.01
epsilon: 4.07669300000483
Episode 1148	Average Score: 0.01
epsilon: 4.076007000004833
Episode 1149	Average Score: 0.01
epsilon: 4.075272000004837
Episode 1150	Average Score: 0.01
epsilon: 4.074586000004841
Episode 1151	Average Score: 0.02
epsilon: 4.073018000004849
Episode 1152	Average Score: 0.02
epsilon: 4.072332000004852
Episode 1153	Average Score: 0.01
epsilon: 4.071646000004856
Episode 1154	Average Score: 0.01
epsilon: 4.07096000000486
Episode 1155	Average Score: 0.01
epsilon: 4.070274000004863
Episode 1156	Average Score: 0.01
epsilon: 4.069539000004867
Episode 1157	Average Score: 0.01
epsilon: 4.068853000004871
Episode 1158	Average Score: 0.01
epsilon: 4.068167000004874
Episode 1159	Average Score: 0.01
epsilon: 4.067481000004878
Episode 1160	Average Score: 0.01
epsilon: 4.066795000004881
Episode 1161	Average Score: 0.01
epsilon: 4.066060000004885
Episode 1162	Average Score: 0.01
epsilon: 4.065374000004889
Episode 1163	Average Score: 0.01
epsilon: 4.064688000004892
Episode 1164	Average Score: 0.01
epsilon: 4.064002000004896
Episode 1165	Average Score: 0.01
epsilon: 4.062483000004904
Episode 1166	Average Score: 0.01
epsilon: 4.060964000004912
Episode 1167	Average Score: 0.01
epsilon: 4.0602780000049155
Episode 1168	Average Score: 0.01
epsilon: 4.059543000004919
Episode 1169	Average Score: 0.01
epsilon: 4.058857000004923
Episode 1170	Average Score: 0.01
epsilon: 4.0581710000049265
Episode 1171	Average Score: 0.01
epsilon: 4.05748500000493
Episode 1172	Average Score: 0.01
epsilon: 4.056799000004934
Episode 1173	Average Score: 0.01
epsilon: 4.0560640000049375
Episode 1174	Average Score: 0.01
epsilon: 4.055427000004941
Episode 1175	Average Score: 0.01
epsilon: 4.054692000004945
Episode 1176	Average Score: 0.01
epsilon: 4.054006000004948
Episode 1177	Average Score: 0.01
epsilon: 4.053369000004952
Episode 1178	Average Score: 0.01
epsilon: 4.052634000004955
Episode 1179	Average Score: 0.01
epsilon: 4.051948000004959
Episode 1180	Average Score: 0.01
epsilon: 4.051262000004963
Episode 1181	Average Score: 0.01
epsilon: 4.050576000004966
Episode 1182	Average Score: 0.01
epsilon: 4.04989000000497
Episode 1183	Average Score: 0.01
epsilon: 4.049204000004973
Episode 1184	Average Score: 0.01
epsilon: 4.048518000004977
Episode 1185	Average Score: 0.01
epsilon: 4.046362000004988
Episode 1186	Average Score: 0.01
epsilon: 4.043667000005002
Episode 1187	Average Score: 0.01
epsilon: 4.042981000005006
Episode 1188	Average Score: 0.01
epsilon: 4.0422950000050095
Episode 1189	Average Score: 0.01
epsilon: 4.041609000005013
Episode 1190	Average Score: 0.01
epsilon: 4.040090000005021
Episode 1191	Average Score: 0.01
epsilon: 4.039404000005025
Episode 1192	Average Score: 0.01
epsilon: 4.037836000005033
Episode 1193	Average Score: 0.01
epsilon: 4.037101000005037
Episode 1194	Average Score: 0.01
epsilon: 4.03641500000504
Episode 1195	Average Score: 0.01
epsilon: 4.035729000005044
Episode 1196	Average Score: 0.01
epsilon: 4.0350430000050475
Episode 1197	Average Score: 0.01
epsilon: 4.0342590000050516
Episode 1198	Average Score: 0.01
epsilon: 4.033573000005055
Episode 1199	Average Score: 0.01
epsilon: 4.0329360000050585
Episode 1200	Average Score: 0.01
epsilon: 4.032201000005062
Episode 1201	Average Score: 0.01
epsilon: 4.031564000005066
Episode 1202	Average Score: 0.01
epsilon: 4.0308290000050695
Episode 1203	Average Score: 0.01
epsilon: 4.030143000005073
Episode 1204	Average Score: 0.01
epsilon: 4.029457000005077
Episode 1205	Average Score: 0.01
epsilon: 4.0287220000050805
Episode 1206	Average Score: 0.01
epsilon: 4.028036000005084
Episode 1207	Average Score: 0.01
epsilon: 4.027350000005088
Episode 1208	Average Score: 0.01
epsilon: 4.026664000005091
Episode 1209	Average Score: 0.01
epsilon: 4.025978000005095
Episode 1210	Average Score: 0.01
epsilon: 4.0252920000050985
Episode 1211	Average Score: 0.01
epsilon: 4.023675000005107
Episode 1212	Average Score: 0.01
epsilon: 4.022940000005111
Episode 1213	Average Score: 0.01
epsilon: 4.022205000005115
Episode 1214	Average Score: 0.01
epsilon: 4.021519000005118
Episode 1215	Average Score: 0.01
epsilon: 4.020833000005122
Episode 1216	Average Score: 0.01
epsilon: 4.020147000005125
Episode 1217	Average Score: 0.01
epsilon: 4.019461000005129
Episode 1218	Average Score: 0.01
epsilon: 4.0187750000051325
Episode 1219	Average Score: 0.01
epsilon: 4.018089000005136
Episode 1220	Average Score: 0.01
epsilon: 4.01740300000514
Episode 1221	Average Score: 0.01
epsilon: 4.016668000005144
Episode 1222	Average Score: 0.01
epsilon: 4.015933000005147
Episode 1223	Average Score: 0.01
epsilon: 4.015247000005151
Episode 1224	Average Score: 0.01
epsilon: 4.014561000005155
Episode 1225	Average Score: 0.01
epsilon: 4.013826000005158
Episode 1226	Average Score: 0.01
epsilon: 4.013140000005162
Episode 1227	Average Score: 0.01
epsilon: 4.012405000005166
Episode 1228	Average Score: 0.01
epsilon: 4.0117190000051695
Episode 1229	Average Score: 0.01
epsilon: 4.011033000005173
Episode 1230	Average Score: 0.01
epsilon: 4.010347000005177
Episode 1231	Average Score: 0.01
epsilon: 4.00966100000518
Episode 1232	Average Score: 0.01
epsilon: 4.008926000005184
Episode 1233	Average Score: 0.01
epsilon: 4.0058390000052
Episode 1234	Average Score: 0.01
epsilon: 4.005153000005204
Episode 1235	Average Score: 0.01
epsilon: 4.004467000005207
Episode 1236	Average Score: 0.01
epsilon: 4.0036830000052115
Episode 1237	Average Score: 0.01
epsilon: 4.002997000005215
Episode 1238	Average Score: 0.01
epsilon: 4.002311000005219
Episode 1239	Average Score: 0.01
epsilon: 4.001625000005222
Episode 1240	Average Score: 0.01
epsilon: 4.000939000005226
Episode 1241	Average Score: 0.01
epsilon: 4.00020400000523
Episode 1242	Average Score: 0.01
epsilon: 3.999518000005229
Episode 1243	Average Score: 0.01
epsilon: 3.998832000005226
Episode 1244	Average Score: 0.01
epsilon: 3.9981460000052236
Episode 1245	Average Score: 0.01
epsilon: 3.997460000005221
Episode 1246	Average Score: 0.01
epsilon: 3.996725000005218
Episode 1247	Average Score: 0.01
epsilon: 3.9960390000052155
Episode 1248	Average Score: 0.01
epsilon: 3.995353000005213
Episode 1249	Average Score: 0.01
epsilon: 3.9946670000052102
Episode 1250	Average Score: 0.01
epsilon: 3.9930990000052042
Episode 1251	Average Score: 0.01
epsilon: 3.992462000005202
Episode 1252	Average Score: 0.01
epsilon: 3.991727000005199
Episode 1253	Average Score: 0.01
epsilon: 3.9903060000051935
Episode 1254	Average Score: 0.01
epsilon: 3.9887380000051875
Episode 1255	Average Score: 0.01
epsilon: 3.988052000005185
Episode 1256	Average Score: 0.01
epsilon: 3.9873660000051823
Episode 1257	Average Score: 0.01
epsilon: 3.9866800000051796
Episode 1258	Average Score: 0.01
epsilon: 3.985994000005177
Episode 1259	Average Score: 0.01
epsilon: 3.985259000005174
Episode 1260	Average Score: 0.01
epsilon: 3.9845730000051716
Episode 1261	Average Score: 0.01
epsilon: 3.983887000005169
Episode 1262	Average Score: 0.01
epsilon: 3.9832010000051663
Episode 1263	Average Score: 0.01
epsilon: 3.9825150000051637
Episode 1264	Average Score: 0.01
epsilon: 3.981780000005161
Episode 1265	Average Score: 0.01
epsilon: 3.9810940000051582
Episode 1266	Average Score: 0.01
epsilon: 3.9804080000051556
Episode 1267	Average Score: 0.01
epsilon: 3.979722000005153
Episode 1268	Average Score: 0.01
epsilon: 3.9790360000051503
Episode 1269	Average Score: 0.01
epsilon: 3.9783010000051475
Episode 1270	Average Score: 0.01
epsilon: 3.977615000005145
Episode 1271	Average Score: 0.01
epsilon: 3.9769290000051423
Episode 1272	Average Score: 0.01
epsilon: 3.9754590000051366
Episode 1273	Average Score: 0.01
epsilon: 3.974773000005134
Episode 1274	Average Score: 0.01
epsilon: 3.974038000005131
Episode 1275	Average Score: 0.01
epsilon: 3.9733030000051284
Episode 1276	Average Score: 0.01
epsilon: 3.9726170000051257
Episode 1277	Average Score: 0.01
epsilon: 3.971931000005123
Episode 1278	Average Score: 0.01
epsilon: 3.9712450000051205
Episode 1279	Average Score: 0.01
epsilon: 3.970559000005118
Episode 1280	Average Score: 0.01
epsilon: 3.9698730000051152
Episode 1281	Average Score: 0.01
epsilon: 3.9691870000051126
Episode 1282	Average Score: 0.01
epsilon: 3.9684030000051096
Episode 1283	Average Score: 0.01
epsilon: 3.967717000005107
Episode 1284	Average Score: 0.01
epsilon: 3.9670800000051045
Episode 1285	Average Score: 0.01
epsilon: 3.966394000005102
Episode 1286	Average Score: 0.01
epsilon: 3.9657080000050993
Episode 1287	Average Score: 0.01
epsilon: 3.9650220000050966
Episode 1288	Average Score: 0.01
epsilon: 3.9638950000050923
Episode 1289	Average Score: 0.01
epsilon: 3.96325800000509
Episode 1290	Average Score: 0.01
epsilon: 3.9625720000050872
Episode 1291	Average Score: 0.01
epsilon: 3.9618370000050844
Episode 1292	Average Score: 0.01
epsilon: 3.960416000005079
Episode 1293	Average Score: 0.01
epsilon: 3.9597300000050764
Episode 1294	Average Score: 0.01
epsilon: 3.95811300000507
Episode 1295	Average Score: 0.01
epsilon: 3.9573780000050673
Episode 1296	Average Score: 0.01
epsilon: 3.9566920000050647
Episode 1297	Average Score: 0.01
epsilon: 3.956006000005062
Episode 1298	Average Score: 0.01
epsilon: 3.9553200000050595
Episode 1299	Average Score: 0.01
epsilon: 3.954634000005057
Episode 1300	Average Score: 0.01
epsilon: 3.953948000005054
Episode 1301	Average Score: 0.01
epsilon: 3.9532620000050516
Episode 1302	Average Score: 0.01
epsilon: 3.9525270000050488
Episode 1303	Average Score: 0.01
epsilon: 3.9518900000050463
Episode 1304	Average Score: 0.01
epsilon: 3.9511550000050435
Episode 1305	Average Score: 0.01
epsilon: 3.9492930000050364
Episode 1306	Average Score: 0.01
epsilon: 3.9486070000050337
Episode 1307	Average Score: 0.01
epsilon: 3.947921000005031
Episode 1308	Average Score: 0.01
epsilon: 3.946353000005025
Episode 1309	Average Score: 0.01
epsilon: 3.9415510000050067
Episode 1310	Average Score: 0.01
epsilon: 3.940865000005004
Episode 1311	Average Score: 0.01
epsilon: 3.9402280000050016
Episode 1312	Average Score: 0.01
epsilon: 3.939493000004999
Episode 1313	Average Score: 0.01
epsilon: 3.938807000004996
Episode 1314	Average Score: 0.01
epsilon: 3.9381700000049937
Episode 1315	Average Score: 0.01
epsilon: 3.937435000004991
Episode 1316	Average Score: 0.01
epsilon: 3.936700000004988
Episode 1317	Average Score: 0.01
epsilon: 3.9360140000049855
Episode 1318	Average Score: 0.01
epsilon: 3.935328000004983
Episode 1319	Average Score: 0.01
epsilon: 3.93464200000498
Episode 1320	Average Score: 0.01
epsilon: 3.9339560000049776
Episode 1321	Average Score: 0.01
epsilon: 3.9332210000049748
Episode 1322	Average Score: 0.01
epsilon: 3.9325840000049723
Episode 1323	Average Score: 0.01
epsilon: 3.9318490000049695
Episode 1324	Average Score: 0.01
epsilon: 3.931163000004967
Episode 1325	Average Score: 0.01
epsilon: 3.9296930000049612
Episode 1326	Average Score: 0.01
epsilon: 3.9290070000049586
Episode 1327	Average Score: 0.01
epsilon: 3.928321000004956
Episode 1328	Average Score: 0.01
epsilon: 3.9276350000049534
Episode 1329	Average Score: 0.01
epsilon: 3.9269000000049505
Episode 1330	Average Score: 0.01
epsilon: 3.9261160000049475
Episode 1331	Average Score: 0.01
epsilon: 3.9253320000049445
Episode 1332	Average Score: 0.01
epsilon: 3.924646000004942
Episode 1333	Average Score: 0.01
epsilon: 3.923911000004939
Episode 1334	Average Score: 0.01
epsilon: 3.9232250000049365
Episode 1335	Average Score: 0.01
epsilon: 3.9216570000049304
Episode 1336	Average Score: 0.01
epsilon: 3.920971000004928
Episode 1337	Average Score: 0.01
epsilon: 3.919452000004922
Episode 1338	Average Score: 0.01
epsilon: 3.917884000004916
Episode 1339	Average Score: 0.01
epsilon: 3.91631600000491
Episode 1340	Average Score: 0.01
epsilon: 3.9156300000049074
Episode 1341	Average Score: 0.02
epsilon: 3.9140620000049013
Episode 1342	Average Score: 0.02
epsilon: 3.9133760000048987
Episode 1343	Average Score: 0.02
epsilon: 3.912690000004896
Episode 1344	Average Score: 0.02
epsilon: 3.9111710000048903
Episode 1345	Average Score: 0.02
epsilon: 3.9104850000048876
Episode 1346	Average Score: 0.02
epsilon: 3.909799000004885
Episode 1347	Average Score: 0.02
epsilon: 3.9091130000048824
Episode 1348	Average Score: 0.02
epsilon: 3.9084270000048797
Episode 1349	Average Score: 0.02
epsilon: 3.907692000004877
Episode 1350	Average Score: 0.02
epsilon: 3.9070060000048743
Episode 1351	Average Score: 0.02
epsilon: 3.9045070000048647
Episode 1352	Average Score: 0.02
epsilon: 3.903821000004862
Episode 1353	Average Score: 0.02
epsilon: 3.9002930000048486
Episode 1354	Average Score: 0.02
epsilon: 3.8995580000048458
Episode 1355	Average Score: 0.02
epsilon: 3.898872000004843
Episode 1356	Average Score: 0.02
epsilon: 3.8981860000048405
Episode 1357	Average Score: 0.02
epsilon: 3.897500000004838
Episode 1358	Average Score: 0.02
epsilon: 3.8968140000048352
Episode 1359	Average Score: 0.02
epsilon: 3.8960790000048324
Episode 1360	Average Score: 0.02
epsilon: 3.89539300000483
Episode 1361	Average Score: 0.02
epsilon: 3.894658000004827
Episode 1362	Average Score: 0.02
epsilon: 3.893139000004821
Episode 1363	Average Score: 0.02
epsilon: 3.890493000004811
Episode 1364	Average Score: 0.02
epsilon: 3.889709000004808
Episode 1365	Average Score: 0.02
epsilon: 3.8890230000048054
Episode 1366	Average Score: 0.02
epsilon: 3.8883370000048028
Episode 1367	Average Score: 0.02
epsilon: 3.8866710000047964
Episode 1368	Average Score: 0.02
epsilon: 3.8859850000047937
Episode 1369	Average Score: 0.02
epsilon: 3.885299000004791
Episode 1370	Average Score: 0.02
epsilon: 3.8846130000047885
Episode 1371	Average Score: 0.02
epsilon: 3.883192000004783
Episode 1372	Average Score: 0.02
epsilon: 3.8825060000047804
Episode 1373	Average Score: 0.02
epsilon: 3.8817710000047776
Episode 1374	Average Score: 0.02
epsilon: 3.881134000004775
Episode 1375	Average Score: 0.02
epsilon: 3.8803990000047723
Episode 1376	Average Score: 0.02
epsilon: 3.8797130000047697
Episode 1377	Average Score: 0.02
epsilon: 3.879027000004767
Episode 1378	Average Score: 0.02
epsilon: 3.8764790000047573
Episode 1379	Average Score: 0.02
epsilon: 3.8757930000047547
Episode 1380	Average Score: 0.02
epsilon: 3.875107000004752
Episode 1381	Average Score: 0.02
epsilon: 3.8744210000047494
Episode 1382	Average Score: 0.02
epsilon: 3.873735000004747
Episode 1383	Average Score: 0.02
epsilon: 3.873000000004744
Episode 1384	Average Score: 0.02
epsilon: 3.8723140000047414
Episode 1385	Average Score: 0.02
epsilon: 3.8716280000047387
Episode 1386	Average Score: 0.02
epsilon: 3.870942000004736
Episode 1387	Average Score: 0.02
epsilon: 3.8702560000047335
Episode 1388	Average Score: 0.02
epsilon: 3.8695210000047306
Episode 1389	Average Score: 0.02
epsilon: 3.868051000004725
Episode 1390	Average Score: 0.02
epsilon: 3.867316000004722
Episode 1391	Average Score: 0.02
epsilon: 3.8666300000047196
Episode 1392	Average Score: 0.02
epsilon: 3.8636900000047083
Episode 1393	Average Score: 0.02
epsilon: 3.8629550000047055
Episode 1394	Average Score: 0.02
epsilon: 3.862318000004703
Episode 1395	Average Score: 0.02
epsilon: 3.8615830000047002
Episode 1396	Average Score: 0.02
epsilon: 3.860162000004695
Episode 1397	Average Score: 0.02
epsilon: 3.859427000004692
Episode 1398	Average Score: 0.02
epsilon: 3.8587410000046893
Episode 1399	Average Score: 0.02
epsilon: 3.8572220000046835
Episode 1400	Average Score: 0.02
epsilon: 3.8557030000046777
Episode 1401	Average Score: 0.02
epsilon: 3.8541350000046717
Episode 1402	Average Score: 0.02
epsilon: 3.853449000004669
Episode 1403	Average Score: 0.02
epsilon: 3.8527140000046662
Episode 1404	Average Score: 0.02
epsilon: 3.8520280000046636
Episode 1405	Average Score: 0.02
epsilon: 3.850509000004658
Episode 1406	Average Score: 0.02
epsilon: 3.849823000004655
Episode 1407	Average Score: 0.02
epsilon: 3.8491370000046525
Episode 1408	Average Score: 0.02
epsilon: 3.8483530000046495
Episode 1409	Average Score: 0.02
epsilon: 3.847667000004647
Episode 1410	Average Score: 0.02
epsilon: 3.8469810000046443
Episode 1411	Average Score: 0.02
epsilon: 3.8462950000046416
Episode 1412	Average Score: 0.02
epsilon: 3.845609000004639
Episode 1413	Average Score: 0.02
epsilon: 3.843992000004633
Episode 1414	Average Score: 0.02
epsilon: 3.84330600000463
Episode 1415	Average Score: 0.02
epsilon: 3.8426200000046276
Episode 1416	Average Score: 0.02
epsilon: 3.841934000004625
Episode 1417	Average Score: 0.02
epsilon: 3.8412480000046223
Episode 1418	Average Score: 0.02
epsilon: 3.84061100000462
Episode 1419	Average Score: 0.02
epsilon: 3.839876000004617
Episode 1420	Average Score: 0.02
epsilon: 3.8391900000046144
Episode 1421	Average Score: 0.02
epsilon: 3.8384550000046116
Episode 1422	Average Score: 0.02
epsilon: 3.837769000004609
Episode 1423	Average Score: 0.02
epsilon: 3.836985000004606
Episode 1424	Average Score: 0.02
epsilon: 3.8362990000046033
Episode 1425	Average Score: 0.02
epsilon: 3.8355640000046005
Episode 1426	Average Score: 0.02
epsilon: 3.834878000004598
Episode 1427	Average Score: 0.02
epsilon: 3.8341920000045953
Episode 1428	Average Score: 0.02
epsilon: 3.8335060000045926
Episode 1429	Average Score: 0.02
epsilon: 3.83282000000459
Episode 1430	Average Score: 0.02
epsilon: 3.8321340000045874
Episode 1431	Average Score: 0.02
epsilon: 3.8314480000045847
Episode 1432	Average Score: 0.02
epsilon: 3.830762000004582
Episode 1433	Average Score: 0.02
epsilon: 3.8300760000045795
Episode 1434	Average Score: 0.02
epsilon: 3.829390000004577
Episode 1435	Average Score: 0.02
epsilon: 3.828655000004574
Episode 1436	Average Score: 0.02
epsilon: 3.8279690000045714
Episode 1437	Average Score: 0.02
epsilon: 3.827283000004569
Episode 1438	Average Score: 0.02
epsilon: 3.824784000004559
Episode 1439	Average Score: 0.02
epsilon: 3.8240980000045566
Episode 1440	Average Score: 0.02
epsilon: 3.8233140000045536
Episode 1441	Average Score: 0.02
epsilon: 3.822628000004551
Episode 1442	Average Score: 0.02
epsilon: 3.8219420000045483
Episode 1443	Average Score: 0.02
epsilon: 3.8212560000045457
Episode 1444	Average Score: 0.02
epsilon: 3.820521000004543
Episode 1445	Average Score: 0.02
epsilon: 3.819002000004537
Episode 1446	Average Score: 0.02
epsilon: 3.8183160000045344
Episode 1447	Average Score: 0.02
epsilon: 3.8175810000045316
Episode 1448	Average Score: 0.02
epsilon: 3.816944000004529
Episode 1449	Average Score: 0.02
epsilon: 3.8154250000045233
Episode 1450	Average Score: 0.02
epsilon: 3.8147390000045207
Episode 1451	Average Score: 0.02
epsilon: 3.814053000004518
Episode 1452	Average Score: 0.02
epsilon: 3.8133670000045155
Episode 1453	Average Score: 0.02
epsilon: 3.812681000004513
Episode 1454	Average Score: 0.02
epsilon: 3.81194600000451
Episode 1455	Average Score: 0.02
epsilon: 3.8112600000045074
Episode 1456	Average Score: 0.02
epsilon: 3.8105740000045047
Episode 1457	Average Score: 0.02
epsilon: 3.809888000004502
Episode 1458	Average Score: 0.02
epsilon: 3.8092020000044995
Episode 1459	Average Score: 0.02
epsilon: 3.8084670000044967
Episode 1460	Average Score: 0.02
epsilon: 3.807781000004494
Episode 1461	Average Score: 0.02
epsilon: 3.8070950000044914
Episode 1462	Average Score: 0.01
epsilon: 3.806409000004489
Episode 1463	Average Score: 0.01
epsilon: 3.805723000004486
Episode 1464	Average Score: 0.01
epsilon: 3.8050370000044835
Episode 1465	Average Score: 0.01
epsilon: 3.804351000004481
Episode 1466	Average Score: 0.01
epsilon: 3.803616000004478
Episode 1467	Average Score: 0.01
epsilon: 3.8029300000044755
Episode 1468	Average Score: 0.01
epsilon: 3.802244000004473
Episode 1469	Average Score: 0.01
epsilon: 3.800774000004467
Episode 1470	Average Score: 0.01
epsilon: 3.8001370000044647
Episode 1471	Average Score: 0.01
epsilon: 3.799451000004462
Episode 1472	Average Score: 0.01
epsilon: 3.7987650000044595
Episode 1473	Average Score: 0.01
epsilon: 3.798079000004457
Episode 1474	Average Score: 0.01
epsilon: 3.796560000004451
Episode 1475	Average Score: 0.01
epsilon: 3.7958740000044484
Episode 1476	Average Score: 0.01
epsilon: 3.795188000004446
Episode 1477	Average Score: 0.01
epsilon: 3.794502000004443
Episode 1478	Average Score: 0.01
epsilon: 3.7938160000044405
Episode 1479	Average Score: 0.01
epsilon: 3.7930810000044377
Episode 1480	Average Score: 0.01
epsilon: 3.792395000004435
Episode 1481	Average Score: 0.01
epsilon: 3.7917580000044326
Episode 1482	Average Score: 0.01
epsilon: 3.79102300000443
Episode 1483	Average Score: 0.01
epsilon: 3.790337000004427
Episode 1484	Average Score: 0.01
epsilon: 3.7881320000044187
Episode 1485	Average Score: 0.01
epsilon: 3.787446000004416
Episode 1486	Average Score: 0.01
epsilon: 3.7867600000044135
Episode 1487	Average Score: 0.01
epsilon: 3.785290000004408
Episode 1488	Average Score: 0.01
epsilon: 3.784555000004405
Episode 1489	Average Score: 0.01
epsilon: 3.7839180000044026
Episode 1490	Average Score: 0.01
epsilon: 3.7831830000043998
Episode 1491	Average Score: 0.01
epsilon: 3.782497000004397
Episode 1492	Average Score: 0.01
epsilon: 3.7809780000043913
Episode 1493	Average Score: 0.01
epsilon: 3.7802920000043887
Episode 1494	Average Score: 0.01
epsilon: 3.778773000004383
Episode 1495	Average Score: 0.01
epsilon: 3.77803800000438
Episode 1496	Average Score: 0.01
epsilon: 3.7774010000043776
Episode 1497	Average Score: 0.01
epsilon: 3.776666000004375
Episode 1498	Average Score: 0.01
epsilon: 3.775980000004372
Episode 1499	Average Score: 0.01
epsilon: 3.7752940000043695
Episode 1500	Average Score: 0.01
epsilon: 3.774608000004367
Episode 1501	Average Score: 0.01
epsilon: 3.7739220000043643
Episode 1502	Average Score: 0.01
epsilon: 3.7732360000043617
Episode 1503	Average Score: 0.01
epsilon: 3.772550000004359
Episode 1504	Average Score: 0.01
epsilon: 3.7718640000043564
Episode 1505	Average Score: 0.01
epsilon: 3.7711780000043538
Episode 1506	Average Score: 0.01
epsilon: 3.7693160000043466
Episode 1507	Average Score: 0.01
epsilon: 3.768630000004344
Episode 1508	Average Score: 0.01
epsilon: 3.7679440000043414
Episode 1509	Average Score: 0.01
epsilon: 3.7672580000043387
Episode 1510	Average Score: 0.01
epsilon: 3.766572000004336
Episode 1511	Average Score: 0.01
epsilon: 3.7658370000043333
Episode 1512	Average Score: 0.01
epsilon: 3.7651510000043307
Episode 1513	Average Score: 0.01
epsilon: 3.764465000004328
Episode 1514	Average Score: 0.01
epsilon: 3.7637790000043254
Episode 1515	Average Score: 0.01
epsilon: 3.763093000004323
Episode 1516	Average Score: 0.01
epsilon: 3.76240700000432
Episode 1517	Average Score: 0.01
epsilon: 3.7616720000043173
Episode 1518	Average Score: 0.01
epsilon: 3.7609860000043147
Episode 1519	Average Score: 0.01
epsilon: 3.7602020000043117
Episode 1520	Average Score: 0.01
epsilon: 3.7586340000043057
Episode 1521	Average Score: 0.01
epsilon: 3.757948000004303
Episode 1522	Average Score: 0.01
epsilon: 3.7572620000043004
Episode 1523	Average Score: 0.01
epsilon: 3.756576000004298
Episode 1524	Average Score: 0.01
epsilon: 3.755057000004292
Episode 1525	Average Score: 0.01
epsilon: 3.754322000004289
Episode 1526	Average Score: 0.01
epsilon: 3.7536360000042865
Episode 1527	Average Score: 0.01
epsilon: 3.752950000004284
Episode 1528	Average Score: 0.01
epsilon: 3.7522640000042813
Episode 1529	Average Score: 0.01
epsilon: 3.7515780000042787
Episode 1530	Average Score: 0.01
epsilon: 3.7507450000042755
Episode 1531	Average Score: 0.01
epsilon: 3.750059000004273
Episode 1532	Average Score: 0.01
epsilon: 3.74937300000427
Episode 1533	Average Score: 0.01
epsilon: 3.7486870000042676
Episode 1534	Average Score: 0.01
epsilon: 3.7479520000042648
Episode 1535	Average Score: 0.01
epsilon: 3.7465310000042593
Episode 1536	Average Score: 0.01
epsilon: 3.7458450000042567
Episode 1537	Average Score: 0.01
epsilon: 3.745159000004254
Episode 1538	Average Score: 0.01
epsilon: 3.7444730000042514
Episode 1539	Average Score: 0.01
epsilon: 3.7437380000042486
Episode 1540	Average Score: 0.01
epsilon: 3.743101000004246
Episode 1541	Average Score: 0.01
epsilon: 3.7424150000042435
Episode 1542	Average Score: 0.01
epsilon: 3.7413860000042396
Episode 1543	Average Score: 0.01
epsilon: 3.7398670000042338
Episode 1544	Average Score: 0.01
epsilon: 3.739132000004231
Episode 1545	Average Score: 0.01
epsilon: 3.7384460000042283
Episode 1546	Average Score: 0.01
epsilon: 3.7377600000042257
Episode 1547	Average Score: 0.01
epsilon: 3.7371230000042233
Episode 1548	Average Score: 0.01
epsilon: 3.735506000004217
Episode 1549	Average Score: 0.01
epsilon: 3.7339870000042112
Episode 1550	Average Score: 0.01
epsilon: 3.7324680000042054
Episode 1551	Average Score: 0.01
epsilon: 3.731782000004203
Episode 1552	Average Score: 0.01
epsilon: 3.7310960000042
Episode 1553	Average Score: 0.01
epsilon: 3.7304100000041975
Episode 1554	Average Score: 0.01
epsilon: 3.729724000004195
Episode 1555	Average Score: 0.01
epsilon: 3.7290380000041923
Episode 1556	Average Score: 0.01
epsilon: 3.7283520000041896
Episode 1557	Average Score: 0.01
epsilon: 3.727666000004187
Episode 1558	Average Score: 0.01
epsilon: 3.726882000004184
Episode 1559	Average Score: 0.01
epsilon: 3.726147000004181
Episode 1560	Average Score: 0.01
epsilon: 3.7254610000041786
Episode 1561	Average Score: 0.01
epsilon: 3.724824000004176
Episode 1562	Average Score: 0.01
epsilon: 3.7240890000041733
Episode 1563	Average Score: 0.01
epsilon: 3.7234030000041707
Episode 1564	Average Score: 0.01
epsilon: 3.721933000004165
Episode 1565	Average Score: 0.01
epsilon: 3.7212470000041624
Episode 1566	Average Score: 0.01
epsilon: 3.72056100000416
Episode 1567	Average Score: 0.01
epsilon: 3.719875000004157
Episode 1568	Average Score: 0.01
epsilon: 3.717180000004147
Episode 1569	Average Score: 0.01
epsilon: 3.716494000004144
Episode 1570	Average Score: 0.01
epsilon: 3.7158080000041416
Episode 1571	Average Score: 0.01
epsilon: 3.7150730000041388
Episode 1572	Average Score: 0.01
epsilon: 3.714387000004136
Episode 1573	Average Score: 0.01
epsilon: 3.7137010000041335
Episode 1574	Average Score: 0.01
epsilon: 3.7129170000041305
Episode 1575	Average Score: 0.01
epsilon: 3.712231000004128
Episode 1576	Average Score: 0.01
epsilon: 3.7115450000041252
Episode 1577	Average Score: 0.01
epsilon: 3.7108100000041224
Episode 1578	Average Score: 0.01
epsilon: 3.71012400000412
Episode 1579	Average Score: 0.01
epsilon: 3.709438000004117
Episode 1580	Average Score: 0.01
epsilon: 3.7068900000041074
Episode 1581	Average Score: 0.01
epsilon: 3.7062040000041048
Episode 1582	Average Score: 0.01
epsilon: 3.705469000004102
Episode 1583	Average Score: 0.01
epsilon: 3.7047830000040993
Episode 1584	Average Score: 0.01
epsilon: 3.7040970000040967
Episode 1585	Average Score: 0.01
epsilon: 3.703411000004094
Episode 1586	Average Score: 0.01
epsilon: 3.7027250000040914
Episode 1587	Average Score: 0.01
epsilon: 3.702039000004089
Episode 1588	Average Score: 0.01
epsilon: 3.701353000004086
Episode 1589	Average Score: 0.01
epsilon: 3.7006180000040834
Episode 1590	Average Score: 0.01
epsilon: 3.699981000004081
Episode 1591	Average Score: 0.01
epsilon: 3.6983640000040747
Episode 1592	Average Score: 0.01
epsilon: 3.697678000004072
Episode 1593	Average Score: 0.01
epsilon: 3.6969920000040695
Episode 1594	Average Score: 0.01
epsilon: 3.696306000004067
Episode 1595	Average Score: 0.01
epsilon: 3.695620000004064
Episode 1596	Average Score: 0.01
epsilon: 3.6949340000040616
Episode 1597	Average Score: 0.01
epsilon: 3.694248000004059
Episode 1598	Average Score: 0.01
epsilon: 3.6927780000040533
Episode 1599	Average Score: 0.01
epsilon: 3.6920920000040507
Episode 1600	Average Score: 0.01
epsilon: 3.6912590000040475
Episode 1601	Average Score: 0.01
epsilon: 3.6897400000040417
Episode 1602	Average Score: 0.01
epsilon: 3.689054000004039
Episode 1603	Average Score: 0.01
epsilon: 3.6883190000040362
Episode 1604	Average Score: 0.01
epsilon: 3.6876330000040336
Episode 1605	Average Score: 0.01
epsilon: 3.686947000004031
Episode 1606	Average Score: 0.01
epsilon: 3.685379000004025
Episode 1607	Average Score: 0.01
epsilon: 3.6846930000040223
Episode 1608	Average Score: 0.01
epsilon: 3.6840070000040197
Episode 1609	Average Score: 0.01
epsilon: 3.683321000004017
Episode 1610	Average Score: 0.01
epsilon: 3.6826350000040144
Episode 1611	Average Score: 0.01
epsilon: 3.6819000000040116
Episode 1612	Average Score: 0.01
epsilon: 3.681214000004009
Episode 1613	Average Score: 0.01
epsilon: 3.680479000004006
Episode 1614	Average Score: 0.01
epsilon: 3.6797930000040036
Episode 1615	Average Score: 0.01
epsilon: 3.679156000004001
Episode 1616	Average Score: 0.01
epsilon: 3.6784210000039983
Episode 1617	Average Score: 0.01
epsilon: 3.6777350000039957
Episode 1618	Average Score: 0.01
epsilon: 3.677049000003993
Episode 1619	Average Score: 0.01
epsilon: 3.6763630000039904
Episode 1620	Average Score: 0.01
epsilon: 3.675677000003988
Episode 1621	Average Score: 0.01
epsilon: 3.674942000003985
Episode 1622	Average Score: 0.01
epsilon: 3.674207000003982
Episode 1623	Average Score: 0.01
epsilon: 3.6735210000039795
Episode 1624	Average Score: 0.01
epsilon: 3.672835000003977
Episode 1625	Average Score: 0.01
epsilon: 3.6721490000039743
Episode 1626	Average Score: 0.01
epsilon: 3.6714630000039716
Episode 1627	Average Score: 0.01
epsilon: 3.670728000003969
Episode 1628	Average Score: 0.01
epsilon: 3.670042000003966
Episode 1629	Average Score: 0.01
epsilon: 3.6693560000039636
Episode 1630	Average Score: 0.01
epsilon: 3.668670000003961
Episode 1631	Average Score: 0.01
epsilon: 3.6679840000039583
Episode 1632	Average Score: 0.01
epsilon: 3.6672490000039555
Episode 1633	Average Score: 0.01
epsilon: 3.666563000003953
Episode 1634	Average Score: 0.01
epsilon: 3.6658770000039502
Episode 1635	Average Score: 0.01
epsilon: 3.6644070000039446
Episode 1636	Average Score: 0.01
epsilon: 3.663770000003942
Episode 1637	Average Score: 0.01
epsilon: 3.6630840000039395
Episode 1638	Average Score: 0.01
epsilon: 3.6623490000039367
Episode 1639	Average Score: 0.01
epsilon: 3.660879000003931
Episode 1640	Average Score: 0.01
epsilon: 3.6601930000039284
Episode 1641	Average Score: 0.01
epsilon: 3.659507000003926
Episode 1642	Average Score: 0.01
epsilon: 3.658821000003923
Episode 1643	Average Score: 0.01
epsilon: 3.6580860000039204
Episode 1644	Average Score: 0.01
epsilon: 3.6565670000039145
Episode 1645	Average Score: 0.01
epsilon: 3.6558320000039117
Episode 1646	Average Score: 0.01
epsilon: 3.655146000003909
Episode 1647	Average Score: 0.01
epsilon: 3.6544600000039065
Episode 1648	Average Score: 0.01
epsilon: 3.653774000003904
Episode 1649	Average Score: 0.01
epsilon: 3.6500500000038896
Episode 1650	Average Score: 0.01
epsilon: 3.648629000003884
Episode 1651	Average Score: 0.01
epsilon: 3.6479430000038815
Episode 1652	Average Score: 0.01
epsilon: 3.647257000003879
Episode 1653	Average Score: 0.01
epsilon: 3.645738000003873
Episode 1654	Average Score: 0.01
epsilon: 3.6450520000038704
Episode 1655	Average Score: 0.01
epsilon: 3.644366000003868
Episode 1656	Average Score: 0.01
epsilon: 3.643680000003865
Episode 1657	Average Score: 0.01
epsilon: 3.6421610000038593
Episode 1658	Average Score: 0.01
epsilon: 3.6414750000038567
Episode 1659	Average Score: 0.01
epsilon: 3.640740000003854
Episode 1660	Average Score: 0.01
epsilon: 3.6400540000038513
Episode 1661	Average Score: 0.01
epsilon: 3.6393680000038486
Episode 1662	Average Score: 0.01
epsilon: 3.6385840000038456
Episode 1663	Average Score: 0.02
epsilon: 3.636281000003837
Episode 1664	Average Score: 0.01
epsilon: 3.635595000003834
Episode 1665	Average Score: 0.01
epsilon: 3.6349090000038315
Episode 1666	Average Score: 0.01
epsilon: 3.634223000003829
Episode 1667	Average Score: 0.01
epsilon: 3.6335370000038263
Episode 1668	Average Score: 0.01
epsilon: 3.6328510000038237
Episode 1669	Average Score: 0.01
epsilon: 3.632165000003821
Episode 1670	Average Score: 0.01
epsilon: 3.631430000003818
Episode 1671	Average Score: 0.01
epsilon: 3.6307440000038156
Episode 1672	Average Score: 0.01
epsilon: 3.630058000003813
Episode 1673	Average Score: 0.01
epsilon: 3.628539000003807
Episode 1674	Average Score: 0.01
epsilon: 3.6278530000038045
Episode 1675	Average Score: 0.01
epsilon: 3.627167000003802
Episode 1676	Average Score: 0.02
epsilon: 3.6256970000037962
Episode 1677	Average Score: 0.02
epsilon: 3.6249620000037934
Episode 1678	Average Score: 0.02
epsilon: 3.624276000003791
Episode 1679	Average Score: 0.02
epsilon: 3.623590000003788
Episode 1680	Average Score: 0.01
epsilon: 3.622512000003784
Episode 1681	Average Score: 0.01
epsilon: 3.6218260000037814
Episode 1682	Average Score: 0.01
epsilon: 3.621140000003779
Episode 1683	Average Score: 0.01
epsilon: 3.620454000003776
Episode 1684	Average Score: 0.01
epsilon: 3.6197190000037733
Episode 1685	Average Score: 0.01
epsilon: 3.6190330000037707
Episode 1686	Average Score: 0.01
epsilon: 3.618347000003768
Episode 1687	Average Score: 0.01
epsilon: 3.6176610000037654
Episode 1688	Average Score: 0.01
epsilon: 3.616975000003763
Episode 1689	Average Score: 0.01
epsilon: 3.61628900000376
Episode 1690	Average Score: 0.02
epsilon: 3.6147700000037544
Episode 1691	Average Score: 0.01
epsilon: 3.6140840000037517
Episode 1692	Average Score: 0.01
epsilon: 3.613398000003749
Episode 1693	Average Score: 0.01
epsilon: 3.6127120000037465
Episode 1694	Average Score: 0.01
epsilon: 3.6119770000037437
Episode 1695	Average Score: 0.01
epsilon: 3.611291000003741
Episode 1696	Average Score: 0.01
epsilon: 3.6106050000037384
Episode 1697	Average Score: 0.01
epsilon: 3.609919000003736
Episode 1698	Average Score: 0.01
epsilon: 3.609233000003733
Episode 1699	Average Score: 0.01
epsilon: 3.607665000003727
Episode 1700	Average Score: 0.01
epsilon: 3.6069790000037245
Episode 1701	Average Score: 0.01
epsilon: 3.606293000003722
Episode 1702	Average Score: 0.01
epsilon: 3.605558000003719
Episode 1703	Average Score: 0.01
epsilon: 3.6048720000037164
Episode 1704	Average Score: 0.01
epsilon: 3.604186000003714
Episode 1705	Average Score: 0.01
epsilon: 3.603451000003711
Episode 1706	Average Score: 0.01
epsilon: 3.6027650000037084
Episode 1707	Average Score: 0.01
epsilon: 3.6020790000037057
Episode 1708	Average Score: 0.01
epsilon: 3.601393000003703
Episode 1709	Average Score: 0.01
epsilon: 3.6006580000037003
Episode 1710	Average Score: 0.01
epsilon: 3.5999720000036977
Episode 1711	Average Score: 0.01
epsilon: 3.599286000003695
Episode 1712	Average Score: 0.01
epsilon: 3.5986000000036924
Episode 1713	Average Score: 0.01
epsilon: 3.5978650000036896
Episode 1714	Average Score: 0.01
epsilon: 3.597179000003687
Episode 1715	Average Score: 0.01
epsilon: 3.595660000003681
Episode 1716	Average Score: 0.01
epsilon: 3.5949740000036785
Episode 1717	Average Score: 0.01
epsilon: 3.594288000003676
Episode 1718	Average Score: 0.01
epsilon: 3.5936020000036732
Episode 1719	Average Score: 0.01
epsilon: 3.5929160000036706
Episode 1720	Average Score: 0.01
epsilon: 3.592230000003668
Episode 1721	Average Score: 0.01
epsilon: 3.591495000003665
Episode 1722	Average Score: 0.01
epsilon: 3.5908090000036625
Episode 1723	Average Score: 0.01
epsilon: 3.59012300000366
Episode 1724	Average Score: 0.01
epsilon: 3.5894370000036573
Episode 1725	Average Score: 0.01
epsilon: 3.585762000003643
Episode 1726	Average Score: 0.01
epsilon: 3.5850760000036406
Episode 1727	Average Score: 0.01
epsilon: 3.584390000003638
Episode 1728	Average Score: 0.01
epsilon: 3.5837040000036353
Episode 1729	Average Score: 0.01
epsilon: 3.5829690000036325
Episode 1730	Average Score: 0.02
epsilon: 3.5813520000036263
Episode 1731	Average Score: 0.02
epsilon: 3.5788040000036165
Episode 1732	Average Score: 0.02
epsilon: 3.577579000003612
Episode 1733	Average Score: 0.02
epsilon: 3.576893000003609
Episode 1734	Average Score: 0.02
epsilon: 3.5762070000036066
Episode 1735	Average Score: 0.02
epsilon: 3.575521000003604
Episode 1736	Average Score: 0.02
epsilon: 3.574786000003601
Episode 1737	Average Score: 0.02
epsilon: 3.573218000003595
Episode 1738	Average Score: 0.02
epsilon: 3.5725320000035925
Episode 1739	Average Score: 0.02
epsilon: 3.57184600000359
Episode 1740	Average Score: 0.02
epsilon: 3.5711600000035872
Episode 1741	Average Score: 0.02
epsilon: 3.5704740000035846
Episode 1742	Average Score: 0.02
epsilon: 3.569788000003582
Episode 1743	Average Score: 0.02
epsilon: 3.5691020000035794
Episode 1744	Average Score: 0.02
epsilon: 3.5665540000035696
Episode 1745	Average Score: 0.02
epsilon: 3.565084000003564
Episode 1746	Average Score: 0.02
epsilon: 3.5636140000035583
Episode 1747	Average Score: 0.02
epsilon: 3.5629280000035557
Episode 1748	Average Score: 0.02
epsilon: 3.562242000003553
Episode 1749	Average Score: 0.02
epsilon: 3.5615070000035502
Episode 1750	Average Score: 0.01
epsilon: 3.5608210000035476
Episode 1751	Average Score: 0.01
epsilon: 3.560135000003545
Episode 1752	Average Score: 0.01
epsilon: 3.5594980000035426
Episode 1753	Average Score: 0.01
epsilon: 3.5587630000035397
Episode 1754	Average Score: 0.01
epsilon: 3.558077000003537
Episode 1755	Average Score: 0.01
epsilon: 3.5573910000035345
Episode 1756	Average Score: 0.01
epsilon: 3.556705000003532
Episode 1757	Average Score: 0.01
epsilon: 3.556019000003529
Episode 1758	Average Score: 0.01
epsilon: 3.5553330000035266
Episode 1759	Average Score: 0.01
epsilon: 3.5545980000035238
Episode 1760	Average Score: 0.01
epsilon: 3.5539610000035213
Episode 1761	Average Score: 0.01
epsilon: 3.5532750000035187
Episode 1762	Average Score: 0.01
epsilon: 3.552589000003516
Episode 1763	Average Score: 0.01
epsilon: 3.5519030000035134
Episode 1764	Average Score: 0.01
epsilon: 3.551217000003511
Episode 1765	Average Score: 0.01
epsilon: 3.549698000003505
Episode 1766	Average Score: 0.01
epsilon: 3.5490120000035024
Episode 1767	Average Score: 0.01
epsilon: 3.548179000003499
Episode 1768	Average Score: 0.01
epsilon: 3.5474930000034965
Episode 1769	Average Score: 0.01
epsilon: 3.546807000003494
Episode 1770	Average Score: 0.01
epsilon: 3.546023000003491
Episode 1771	Average Score: 0.01
epsilon: 3.5453370000034883
Episode 1772	Average Score: 0.01
epsilon: 3.5446510000034857
Episode 1773	Average Score: 0.01
epsilon: 3.54323000000348
Episode 1774	Average Score: 0.01
epsilon: 3.5425440000034776
Episode 1775	Average Score: 0.01
epsilon: 3.5410250000034718
Episode 1776	Average Score: 0.01
epsilon: 3.540339000003469
Episode 1777	Average Score: 0.01
epsilon: 3.539555000003466
Episode 1778	Average Score: 0.01
epsilon: 3.5388690000034635
Episode 1779	Average Score: 0.01
epsilon: 3.538183000003461
Episode 1780	Average Score: 0.01
epsilon: 3.5374970000034582
Episode 1781	Average Score: 0.01
epsilon: 3.5368110000034556
Episode 1782	Average Score: 0.01
epsilon: 3.536125000003453
Episode 1783	Average Score: 0.01
epsilon: 3.5354390000034503
Episode 1784	Average Score: 0.01
epsilon: 3.5339690000034447
Episode 1785	Average Score: 0.01
epsilon: 3.532450000003439
Episode 1786	Average Score: 0.01
epsilon: 3.5317640000034363
Episode 1787	Average Score: 0.01
epsilon: 3.5310290000034334
Episode 1788	Average Score: 0.01
epsilon: 3.530343000003431
Episode 1789	Average Score: 0.01
epsilon: 3.529657000003428
Episode 1790	Average Score: 0.01
epsilon: 3.5289710000034256
Episode 1791	Average Score: 0.01
epsilon: 3.52750100000342
Episode 1792	Average Score: 0.01
epsilon: 3.5268150000034173
Episode 1793	Average Score: 0.01
epsilon: 3.5261290000034147
Episode 1794	Average Score: 0.01
epsilon: 3.525443000003412
Episode 1795	Average Score: 0.01
epsilon: 3.5247570000034094
Episode 1796	Average Score: 0.01
epsilon: 3.524071000003407
Episode 1797	Average Score: 0.01
epsilon: 3.523385000003404
Episode 1798	Average Score: 0.01
epsilon: 3.5226500000034013
Episode 1799	Average Score: 0.01
epsilon: 3.5219640000033987
Episode 1800	Average Score: 0.01
epsilon: 3.521278000003396
Episode 1801	Average Score: 0.01
epsilon: 3.5205920000033935
Episode 1802	Average Score: 0.01
epsilon: 3.519906000003391
Episode 1803	Average Score: 0.01
epsilon: 3.519220000003388
Episode 1804	Average Score: 0.01
epsilon: 3.5177990000033827
Episode 1805	Average Score: 0.01
epsilon: 3.5169660000033796
Episode 1806	Average Score: 0.01
epsilon: 3.516329000003377
Episode 1807	Average Score: 0.02
epsilon: 3.5148590000033715
Episode 1808	Average Score: 0.02
epsilon: 3.514173000003369
Episode 1809	Average Score: 0.02
epsilon: 3.5114780000033585
Episode 1810	Average Score: 0.02
epsilon: 3.5099590000033527
Episode 1811	Average Score: 0.02
epsilon: 3.50927300000335
Episode 1812	Average Score: 0.02
epsilon: 3.5085870000033474
Episode 1813	Average Score: 0.02
epsilon: 3.507901000003345
Episode 1814	Average Score: 0.02
epsilon: 3.507215000003342
Episode 1815	Average Score: 0.02
epsilon: 3.5064800000033394
Episode 1816	Average Score: 0.02
epsilon: 3.5057940000033367
Episode 1817	Average Score: 0.02
epsilon: 3.505108000003334
Episode 1818	Average Score: 0.02
epsilon: 3.5028050000033253
Episode 1819	Average Score: 0.02
epsilon: 3.502168000003323
Episode 1820	Average Score: 0.02
epsilon: 3.50143300000332
Episode 1821	Average Score: 0.02
epsilon: 3.5007470000033174
Episode 1822	Average Score: 0.02
epsilon: 3.5000610000033148
Episode 1823	Average Score: 0.02
epsilon: 3.499375000003312
Episode 1824	Average Score: 0.02
epsilon: 3.498591000003309
Episode 1825	Average Score: 0.02
epsilon: 3.4979050000033065
Episode 1826	Average Score: 0.02
epsilon: 3.497219000003304
Episode 1827	Average Score: 0.02
epsilon: 3.4965330000033013
Episode 1828	Average Score: 0.02
epsilon: 3.4958470000032986
Episode 1829	Average Score: 0.02
epsilon: 3.495161000003296
Episode 1830	Average Score: 0.02
epsilon: 3.4944750000032934
Episode 1831	Average Score: 0.01
epsilon: 3.4937400000032905
Episode 1832	Average Score: 0.01
epsilon: 3.493054000003288
Episode 1833	Average Score: 0.02
epsilon: 3.491486000003282
Episode 1834	Average Score: 0.02
epsilon: 3.4908000000032793
Episode 1835	Average Score: 0.02
epsilon: 3.4901140000032767
Episode 1836	Average Score: 0.02
epsilon: 3.487125000003265
Episode 1837	Average Score: 0.02
epsilon: 3.4864390000032626
Episode 1838	Average Score: 0.02
epsilon: 3.4857040000032598
Episode 1839	Average Score: 0.02
epsilon: 3.485018000003257
Episode 1840	Average Score: 0.02
epsilon: 3.4843320000032545
Episode 1841	Average Score: 0.02
epsilon: 3.483646000003252
Episode 1842	Average Score: 0.02
epsilon: 3.4829600000032492
Episode 1843	Average Score: 0.02
epsilon: 3.4822740000032466
Episode 1844	Average Score: 0.01
epsilon: 3.481588000003244
Episode 1845	Average Score: 0.01
epsilon: 3.4809020000032413
Episode 1846	Average Score: 0.01
epsilon: 3.4802160000032387
Episode 1847	Average Score: 0.01
epsilon: 3.479481000003236
Episode 1848	Average Score: 0.01
epsilon: 3.4787950000032333
Episode 1849	Average Score: 0.01
epsilon: 3.4781090000032306
Episode 1850	Average Score: 0.01
epsilon: 3.477423000003228
Episode 1851	Average Score: 0.01
epsilon: 3.4767370000032254
Episode 1852	Average Score: 0.01
epsilon: 3.4760510000032228
Episode 1853	Average Score: 0.01
epsilon: 3.47536500000322
Episode 1854	Average Score: 0.01
epsilon: 3.4746790000032175
Episode 1855	Average Score: 0.01
epsilon: 3.4731110000032115
Episode 1856	Average Score: 0.01
epsilon: 3.472425000003209
Episode 1857	Average Score: 0.01
epsilon: 3.4717390000032062
Episode 1858	Average Score: 0.01
epsilon: 3.4710530000032036
Episode 1859	Average Score: 0.01
epsilon: 3.470318000003201
Episode 1860	Average Score: 0.01
epsilon: 3.469632000003198
Episode 1861	Average Score: 0.01
epsilon: 3.4689460000031955
Episode 1862	Average Score: 0.01
epsilon: 3.468309000003193
Episode 1863	Average Score: 0.01
epsilon: 3.4675740000031903
Episode 1864	Average Score: 0.01
epsilon: 3.4668880000031876
Episode 1865	Average Score: 0.01
epsilon: 3.466202000003185
Episode 1866	Average Score: 0.01
epsilon: 3.4655160000031824
Episode 1867	Average Score: 0.01
epsilon: 3.4648300000031798
Episode 1868	Average Score: 0.01
epsilon: 3.464144000003177
Episode 1869	Average Score: 0.01
epsilon: 3.4634580000031745
Episode 1870	Average Score: 0.01
epsilon: 3.462772000003172
Episode 1871	Average Score: 0.01
epsilon: 3.4620860000031692
Episode 1872	Average Score: 0.01
epsilon: 3.4614000000031666
Episode 1873	Average Score: 0.01
epsilon: 3.460714000003164
Episode 1874	Average Score: 0.01
epsilon: 3.4600280000031614
Episode 1875	Average Score: 0.01
epsilon: 3.4585580000031557
Episode 1876	Average Score: 0.01
epsilon: 3.457872000003153
Episode 1877	Average Score: 0.01
epsilon: 3.4571860000031505
Episode 1878	Average Score: 0.01
epsilon: 3.4564510000031476
Episode 1879	Average Score: 0.01
epsilon: 3.455765000003145
Episode 1880	Average Score: 0.01
epsilon: 3.4550790000031424
Episode 1881	Average Score: 0.01
epsilon: 3.4543930000031398
Episode 1882	Average Score: 0.01
epsilon: 3.4529720000031343
Episode 1883	Average Score: 0.01
epsilon: 3.4514530000031285
Episode 1884	Average Score: 0.01
epsilon: 3.450767000003126
Episode 1885	Average Score: 0.01
epsilon: 3.4500810000031232
Episode 1886	Average Score: 0.01
epsilon: 3.4475330000031135
Episode 1887	Average Score: 0.01
epsilon: 3.446847000003111
Episode 1888	Average Score: 0.01
epsilon: 3.446112000003108
Episode 1889	Average Score: 0.01
epsilon: 3.4454260000031054
Episode 1890	Average Score: 0.01
epsilon: 3.4447400000031028
Episode 1891	Average Score: 0.01
epsilon: 3.4440540000031
Episode 1892	Average Score: 0.01
epsilon: 3.4433680000030975
Episode 1893	Average Score: 0.01
epsilon: 3.440575000003087
Episode 1894	Average Score: 0.01
epsilon: 3.4399380000030844
Episode 1895	Average Score: 0.01
epsilon: 3.4392520000030817
Episode 1896	Average Score: 0.01
epsilon: 3.438566000003079
Episode 1897	Average Score: 0.01
epsilon: 3.4378800000030765
Episode 1898	Average Score: 0.01
epsilon: 3.437194000003074
Episode 1899	Average Score: 0.01
epsilon: 3.436459000003071
Episode 1900	Average Score: 0.01
epsilon: 3.4352830000030665
Episode 1901	Average Score: 0.01
epsilon: 3.4345480000030637
Episode 1902	Average Score: 0.01
epsilon: 3.433862000003061
Episode 1903	Average Score: 0.01
epsilon: 3.433078000003058
Episode 1904	Average Score: 0.01
epsilon: 3.4323920000030554
Episode 1905	Average Score: 0.01
epsilon: 3.431706000003053
Episode 1906	Average Score: 0.01
epsilon: 3.43102000000305
Episode 1907	Average Score: 0.01
epsilon: 3.4302850000030474
Episode 1908	Average Score: 0.01
epsilon: 3.4295990000030447
Episode 1909	Average Score: 0.01
epsilon: 3.428913000003042
Episode 1910	Average Score: 0.01
epsilon: 3.428129000003039
Episode 1911	Average Score: 0.01
epsilon: 3.4274430000030365
Episode 1912	Average Score: 0.01
epsilon: 3.426757000003034
Episode 1913	Average Score: 0.01
epsilon: 3.423670000003022
Episode 1914	Average Score: 0.01
epsilon: 3.4229840000030194
Episode 1915	Average Score: 0.01
epsilon: 3.4222490000030166
Episode 1916	Average Score: 0.01
epsilon: 3.421563000003014
Episode 1917	Average Score: 0.01
epsilon: 3.4208770000030113
Episode 1918	Average Score: 0.01
epsilon: 3.4194070000030057
Episode 1919	Average Score: 0.01
epsilon: 3.418672000003003
Episode 1920	Average Score: 0.01
epsilon: 3.417104000002997
Episode 1921	Average Score: 0.01
epsilon: 3.4164180000029942
Episode 1922	Average Score: 0.01
epsilon: 3.4157320000029916
Episode 1923	Average Score: 0.01
epsilon: 3.415046000002989
Episode 1924	Average Score: 0.01
epsilon: 3.413527000002983
Episode 1925	Average Score: 0.01
epsilon: 3.4120570000029775
Episode 1926	Average Score: 0.01
epsilon: 3.410636000002972
Episode 1927	Average Score: 0.01
epsilon: 3.4099500000029694
Episode 1928	Average Score: 0.01
epsilon: 3.409264000002967
Episode 1929	Average Score: 0.01
epsilon: 3.408529000002964
Episode 1930	Average Score: 0.01
epsilon: 3.4078430000029614
Episode 1931	Average Score: 0.01
epsilon: 3.4071570000029587
Episode 1932	Average Score: 0.01
epsilon: 3.4065200000029563
Episode 1933	Average Score: 0.01
epsilon: 3.4057850000029535
Episode 1934	Average Score: 0.01
epsilon: 3.404315000002948
Episode 1935	Average Score: 0.01
epsilon: 3.403629000002945
Episode 1936	Average Score: 0.01
epsilon: 3.4029430000029426
Episode 1937	Average Score: 0.01
epsilon: 3.4022080000029398
Episode 1938	Average Score: 0.01
epsilon: 3.4005910000029336
Episode 1939	Average Score: 0.01
epsilon: 3.399954000002931
Episode 1940	Average Score: 0.01
epsilon: 3.3992680000029285
Episode 1941	Average Score: 0.02
epsilon: 3.396475000002918
Episode 1942	Average Score: 0.02
epsilon: 3.395789000002915
Episode 1943	Average Score: 0.02
epsilon: 3.3951030000029125
Episode 1944	Average Score: 0.02
epsilon: 3.3943680000029097
Episode 1945	Average Score: 0.02
epsilon: 3.393682000002907
Episode 1946	Average Score: 0.02
epsilon: 3.3921630000029013
Episode 1947	Average Score: 0.02
epsilon: 3.3914770000028986
Episode 1948	Average Score: 0.02
epsilon: 3.390742000002896
Episode 1949	Average Score: 0.02
epsilon: 3.390056000002893
Episode 1950	Average Score: 0.02
epsilon: 3.3893700000028906
Episode 1951	Average Score: 0.02
epsilon: 3.388684000002888
Episode 1952	Average Score: 0.02
epsilon: 3.3879980000028853
Episode 1953	Average Score: 0.02
epsilon: 3.3873120000028827
Episode 1954	Average Score: 0.02
epsilon: 3.3856950000028765
Episode 1955	Average Score: 0.02
epsilon: 3.385058000002874
Episode 1956	Average Score: 0.02
epsilon: 3.3843230000028712
Episode 1957	Average Score: 0.02
epsilon: 3.383686000002869
Episode 1958	Average Score: 0.02
epsilon: 3.382951000002866
Episode 1959	Average Score: 0.02
epsilon: 3.3822650000028633
Episode 1960	Average Score: 0.02
epsilon: 3.3815790000028607
Episode 1961	Average Score: 0.02
epsilon: 3.380893000002858
Episode 1962	Average Score: 0.02
epsilon: 3.3802070000028555
Episode 1963	Average Score: 0.02
epsilon: 3.379521000002853
Episode 1964	Average Score: 0.02
epsilon: 3.37883500000285
Episode 1965	Average Score: 0.02
epsilon: 3.3773650000028446
Episode 1966	Average Score: 0.02
epsilon: 3.3765810000028416
Episode 1967	Average Score: 0.02
epsilon: 3.3750620000028357
Episode 1968	Average Score: 0.02
epsilon: 3.374376000002833
Episode 1969	Average Score: 0.02
epsilon: 3.3736900000028305
Episode 1970	Average Score: 0.02
epsilon: 3.373004000002828
Episode 1971	Average Score: 0.02
epsilon: 3.3723180000028252
Episode 1972	Average Score: 0.02
epsilon: 3.3715830000028224
Episode 1973	Average Score: 0.02
epsilon: 3.3708970000028198
Episode 1974	Average Score: 0.02
epsilon: 3.370211000002817
Episode 1975	Average Score: 0.02
epsilon: 3.3695250000028145
Episode 1976	Average Score: 0.02
epsilon: 3.368839000002812
Episode 1977	Average Score: 0.02
epsilon: 3.368104000002809
Episode 1978	Average Score: 0.02
epsilon: 3.3674180000028064
Episode 1979	Average Score: 0.02
epsilon: 3.366732000002804
Episode 1980	Average Score: 0.02
epsilon: 3.366046000002801
Episode 1981	Average Score: 0.02
epsilon: 3.364478000002795
Episode 1982	Average Score: 0.02
epsilon: 3.3637920000027925
Episode 1983	Average Score: 0.02
epsilon: 3.36310600000279
Episode 1984	Average Score: 0.02
epsilon: 3.3624200000027873
Episode 1985	Average Score: 0.02
epsilon: 3.3616360000027843
Episode 1986	Average Score: 0.02
epsilon: 3.360803000002781
Episode 1987	Average Score: 0.02
epsilon: 3.3601170000027785
Episode 1988	Average Score: 0.02
epsilon: 3.3593330000027755
Episode 1989	Average Score: 0.02
epsilon: 3.358647000002773
Episode 1990	Average Score: 0.02
epsilon: 3.35796100000277
Episode 1991	Average Score: 0.02
epsilon: 3.356393000002764
Episode 1992	Average Score: 0.02
epsilon: 3.3537960000027542
Episode 1993	Average Score: 0.02
epsilon: 3.3531100000027516
Episode 1994	Average Score: 0.02
epsilon: 3.352424000002749
Episode 1995	Average Score: 0.02
epsilon: 3.3517380000027464
Episode 1996	Average Score: 0.02
epsilon: 3.3502680000027407
Episode 1997	Average Score: 0.02
epsilon: 3.349533000002738
Episode 1998	Average Score: 0.02
epsilon: 3.348798000002735
Episode 1999	Average Score: 0.02
epsilon: 3.3473770000027296
Episode 2000	Average Score: 0.02
epsilon: 3.346691000002727
Episode 2001	Average Score: 0.02
epsilon: 3.3460050000027244
Episode 2002	Average Score: 0.02
epsilon: 3.3452700000027216
Episode 2003	Average Score: 0.02
epsilon: 3.344633000002719
Episode 2004	Average Score: 0.02
epsilon: 3.3439470000027165
Episode 2005	Average Score: 0.02
epsilon: 3.3432120000027137
Episode 2006	Average Score: 0.02
epsilon: 3.3425750000027112
Episode 2007	Average Score: 0.02
epsilon: 3.3418890000027086
Episode 2008	Average Score: 0.02
epsilon: 3.341154000002706
Episode 2009	Average Score: 0.02
epsilon: 3.340468000002703
Episode 2010	Average Score: 0.02
epsilon: 3.3398310000027007
Episode 2011	Average Score: 0.02
epsilon: 3.339096000002698
Episode 2012	Average Score: 0.02
epsilon: 3.338361000002695
Episode 2013	Average Score: 0.02
epsilon: 3.3377240000026926
Episode 2014	Average Score: 0.02
epsilon: 3.3361560000026866
Episode 2015	Average Score: 0.02
epsilon: 3.335470000002684
Episode 2016	Average Score: 0.02
epsilon: 3.3347840000026814
Episode 2017	Average Score: 0.02
epsilon: 3.3340980000026788
Episode 2018	Average Score: 0.02
epsilon: 3.3316480000026694
Episode 2019	Average Score: 0.02
epsilon: 3.3309620000026667
Episode 2020	Average Score: 0.02
epsilon: 3.330227000002664
Episode 2021	Average Score: 0.02
epsilon: 3.3271890000026523
Episode 2022	Average Score: 0.02
epsilon: 3.326356000002649
Episode 2023	Average Score: 0.02
epsilon: 3.3256700000026465
Episode 2024	Average Score: 0.02
epsilon: 3.324200000002641
Episode 2025	Average Score: 0.02
epsilon: 3.322681000002635
Episode 2026	Average Score: 0.02
epsilon: 3.3219950000026324
Episode 2027	Average Score: 0.02
epsilon: 3.3213090000026297
Episode 2028	Average Score: 0.02
epsilon: 3.320623000002627
Episode 2029	Average Score: 0.02
epsilon: 3.3191040000026213
Episode 2030	Average Score: 0.02
epsilon: 3.3184180000026187
Episode 2031	Average Score: 0.02
epsilon: 3.317683000002616
Episode 2032	Average Score: 0.02
epsilon: 3.316997000002613
Episode 2033	Average Score: 0.02
epsilon: 3.3163110000026106
Episode 2034	Average Score: 0.02
epsilon: 3.3155760000026078
Episode 2035	Average Score: 0.02
epsilon: 3.314890000002605
Episode 2036	Average Score: 0.02
epsilon: 3.3142040000026025
Episode 2037	Average Score: 0.02
epsilon: 3.3135180000026
Episode 2038	Average Score: 0.02
epsilon: 3.3119010000025937
Episode 2039	Average Score: 0.02
epsilon: 3.311215000002591
Episode 2040	Average Score: 0.02
epsilon: 3.3105290000025884
Episode 2041	Average Score: 0.01
epsilon: 3.309843000002586
Episode 2042	Average Score: 0.01
epsilon: 3.309157000002583
Episode 2043	Average Score: 0.01
epsilon: 3.3084220000025804
Episode 2044	Average Score: 0.02
epsilon: 3.3069030000025745
Episode 2045	Average Score: 0.02
epsilon: 3.306217000002572
Episode 2046	Average Score: 0.01
epsilon: 3.3055310000025693
Episode 2047	Average Score: 0.01
epsilon: 3.3048450000025666
Episode 2048	Average Score: 0.01
epsilon: 3.304159000002564
Episode 2049	Average Score: 0.01
epsilon: 3.3034730000025614
Episode 2050	Average Score: 0.01
epsilon: 3.3027870000025588
Episode 2051	Average Score: 0.01
epsilon: 3.302101000002556
Episode 2052	Average Score: 0.01
epsilon: 3.3014150000025535
Episode 2053	Average Score: 0.02
epsilon: 3.2998470000025475
Episode 2054	Average Score: 0.01
epsilon: 3.299161000002545
Episode 2055	Average Score: 0.01
epsilon: 3.2984750000025422
Episode 2056	Average Score: 0.02
epsilon: 3.2969560000025364
Episode 2057	Average Score: 0.02
epsilon: 3.296270000002534
Episode 2058	Average Score: 0.02
epsilon: 3.295584000002531
Episode 2059	Average Score: 0.02
epsilon: 3.2948980000025285
Episode 2060	Average Score: 0.02
epsilon: 3.2933300000025225
Episode 2061	Average Score: 0.02
epsilon: 3.29264400000252
Episode 2062	Average Score: 0.02
epsilon: 3.2919580000025173
Episode 2063	Average Score: 0.02
epsilon: 3.2912230000025144
Episode 2064	Average Score: 0.02
epsilon: 3.290537000002512
Episode 2065	Average Score: 0.02
epsilon: 3.2899000000025094
Episode 2066	Average Score: 0.02
epsilon: 3.2891650000025066
Episode 2067	Average Score: 0.01
epsilon: 3.288479000002504
Episode 2068	Average Score: 0.02
epsilon: 3.286960000002498
Episode 2069	Average Score: 0.02
epsilon: 3.2863230000024957
Episode 2070	Average Score: 0.02
epsilon: 3.285637000002493
Episode 2071	Average Score: 0.02
epsilon: 3.2849510000024904
Episode 2072	Average Score: 0.02
epsilon: 3.2841670000024874
Episode 2073	Average Score: 0.02
epsilon: 3.2834320000024846
Episode 2074	Average Score: 0.02
epsilon: 3.282746000002482
Episode 2075	Average Score: 0.02
epsilon: 3.2820600000024793
Episode 2076	Average Score: 0.02
epsilon: 3.2813740000024767
Episode 2077	Average Score: 0.02
epsilon: 3.280688000002474
Episode 2078	Average Score: 0.02
epsilon: 3.2800020000024714
Episode 2079	Average Score: 0.02
epsilon: 3.279316000002469
Episode 2080	Average Score: 0.02
epsilon: 3.278630000002466
Episode 2081	Average Score: 0.01
epsilon: 3.2779440000024636
Episode 2082	Average Score: 0.01
epsilon: 3.277258000002461
Episode 2083	Average Score: 0.01
epsilon: 3.2765720000024583
Episode 2084	Average Score: 0.01
epsilon: 3.2758860000024557
Episode 2085	Average Score: 0.01
epsilon: 3.275200000002453
Episode 2086	Average Score: 0.01
epsilon: 3.2745140000024504
Episode 2087	Average Score: 0.01
epsilon: 3.273828000002448
Episode 2088	Average Score: 0.01
epsilon: 3.273093000002445
Episode 2089	Average Score: 0.01
epsilon: 3.2724070000024423
Episode 2090	Average Score: 0.01
epsilon: 3.2717210000024397
Episode 2091	Average Score: 0.01
epsilon: 3.270202000002434
Episode 2092	Average Score: 0.01
epsilon: 3.2695160000024313
Episode 2093	Average Score: 0.01
epsilon: 3.2679970000024254
Episode 2094	Average Score: 0.01
epsilon: 3.267311000002423
Episode 2095	Average Score: 0.01
epsilon: 3.26662500000242
Episode 2096	Average Score: 0.01
epsilon: 3.265743000002417
Episode 2097	Average Score: 0.01
epsilon: 3.265008000002414
Episode 2098	Average Score: 0.01
epsilon: 3.2643220000024114
Episode 2099	Average Score: 0.01
epsilon: 3.2636360000024087
Episode 2100	Average Score: 0.01
epsilon: 3.262901000002406
Episode 2101	Average Score: 0.01
epsilon: 3.2622640000024035
Episode 2102	Average Score: 0.01
epsilon: 3.2615290000024006
Episode 2103	Average Score: 0.01
epsilon: 3.260843000002398
Episode 2104	Average Score: 0.01
epsilon: 3.260108000002395
Episode 2105	Average Score: 0.01
epsilon: 3.2594220000023926
Episode 2106	Average Score: 0.01
epsilon: 3.25878500000239
Episode 2107	Average Score: 0.01
epsilon: 3.2580990000023875
Episode 2108	Average Score: 0.01
epsilon: 3.2565800000023817
Episode 2109	Average Score: 0.01
epsilon: 3.255894000002379
Episode 2110	Average Score: 0.01
epsilon: 3.2552080000023764
Episode 2111	Average Score: 0.01
epsilon: 3.2544730000023736
Episode 2112	Average Score: 0.01
epsilon: 3.253787000002371
Episode 2113	Average Score: 0.01
epsilon: 3.2531010000023683
Episode 2114	Average Score: 0.01
epsilon: 3.2524150000023657
Episode 2115	Average Score: 0.01
epsilon: 3.251680000002363
Episode 2116	Average Score: 0.01
epsilon: 3.2509940000023603
Episode 2117	Average Score: 0.01
epsilon: 3.2503080000023576
Episode 2118	Average Score: 0.01
epsilon: 3.249622000002355
Episode 2119	Average Score: 0.01
epsilon: 3.247319000002346
Episode 2120	Average Score: 0.01
epsilon: 3.246486000002343
Episode 2121	Average Score: 0.01
epsilon: 3.2458000000023404
Episode 2122	Average Score: 0.01
epsilon: 3.2451140000023377
Episode 2123	Average Score: 0.01
epsilon: 3.244379000002335
Episode 2124	Average Score: 0.01
epsilon: 3.2436930000023323
Episode 2125	Average Score: 0.01
epsilon: 3.2430070000023297
Episode 2126	Average Score: 0.01
epsilon: 3.242321000002327
Episode 2127	Average Score: 0.01
epsilon: 3.2416350000023244
Episode 2128	Average Score: 0.01
epsilon: 3.240949000002322
Episode 2129	Average Score: 0.01
epsilon: 3.240263000002319
Episode 2130	Average Score: 0.01
epsilon: 3.2395770000023165
Episode 2131	Average Score: 0.01
epsilon: 3.2380580000023107
Episode 2132	Average Score: 0.01
epsilon: 3.235755000002302
Episode 2133	Average Score: 0.01
epsilon: 3.234775000002298
Episode 2134	Average Score: 0.01
epsilon: 3.2333050000022925
Episode 2135	Average Score: 0.01
epsilon: 3.2325700000022897
Episode 2136	Average Score: 0.01
epsilon: 3.231884000002287
Episode 2137	Average Score: 0.01
epsilon: 3.2311980000022844
Episode 2138	Average Score: 0.01
epsilon: 3.230512000002282
Episode 2139	Average Score: 0.01
epsilon: 3.229826000002279
Episode 2140	Average Score: 0.01
epsilon: 3.2290910000022763
Episode 2141	Average Score: 0.01
epsilon: 3.2284050000022737
Episode 2142	Average Score: 0.01
epsilon: 3.227719000002271
Episode 2143	Average Score: 0.01
epsilon: 3.2270330000022684
Episode 2144	Average Score: 0.01
epsilon: 3.226347000002266
Episode 2145	Average Score: 0.01
epsilon: 3.225612000002263
Episode 2146	Average Score: 0.01
epsilon: 3.2249260000022604
Episode 2147	Average Score: 0.01
epsilon: 3.2242400000022577
Episode 2148	Average Score: 0.01
epsilon: 3.223554000002255
Episode 2149	Average Score: 0.01
epsilon: 3.2228680000022525
Episode 2150	Average Score: 0.01
epsilon: 3.22218200000225
Episode 2151	Average Score: 0.01
epsilon: 3.2214960000022472
Episode 2152	Average Score: 0.01
epsilon: 3.2207610000022444
Episode 2153	Average Score: 0.01
epsilon: 3.220075000002242
Episode 2154	Average Score: 0.01
epsilon: 3.219389000002239
Episode 2155	Average Score: 0.01
epsilon: 3.2187030000022365
Episode 2156	Average Score: 0.01
epsilon: 3.2171840000022307
Episode 2157	Average Score: 0.01
epsilon: 3.216498000002228
Episode 2158	Average Score: 0.01
epsilon: 3.2158120000022254
Episode 2159	Average Score: 0.01
epsilon: 3.215126000002223
Episode 2160	Average Score: 0.01
epsilon: 3.21444000000222
Episode 2161	Average Score: 0.01
epsilon: 3.2137050000022174
Episode 2162	Average Score: 0.01
epsilon: 3.2130190000022147
Episode 2163	Average Score: 0.01
epsilon: 3.212333000002212
Episode 2164	Average Score: 0.01
epsilon: 3.2116470000022095
Episode 2165	Average Score: 0.01
epsilon: 3.210961000002207
Episode 2166	Average Score: 0.01
epsilon: 3.210226000002204
Episode 2167	Average Score: 0.01
epsilon: 3.2095400000022014
Episode 2168	Average Score: 0.01
epsilon: 3.208854000002199
Episode 2169	Average Score: 0.01
epsilon: 3.2080700000021958
Episode 2170	Average Score: 0.01
epsilon: 3.207384000002193
Episode 2171	Average Score: 0.01
epsilon: 3.205816000002187
Episode 2172	Average Score: 0.01
epsilon: 3.204248000002181
Episode 2173	Average Score: 0.01
epsilon: 3.2035620000021785
Episode 2174	Average Score: 0.01
epsilon: 3.202876000002176
Episode 2175	Average Score: 0.01
epsilon: 3.20135700000217
Episode 2176	Average Score: 0.01
epsilon: 3.2006710000021674
Episode 2177	Average Score: 0.01
epsilon: 3.199985000002165
Episode 2178	Average Score: 0.01
epsilon: 3.199250000002162
Episode 2179	Average Score: 0.01
epsilon: 3.1985640000021593
Episode 2180	Average Score: 0.01
epsilon: 3.1978780000021567
Episode 2181	Average Score: 0.01
epsilon: 3.197192000002154
Episode 2182	Average Score: 0.01
epsilon: 3.1965060000021515
Episode 2183	Average Score: 0.01
epsilon: 3.1957710000021486
Episode 2184	Average Score: 0.01
epsilon: 3.195134000002146
Episode 2185	Average Score: 0.01
epsilon: 3.1944480000021436
Episode 2186	Average Score: 0.01
epsilon: 3.1936640000021406
Episode 2187	Average Score: 0.01
epsilon: 3.1929290000021378
Episode 2188	Average Score: 0.01
epsilon: 3.192243000002135
Episode 2189	Average Score: 0.01
epsilon: 3.1915570000021325
Episode 2190	Average Score: 0.01
epsilon: 3.19087100000213
Episode 2191	Average Score: 0.01
epsilon: 3.1901850000021272
Episode 2192	Average Score: 0.01
epsilon: 3.1894500000021244
Episode 2193	Average Score: 0.01
epsilon: 3.188764000002122
Episode 2194	Average Score: 0.01
epsilon: 3.188029000002119
Episode 2195	Average Score: 0.01
epsilon: 3.1873430000021163
Episode 2196	Average Score: 0.01
epsilon: 3.1866570000021137
Episode 2197	Average Score: 0.01
epsilon: 3.1850890000021077
Episode 2198	Average Score: 0.01
epsilon: 3.1836680000021023
Episode 2199	Average Score: 0.01
epsilon: 3.1829330000020994
Episode 2200	Average Score: 0.01
epsilon: 3.182247000002097
Episode 2201	Average Score: 0.01
epsilon: 3.181561000002094
Episode 2202	Average Score: 0.01
epsilon: 3.180777000002091
Episode 2203	Average Score: 0.01
epsilon: 3.1800420000020884
Episode 2204	Average Score: 0.01
epsilon: 3.179405000002086
Episode 2205	Average Score: 0.01
epsilon: 3.1787190000020833
Episode 2206	Average Score: 0.01
epsilon: 3.1780330000020807
Episode 2207	Average Score: 0.01
epsilon: 3.177347000002078
Episode 2208	Average Score: 0.01
epsilon: 3.1766610000020754
Episode 2209	Average Score: 0.01
epsilon: 3.1751420000020696
Episode 2210	Average Score: 0.01
epsilon: 3.174456000002067
Episode 2211	Average Score: 0.01
epsilon: 3.1720550000020578
Episode 2212	Average Score: 0.01
epsilon: 3.171369000002055
Episode 2213	Average Score: 0.01
epsilon: 3.1706830000020525
Episode 2214	Average Score: 0.01
epsilon: 3.16999700000205
Episode 2215	Average Score: 0.01
epsilon: 3.1693110000020472
Episode 2216	Average Score: 0.01
epsilon: 3.1685760000020444
Episode 2217	Average Score: 0.01
epsilon: 3.167890000002042
Episode 2218	Average Score: 0.01
epsilon: 3.167204000002039
Episode 2219	Average Score: 0.01
epsilon: 3.166420000002036
Episode 2220	Average Score: 0.01
epsilon: 3.1657340000020335
Episode 2221	Average Score: 0.01
epsilon: 3.165048000002031
Episode 2222	Average Score: 0.01
epsilon: 3.1643620000020283
Episode 2223	Average Score: 0.01
epsilon: 3.1636760000020256
Episode 2224	Average Score: 0.01
epsilon: 3.162990000002023
Episode 2225	Average Score: 0.01
epsilon: 3.16225500000202
Episode 2226	Average Score: 0.01
epsilon: 3.1607360000020144
Episode 2227	Average Score: 0.01
epsilon: 3.1600010000020116
Episode 2228	Average Score: 0.01
epsilon: 3.159315000002009
Episode 2229	Average Score: 0.01
epsilon: 3.1586290000020063
Episode 2230	Average Score: 0.01
epsilon: 3.1579430000020037
Episode 2231	Average Score: 0.01
epsilon: 3.157257000002001
Episode 2232	Average Score: 0.01
epsilon: 3.1565220000019982
Episode 2233	Average Score: 0.01
epsilon: 3.1550030000019924
Episode 2234	Average Score: 0.01
epsilon: 3.15436600000199
Episode 2235	Average Score: 0.01
epsilon: 3.1536800000019873
Episode 2236	Average Score: 0.01
epsilon: 3.1529940000019847
Episode 2237	Average Score: 0.01
epsilon: 3.152259000001982
Episode 2238	Average Score: 0.01
epsilon: 3.1516220000019795
Episode 2239	Average Score: 0.01
epsilon: 3.1508870000019766
Episode 2240	Average Score: 0.01
epsilon: 3.150250000001974
Episode 2241	Average Score: 0.01
epsilon: 3.1495150000019714
Episode 2242	Average Score: 0.01
epsilon: 3.1488290000019687
Episode 2243	Average Score: 0.01
epsilon: 3.148143000001966
Episode 2244	Average Score: 0.01
epsilon: 3.1474570000019635
Episode 2245	Average Score: 0.01
epsilon: 3.146771000001961
Episode 2246	Average Score: 0.01
epsilon: 3.146036000001958
Episode 2247	Average Score: 0.01
epsilon: 3.1453500000019554
Episode 2248	Average Score: 0.01
epsilon: 3.144664000001953
Episode 2249	Average Score: 0.01
epsilon: 3.14392900000195
Episode 2250	Average Score: 0.01
epsilon: 3.1424590000019443
Episode 2251	Average Score: 0.01
epsilon: 3.1417730000019417
Episode 2252	Average Score: 0.01
epsilon: 3.141087000001939
Episode 2253	Average Score: 0.01
epsilon: 3.1404500000019366
Episode 2254	Average Score: 0.01
epsilon: 3.139715000001934
Episode 2255	Average Score: 0.01
epsilon: 3.139029000001931
Episode 2256	Average Score: 0.01
epsilon: 3.1383430000019286
Episode 2257	Average Score: 0.01
epsilon: 3.1367750000019226
Episode 2258	Average Score: 0.01
epsilon: 3.13608900000192
Episode 2259	Average Score: 0.01
epsilon: 3.1354030000019173
Episode 2260	Average Score: 0.01
epsilon: 3.1347170000019147
Episode 2261	Average Score: 0.01
epsilon: 3.1339330000019117
Episode 2262	Average Score: 0.01
epsilon: 3.133198000001909
Episode 2263	Average Score: 0.01
epsilon: 3.132512000001906
Episode 2264	Average Score: 0.01
epsilon: 3.1318260000019036
Episode 2265	Average Score: 0.01
epsilon: 3.131140000001901
Episode 2266	Average Score: 0.01
epsilon: 3.1304540000018983
Episode 2267	Average Score: 0.01
epsilon: 3.1297680000018957
Episode 2268	Average Score: 0.01
epsilon: 3.129033000001893
Episode 2269	Average Score: 0.01
epsilon: 3.1283470000018903
Episode 2270	Average Score: 0.01
epsilon: 3.1268770000018846
Episode 2271	Average Score: 0.01
epsilon: 3.126191000001882
Episode 2272	Average Score: 0.01
epsilon: 3.125407000001879
Episode 2273	Average Score: 0.01
epsilon: 3.1247210000018764
Episode 2274	Average Score: 0.01
epsilon: 3.1240350000018737
Episode 2275	Average Score: 0.01
epsilon: 3.123300000001871
Episode 2276	Average Score: 0.01
epsilon: 3.1226140000018683
Episode 2277	Average Score: 0.01
epsilon: 3.1219280000018657
Episode 2278	Average Score: 0.01
epsilon: 3.121242000001863
Episode 2279	Average Score: 0.01
epsilon: 3.119674000001857
Episode 2280	Average Score: 0.01
epsilon: 3.118155000001851
Episode 2281	Average Score: 0.01
epsilon: 3.1174690000018486
Episode 2282	Average Score: 0.01
epsilon: 3.116783000001846
Episode 2283	Average Score: 0.01
epsilon: 3.1160970000018433
Episode 2284	Average Score: 0.01
epsilon: 3.1135490000018335
Episode 2285	Average Score: 0.01
epsilon: 3.112863000001831
Episode 2286	Average Score: 0.01
epsilon: 3.1121770000018283
Episode 2287	Average Score: 0.01
epsilon: 3.1114910000018257
Episode 2288	Average Score: 0.01
epsilon: 3.110756000001823
Episode 2289	Average Score: 0.01
epsilon: 3.11007000000182
Episode 2290	Average Score: 0.01
epsilon: 3.1093840000018176
Episode 2291	Average Score: 0.01
epsilon: 3.108698000001815
Episode 2292	Average Score: 0.01
epsilon: 3.1080120000018123
Episode 2293	Average Score: 0.01
epsilon: 3.1072770000018095
Episode 2294	Average Score: 0.01
epsilon: 3.1065420000018067
Episode 2295	Average Score: 0.01
epsilon: 3.1059050000018043
Episode 2296	Average Score: 0.01
epsilon: 3.1043860000017984
Episode 2297	Average Score: 0.01
epsilon: 3.103700000001796
Episode 2298	Average Score: 0.01
epsilon: 3.103014000001793
Episode 2299	Average Score: 0.01
epsilon: 3.1023280000017905
Episode 2300	Average Score: 0.01
epsilon: 3.101642000001788
Episode 2301	Average Score: 0.01
epsilon: 3.100907000001785
Episode 2302	Average Score: 0.01
epsilon: 3.1002210000017825
Episode 2303	Average Score: 0.01
epsilon: 3.09953500000178
Episode 2304	Average Score: 0.01
epsilon: 3.098751000001777
Episode 2305	Average Score: 0.01
epsilon: 3.097183000001771
Episode 2306	Average Score: 0.01
epsilon: 3.096497000001768
Episode 2307	Average Score: 0.01
epsilon: 3.0958110000017656
Episode 2308	Average Score: 0.01
epsilon: 3.095125000001763
Episode 2309	Average Score: 0.01
epsilon: 3.093606000001757
Episode 2310	Average Score: 0.01
epsilon: 3.0929200000017545
Episode 2311	Average Score: 0.01
epsilon: 3.0921850000017517
Episode 2312	Average Score: 0.01
epsilon: 3.090715000001746
Episode 2313	Average Score: 0.01
epsilon: 3.0900290000017434
Episode 2314	Average Score: 0.01
epsilon: 3.089343000001741
Episode 2315	Average Score: 0.01
epsilon: 3.088657000001738
Episode 2316	Average Score: 0.01
epsilon: 3.0879710000017355
Episode 2317	Average Score: 0.01
epsilon: 3.087285000001733
Episode 2318	Average Score: 0.01
epsilon: 3.0865990000017303
Episode 2319	Average Score: 0.01
epsilon: 3.0858640000017274
Episode 2320	Average Score: 0.01
epsilon: 3.085178000001725
Episode 2321	Average Score: 0.01
epsilon: 3.084492000001722
Episode 2322	Average Score: 0.01
epsilon: 3.0838060000017196
Episode 2323	Average Score: 0.01
epsilon: 3.0822870000017137
Episode 2324	Average Score: 0.01
epsilon: 3.081601000001711
Episode 2325	Average Score: 0.01
epsilon: 3.0801310000017055
Episode 2326	Average Score: 0.01
epsilon: 3.0786610000017
Episode 2327	Average Score: 0.01
epsilon: 3.077142000001694
Episode 2328	Average Score: 0.01
epsilon: 3.0764560000016914
Episode 2329	Average Score: 0.01
epsilon: 3.0757700000016888
Episode 2330	Average Score: 0.01
epsilon: 3.075084000001686
Episode 2331	Average Score: 0.01
epsilon: 3.0743980000016835
Episode 2332	Average Score: 0.01
epsilon: 3.0736630000016807
Episode 2333	Average Score: 0.01
epsilon: 3.072977000001678
Episode 2334	Average Score: 0.01
epsilon: 3.0722910000016754
Episode 2335	Average Score: 0.01
epsilon: 3.071605000001673
Episode 2336	Average Score: 0.01
epsilon: 3.07091900000167
Episode 2337	Average Score: 0.01
epsilon: 3.0701840000016674
Episode 2338	Average Score: 0.01
epsilon: 3.069547000001665
Episode 2339	Average Score: 0.01
epsilon: 3.068028000001659
Episode 2340	Average Score: 0.01
epsilon: 3.0665090000016533
Episode 2341	Average Score: 0.01
epsilon: 3.0658230000016506
Episode 2342	Average Score: 0.01
epsilon: 3.065137000001648
Episode 2343	Average Score: 0.02
epsilon: 3.0636670000016424
Episode 2344	Average Score: 0.02
epsilon: 3.0629810000016398
Episode 2345	Average Score: 0.02
epsilon: 3.062295000001637
Episode 2346	Average Score: 0.02
epsilon: 3.0615600000016343
Episode 2347	Average Score: 0.02
epsilon: 3.0608740000016317
Episode 2348	Average Score: 0.02
epsilon: 3.060139000001629
Episode 2349	Average Score: 0.02
epsilon: 3.0594530000016262
Episode 2350	Average Score: 0.01
epsilon: 3.0586690000016232
Episode 2351	Average Score: 0.01
epsilon: 3.0579830000016206
Episode 2352	Average Score: 0.01
epsilon: 3.057297000001618
Episode 2353	Average Score: 0.01
epsilon: 3.0566110000016153
Episode 2354	Average Score: 0.01
epsilon: 3.0559250000016127
Episode 2355	Average Score: 0.01
epsilon: 3.0552880000016103
Episode 2356	Average Score: 0.01
epsilon: 3.0546020000016076
Episode 2357	Average Score: 0.01
epsilon: 3.053916000001605
Episode 2358	Average Score: 0.01
epsilon: 3.053181000001602
Episode 2359	Average Score: 0.01
epsilon: 3.0525440000015998
Episode 2360	Average Score: 0.01
epsilon: 3.051809000001597
Episode 2361	Average Score: 0.01
epsilon: 3.0511230000015943
Episode 2362	Average Score: 0.01
epsilon: 3.0504370000015917
Episode 2363	Average Score: 0.01
epsilon: 3.049751000001589
Episode 2364	Average Score: 0.01
epsilon: 3.0490650000015864
Episode 2365	Average Score: 0.01
epsilon: 3.048379000001584
Episode 2366	Average Score: 0.01
epsilon: 3.046860000001578
Episode 2367	Average Score: 0.01
epsilon: 3.0461740000015753
Episode 2368	Average Score: 0.01
epsilon: 3.0454390000015725
Episode 2369	Average Score: 0.01
epsilon: 3.04475300000157
Episode 2370	Average Score: 0.01
epsilon: 3.0440670000015673
Episode 2371	Average Score: 0.01
epsilon: 3.0433810000015646
Episode 2372	Average Score: 0.01
epsilon: 3.042695000001562
Episode 2373	Average Score: 0.01
epsilon: 3.0420090000015594
Episode 2374	Average Score: 0.01
epsilon: 3.0413230000015568
Episode 2375	Average Score: 0.01
epsilon: 3.040637000001554
Episode 2376	Average Score: 0.01
epsilon: 3.0399510000015515
Episode 2377	Average Score: 0.01
epsilon: 3.0392160000015487
Episode 2378	Average Score: 0.01
epsilon: 3.038530000001546
Episode 2379	Average Score: 0.01
epsilon: 3.0378440000015434
Episode 2380	Average Score: 0.01
epsilon: 3.037158000001541
Episode 2381	Average Score: 0.01
epsilon: 3.036472000001538
Episode 2382	Average Score: 0.01
epsilon: 3.0357860000015355
Episode 2383	Average Score: 0.01
epsilon: 3.035100000001533
Episode 2384	Average Score: 0.01
epsilon: 3.03436500000153
Episode 2385	Average Score: 0.01
epsilon: 3.032797000001524
Episode 2386	Average Score: 0.01
epsilon: 3.0321110000015215
Episode 2387	Average Score: 0.01
epsilon: 3.031425000001519
Episode 2388	Average Score: 0.01
epsilon: 3.029955000001513
Episode 2389	Average Score: 0.01
epsilon: 3.028387000001507
Episode 2390	Average Score: 0.01
epsilon: 3.0277010000015045
Episode 2391	Average Score: 0.01
epsilon: 3.027015000001502
Episode 2392	Average Score: 0.01
epsilon: 3.0263290000014993
Episode 2393	Average Score: 0.01
epsilon: 3.0256430000014967
Episode 2394	Average Score: 0.01
epsilon: 3.0248590000014937
Episode 2395	Average Score: 0.01
epsilon: 3.024173000001491
Episode 2396	Average Score: 0.01
epsilon: 3.0234870000014884
Episode 2397	Average Score: 0.01
epsilon: 3.0228010000014858
Episode 2398	Average Score: 0.01
epsilon: 3.022066000001483
Episode 2399	Average Score: 0.01
epsilon: 3.0213800000014803
Episode 2400	Average Score: 0.01
epsilon: 3.0206940000014777
Episode 2401	Average Score: 0.01
epsilon: 3.020008000001475
Episode 2402	Average Score: 0.01
epsilon: 3.0192730000014723
Episode 2403	Average Score: 0.01
epsilon: 3.0185870000014696
Episode 2404	Average Score: 0.01
epsilon: 3.0178030000014666
Episode 2405	Average Score: 0.01
epsilon: 3.017117000001464
Episode 2406	Average Score: 0.01
epsilon: 3.0164310000014614
Episode 2407	Average Score: 0.01
epsilon: 3.0157450000014587
Episode 2408	Average Score: 0.01
epsilon: 3.015010000001456
Episode 2409	Average Score: 0.01
epsilon: 3.0135400000014503
Episode 2410	Average Score: 0.01
epsilon: 3.0119720000014443
Episode 2411	Average Score: 0.01
epsilon: 3.0112860000014416
Episode 2412	Average Score: 0.01
epsilon: 3.010600000001439
Episode 2413	Average Score: 0.01
epsilon: 3.0099140000014364
Episode 2414	Average Score: 0.01
epsilon: 3.0091790000014336
Episode 2415	Average Score: 0.01
epsilon: 3.008493000001431
Episode 2416	Average Score: 0.01
epsilon: 3.0078070000014283
Episode 2417	Average Score: 0.01
epsilon: 3.0071210000014257
Episode 2418	Average Score: 0.01
epsilon: 3.006435000001423
Episode 2419	Average Score: 0.01
epsilon: 3.0049650000014174
Episode 2420	Average Score: 0.01
epsilon: 3.004279000001415
Episode 2421	Average Score: 0.01
epsilon: 3.003593000001412
Episode 2422	Average Score: 0.01
epsilon: 3.0029070000014095
Episode 2423	Average Score: 0.01
epsilon: 3.002221000001407
Episode 2424	Average Score: 0.01
epsilon: 3.0015350000014043
Episode 2425	Average Score: 0.01
epsilon: 3.0008000000014015
Episode 2426	Average Score: 0.01
epsilon: 3.000114000001399
Episode 2427	Average Score: 0.01
epsilon: 2.999428000001396
Episode 2428	Average Score: 0.01
epsilon: 2.9987420000013936
Episode 2429	Average Score: 0.01
epsilon: 2.998056000001391
Episode 2430	Average Score: 0.01
epsilon: 2.996488000001385
Episode 2431	Average Score: 0.01
epsilon: 2.9958020000013823
Episode 2432	Average Score: 0.01
epsilon: 2.9951160000013797
Episode 2433	Average Score: 0.01
epsilon: 2.994430000001377
Episode 2434	Average Score: 0.01
epsilon: 2.9937440000013744
Episode 2435	Average Score: 0.01
epsilon: 2.993058000001372
Episode 2436	Average Score: 0.01
epsilon: 2.992323000001369
Episode 2437	Average Score: 0.01
epsilon: 2.9916370000013663
Episode 2438	Average Score: 0.01
epsilon: 2.9900690000013603
Episode 2439	Average Score: 0.01
epsilon: 2.9893830000013577
Episode 2440	Average Score: 0.01
epsilon: 2.988697000001355
Episode 2441	Average Score: 0.01
epsilon: 2.9880110000013524
Episode 2442	Average Score: 0.01
epsilon: 2.98732500000135
Episode 2443	Average Score: 0.01
epsilon: 2.986590000001347
Episode 2444	Average Score: 0.01
epsilon: 2.985757000001344
Episode 2445	Average Score: 0.01
epsilon: 2.985071000001341
Episode 2446	Average Score: 0.01
epsilon: 2.9843360000013384
Episode 2447	Average Score: 0.01
epsilon: 2.9836010000013355
Episode 2448	Average Score: 0.01
epsilon: 2.982915000001333
Episode 2449	Average Score: 0.01
epsilon: 2.9814450000013273
Episode 2450	Average Score: 0.01
epsilon: 2.980808000001325
Episode 2451	Average Score: 0.01
epsilon: 2.9791910000013186
Episode 2452	Average Score: 0.01
epsilon: 2.978554000001316
Episode 2453	Average Score: 0.01
epsilon: 2.9770350000013104
Episode 2454	Average Score: 0.01
epsilon: 2.9763490000013078
Episode 2455	Average Score: 0.01
epsilon: 2.975663000001305
Episode 2456	Average Score: 0.01
epsilon: 2.9749770000013025
Episode 2457	Average Score: 0.01
epsilon: 2.9742910000013
Episode 2458	Average Score: 0.01
epsilon: 2.9736050000012972
Episode 2459	Average Score: 0.01
epsilon: 2.9729190000012946
Episode 2460	Average Score: 0.01
epsilon: 2.972233000001292
Episode 2461	Average Score: 0.01
epsilon: 2.9715470000012894
Episode 2462	Average Score: 0.01
epsilon: 2.9708120000012865
Episode 2463	Average Score: 0.01
epsilon: 2.970126000001284
Episode 2464	Average Score: 0.01
epsilon: 2.9694890000012815
Episode 2465	Average Score: 0.01
epsilon: 2.9687540000012786
Episode 2466	Average Score: 0.01
epsilon: 2.967284000001273
Episode 2467	Average Score: 0.01
epsilon: 2.9665980000012704
Episode 2468	Average Score: 0.01
epsilon: 2.9659120000012678
Episode 2469	Average Score: 0.01
epsilon: 2.965226000001265
Episode 2470	Average Score: 0.01
epsilon: 2.9644910000012623
Episode 2471	Average Score: 0.01
epsilon: 2.9638050000012597
Episode 2472	Average Score: 0.01
epsilon: 2.963119000001257
Episode 2473	Average Score: 0.01
epsilon: 2.9624330000012544
Episode 2474	Average Score: 0.01
epsilon: 2.961747000001252
Episode 2475	Average Score: 0.01
epsilon: 2.960963000001249
Episode 2476	Average Score: 0.01
epsilon: 2.960277000001246
Episode 2477	Average Score: 0.01
epsilon: 2.9595910000012435
Episode 2478	Average Score: 0.01
epsilon: 2.958905000001241
Episode 2479	Average Score: 0.01
epsilon: 2.9582190000012383
Episode 2480	Average Score: 0.01
epsilon: 2.955818000001229
Episode 2481	Average Score: 0.01
epsilon: 2.9551320000012264
Episode 2482	Average Score: 0.01
epsilon: 2.9543970000012236
Episode 2483	Average Score: 0.01
epsilon: 2.953711000001221
Episode 2484	Average Score: 0.01
epsilon: 2.9530250000012184
Episode 2485	Average Score: 0.01
epsilon: 2.9523390000012157
Episode 2486	Average Score: 0.01
epsilon: 2.95082000000121
Episode 2487	Average Score: 0.01
epsilon: 2.9501340000012073
Episode 2488	Average Score: 0.01
epsilon: 2.9494480000012047
Episode 2489	Average Score: 0.01
epsilon: 2.948762000001202
Episode 2490	Average Score: 0.01
epsilon: 2.9480760000011994
Episode 2491	Average Score: 0.01
epsilon: 2.9473900000011968
Episode 2492	Average Score: 0.01
epsilon: 2.946704000001194
Episode 2493	Average Score: 0.01
epsilon: 2.9460180000011915
Episode 2494	Average Score: 0.01
epsilon: 2.9444500000011855
Episode 2495	Average Score: 0.01
epsilon: 2.943764000001183
Episode 2496	Average Score: 0.01
epsilon: 2.941216000001173
Episode 2497	Average Score: 0.01
epsilon: 2.9404810000011703
Episode 2498	Average Score: 0.01
epsilon: 2.9389130000011643
Episode 2499	Average Score: 0.01
epsilon: 2.9382270000011617
Episode 2500	Average Score: 0.01
epsilon: 2.937492000001159
Episode 2501	Average Score: 0.01
epsilon: 2.936806000001156
Episode 2502	Average Score: 0.01
epsilon: 2.9361200000011536
Episode 2503	Average Score: 0.01
epsilon: 2.935434000001151
Episode 2504	Average Score: 0.01
epsilon: 2.9347480000011483
Episode 2505	Average Score: 0.01
epsilon: 2.9339640000011453
Episode 2506	Average Score: 0.01
epsilon: 2.9332780000011427
Episode 2507	Average Score: 0.01
epsilon: 2.93259200000114
Episode 2508	Average Score: 0.01
epsilon: 2.9319060000011374
Episode 2509	Average Score: 0.01
epsilon: 2.931269000001135
Episode 2510	Average Score: 0.01
epsilon: 2.930534000001132
Episode 2511	Average Score: 0.01
epsilon: 2.9298480000011295
Episode 2512	Average Score: 0.01
epsilon: 2.929162000001127
Episode 2513	Average Score: 0.01
epsilon: 2.9284760000011243
Episode 2514	Average Score: 0.01
epsilon: 2.9277900000011217
Episode 2515	Average Score: 0.01
epsilon: 2.927153000001119
Episode 2516	Average Score: 0.01
epsilon: 2.9264180000011164
Episode 2517	Average Score: 0.01
epsilon: 2.9257320000011138
Episode 2518	Average Score: 0.01
epsilon: 2.925046000001111
Episode 2519	Average Score: 0.01
epsilon: 2.9243600000011085
Episode 2520	Average Score: 0.01
epsilon: 2.923674000001106
Episode 2521	Average Score: 0.01
epsilon: 2.9221060000011
Episode 2522	Average Score: 0.01
epsilon: 2.9214200000010973
Episode 2523	Average Score: 0.01
epsilon: 2.9207340000010946
Episode 2524	Average Score: 0.01
epsilon: 2.920048000001092
Episode 2525	Average Score: 0.01
epsilon: 2.9193620000010894
Episode 2526	Average Score: 0.01
epsilon: 2.9186760000010867
Episode 2527	Average Score: 0.01
epsilon: 2.917990000001084
Episode 2528	Average Score: 0.01
epsilon: 2.9173040000010815
Episode 2529	Average Score: 0.01
epsilon: 2.916618000001079
Episode 2530	Average Score: 0.01
epsilon: 2.915050000001073
Episode 2531	Average Score: 0.01
epsilon: 2.91431500000107
Episode 2532	Average Score: 0.01
epsilon: 2.9136290000010674
Episode 2533	Average Score: 0.01
epsilon: 2.912992000001065
Episode 2534	Average Score: 0.01
epsilon: 2.9123060000010623
Episode 2535	Average Score: 0.01
epsilon: 2.910689000001056
Episode 2536	Average Score: 0.01
epsilon: 2.9100030000010535
Episode 2537	Average Score: 0.01
epsilon: 2.908337000001047
Episode 2538	Average Score: 0.01
epsilon: 2.9076510000010445
Episode 2539	Average Score: 0.01
epsilon: 2.906965000001042
Episode 2540	Average Score: 0.01
epsilon: 2.9062790000010392
Episode 2541	Average Score: 0.01
epsilon: 2.9055930000010366
Episode 2542	Average Score: 0.01
epsilon: 2.904907000001034
Episode 2543	Average Score: 0.01
epsilon: 2.9042210000010313
Episode 2544	Average Score: 0.01
epsilon: 2.9035350000010287
Episode 2545	Average Score: 0.01
epsilon: 2.902849000001026
Episode 2546	Average Score: 0.01
epsilon: 2.9021140000010233
Episode 2547	Average Score: 0.01
epsilon: 2.9014280000010206
Episode 2548	Average Score: 0.01
epsilon: 2.900742000001018
Episode 2549	Average Score: 0.01
epsilon: 2.9000560000010154
Episode 2550	Average Score: 0.01
epsilon: 2.8985370000010096
Episode 2551	Average Score: 0.01
epsilon: 2.8978020000010067
Episode 2552	Average Score: 0.01
epsilon: 2.897116000001004
Episode 2553	Average Score: 0.01
epsilon: 2.8955970000009983
Episode 2554	Average Score: 0.01
epsilon: 2.8930490000009885
Episode 2555	Average Score: 0.01
epsilon: 2.8923140000009857
Episode 2556	Average Score: 0.01
epsilon: 2.891628000000983
Episode 2557	Average Score: 0.01
epsilon: 2.8901580000009774
Episode 2558	Average Score: 0.01
epsilon: 2.8893250000009743
Episode 2559	Average Score: 0.01
epsilon: 2.8885900000009714
Episode 2560	Average Score: 0.01
epsilon: 2.887904000000969
Episode 2561	Average Score: 0.01
epsilon: 2.886434000000963
Episode 2562	Average Score: 0.01
epsilon: 2.8857480000009605
Episode 2563	Average Score: 0.01
epsilon: 2.885062000000958
Episode 2564	Average Score: 0.01
epsilon: 2.8843760000009553
Episode 2565	Average Score: 0.01
epsilon: 2.8836900000009527
Episode 2566	Average Score: 0.01
epsilon: 2.88305300000095
Episode 2567	Average Score: 0.01
epsilon: 2.8823670000009476
Episode 2568	Average Score: 0.01
epsilon: 2.881681000000945
Episode 2569	Average Score: 0.01
epsilon: 2.8809950000009423
Episode 2570	Average Score: 0.01
epsilon: 2.8802600000009395
Episode 2571	Average Score: 0.01
epsilon: 2.879574000000937
Episode 2572	Average Score: 0.01
epsilon: 2.8788880000009343
Episode 2573	Average Score: 0.01
epsilon: 2.8782020000009316
Episode 2574	Average Score: 0.01
epsilon: 2.877516000000929
Episode 2575	Average Score: 0.01
epsilon: 2.8766340000009256
Episode 2576	Average Score: 0.01
epsilon: 2.875948000000923
Episode 2577	Average Score: 0.01
epsilon: 2.87521300000092
Episode 2578	Average Score: 0.01
epsilon: 2.8745270000009175
Episode 2579	Average Score: 0.01
epsilon: 2.873841000000915
Episode 2580	Average Score: 0.01
epsilon: 2.8731550000009123
Episode 2581	Average Score: 0.01
epsilon: 2.8724690000009097
Episode 2582	Average Score: 0.01
epsilon: 2.871783000000907
Episode 2583	Average Score: 0.01
epsilon: 2.8710970000009044
Episode 2584	Average Score: 0.01
epsilon: 2.869480000000898
Episode 2585	Average Score: 0.01
epsilon: 2.8687450000008954
Episode 2586	Average Score: 0.01
epsilon: 2.8680590000008928
Episode 2587	Average Score: 0.01
epsilon: 2.86732400000089
Episode 2588	Average Score: 0.01
epsilon: 2.8666380000008873
Episode 2589	Average Score: 0.01
epsilon: 2.8659520000008847
Episode 2590	Average Score: 0.01
epsilon: 2.865266000000882
Episode 2591	Average Score: 0.01
epsilon: 2.8645800000008794
Episode 2592	Average Score: 0.01
epsilon: 2.8638450000008766
Episode 2593	Average Score: 0.01
epsilon: 2.863159000000874
Episode 2594	Average Score: 0.01
epsilon: 2.8624730000008713
Episode 2595	Average Score: 0.01
epsilon: 2.8617870000008687
Episode 2596	Average Score: 0.01
epsilon: 2.861101000000866
Episode 2597	Average Score: 0.01
epsilon: 2.8604150000008635
Episode 2598	Average Score: 0.01
epsilon: 2.8596800000008606
Episode 2599	Average Score: 0.01
epsilon: 2.858994000000858
Episode 2600	Average Score: 0.01
epsilon: 2.8583080000008554
Episode 2601	Average Score: 0.01
epsilon: 2.8576220000008528
Episode 2602	Average Score: 0.01
epsilon: 2.85693600000085
Episode 2603	Average Score: 0.01
epsilon: 2.8562500000008475
Episode 2604	Average Score: 0.01
epsilon: 2.855564000000845
Episode 2605	Average Score: 0.01
epsilon: 2.854829000000842
Episode 2606	Average Score: 0.01
epsilon: 2.8541430000008394
Episode 2607	Average Score: 0.01
epsilon: 2.853457000000837
Episode 2608	Average Score: 0.01
epsilon: 2.852771000000834
Episode 2609	Average Score: 0.01
epsilon: 2.8513500000008287
Episode 2610	Average Score: 0.01
epsilon: 2.850615000000826
Episode 2611	Average Score: 0.01
epsilon: 2.8499290000008233
Episode 2612	Average Score: 0.01
epsilon: 2.8492430000008206
Episode 2613	Average Score: 0.01
epsilon: 2.848557000000818
Episode 2614	Average Score: 0.01
epsilon: 2.8478710000008154
Episode 2615	Average Score: 0.01
epsilon: 2.8471360000008126
Episode 2616	Average Score: 0.01
epsilon: 2.84645000000081
Episode 2617	Average Score: 0.01
epsilon: 2.8457640000008073
Episode 2618	Average Score: 0.01
epsilon: 2.8442940000008017
Episode 2619	Average Score: 0.01
epsilon: 2.842775000000796
Episode 2620	Average Score: 0.01
epsilon: 2.8420890000007932
Episode 2621	Average Score: 0.01
epsilon: 2.8414030000007906
Episode 2622	Average Score: 0.01
epsilon: 2.840717000000788
Episode 2623	Average Score: 0.01
epsilon: 2.8400310000007853
Episode 2624	Average Score: 0.01
epsilon: 2.8392960000007825
Episode 2625	Average Score: 0.01
epsilon: 2.83861000000078
Episode 2626	Average Score: 0.01
epsilon: 2.8379240000007773
Episode 2627	Average Score: 0.01
epsilon: 2.8372380000007746
Episode 2628	Average Score: 0.01
epsilon: 2.8356700000007686
Episode 2629	Average Score: 0.01
epsilon: 2.834984000000766
Episode 2630	Average Score: 0.01
epsilon: 2.8342980000007634
Episode 2631	Average Score: 0.01
epsilon: 2.8335630000007606
Episode 2632	Average Score: 0.01
epsilon: 2.832877000000758
Episode 2633	Average Score: 0.01
epsilon: 2.8321910000007553
Episode 2634	Average Score: 0.01
epsilon: 2.8315050000007527
Episode 2635	Average Score: 0.01
epsilon: 2.83081900000075
Episode 2636	Average Score: 0.01
epsilon: 2.8301330000007474
Episode 2637	Average Score: 0.01
epsilon: 2.829447000000745
Episode 2638	Average Score: 0.01
epsilon: 2.828712000000742
Episode 2639	Average Score: 0.01
epsilon: 2.8280260000007393
Episode 2640	Average Score: 0.01
epsilon: 2.827389000000737
Episode 2641	Average Score: 0.01
epsilon: 2.826654000000734
Episode 2642	Average Score: 0.01
epsilon: 2.8259680000007315
Episode 2643	Average Score: 0.01
epsilon: 2.825282000000729
Episode 2644	Average Score: 0.01
epsilon: 2.824596000000726
Episode 2645	Average Score: 0.01
epsilon: 2.8239100000007236
Episode 2646	Average Score: 0.01
epsilon: 2.823224000000721
Episode 2647	Average Score: 0.01
epsilon: 2.8225870000007185
Episode 2648	Average Score: 0.01
epsilon: 2.8218520000007157
Episode 2649	Average Score: 0.01
epsilon: 2.8183730000007023
Episode 2650	Average Score: 0.01
epsilon: 2.816952000000697
Episode 2651	Average Score: 0.01
epsilon: 2.8162660000006943
Episode 2652	Average Score: 0.01
epsilon: 2.815384000000691
Episode 2653	Average Score: 0.01
epsilon: 2.814649000000688
Episode 2654	Average Score: 0.01
epsilon: 2.8140120000006856
Episode 2655	Average Score: 0.01
epsilon: 2.813277000000683
Episode 2656	Average Score: 0.01
epsilon: 2.8106310000006727
Episode 2657	Average Score: 0.01
epsilon: 2.809112000000667
Episode 2658	Average Score: 0.01
epsilon: 2.8084260000006642
Episode 2659	Average Score: 0.01
epsilon: 2.8077400000006616
Episode 2660	Average Score: 0.01
epsilon: 2.807054000000659
Episode 2661	Average Score: 0.01
epsilon: 2.8063680000006563
Episode 2662	Average Score: 0.01
epsilon: 2.8056330000006535
Episode 2663	Average Score: 0.01
epsilon: 2.804947000000651
Episode 2664	Average Score: 0.01
epsilon: 2.8042610000006483
Episode 2665	Average Score: 0.01
epsilon: 2.8027910000006426
Episode 2666	Average Score: 0.01
epsilon: 2.80210500000064
Episode 2667	Average Score: 0.01
epsilon: 2.800586000000634
Episode 2668	Average Score: 0.01
epsilon: 2.7999000000006316
Episode 2669	Average Score: 0.01
epsilon: 2.799214000000629
Episode 2670	Average Score: 0.01
epsilon: 2.7985280000006263
Episode 2671	Average Score: 0.01
epsilon: 2.7978420000006237
Episode 2672	Average Score: 0.01
epsilon: 2.797156000000621
Episode 2673	Average Score: 0.01
epsilon: 2.796421000000618
Episode 2674	Average Score: 0.01
epsilon: 2.7957350000006156
Episode 2675	Average Score: 0.01
epsilon: 2.7941670000006096
Episode 2676	Average Score: 0.01
epsilon: 2.793481000000607
Episode 2677	Average Score: 0.01
epsilon: 2.7927950000006043
Episode 2678	Average Score: 0.01
epsilon: 2.7921090000006017
Episode 2679	Average Score: 0.01
epsilon: 2.791423000000599
Episode 2680	Average Score: 0.01
epsilon: 2.7906880000005962
Episode 2681	Average Score: 0.01
epsilon: 2.790051000000594
Episode 2682	Average Score: 0.01
epsilon: 2.789316000000591
Episode 2683	Average Score: 0.01
epsilon: 2.7886300000005884
Episode 2684	Average Score: 0.01
epsilon: 2.7879440000005857
Episode 2685	Average Score: 0.01
epsilon: 2.787258000000583
Episode 2686	Average Score: 0.01
epsilon: 2.7865720000005805
Episode 2687	Average Score: 0.01
epsilon: 2.785886000000578
Episode 2688	Average Score: 0.01
epsilon: 2.785200000000575
Episode 2689	Average Score: 0.01
epsilon: 2.7845140000005726
Episode 2690	Average Score: 0.01
epsilon: 2.78382800000057
Episode 2691	Average Score: 0.01
epsilon: 2.7813290000005604
Episode 2692	Average Score: 0.01
epsilon: 2.7805940000005576
Episode 2693	Average Score: 0.01
epsilon: 2.779908000000555
Episode 2694	Average Score: 0.01
epsilon: 2.7792220000005523
Episode 2695	Average Score: 0.01
epsilon: 2.7784870000005495
Episode 2696	Average Score: 0.01
epsilon: 2.777801000000547
Episode 2697	Average Score: 0.01
epsilon: 2.7771150000005442
Episode 2698	Average Score: 0.01
epsilon: 2.7764290000005416
Episode 2699	Average Score: 0.01
epsilon: 2.775743000000539
Episode 2700	Average Score: 0.01
epsilon: 2.775008000000536
Episode 2701	Average Score: 0.01
epsilon: 2.773881000000532
Episode 2702	Average Score: 0.01
epsilon: 2.773195000000529
Episode 2703	Average Score: 0.01
epsilon: 2.7725090000005266
Episode 2704	Average Score: 0.01
epsilon: 2.771823000000524
Episode 2705	Average Score: 0.01
epsilon: 2.7711370000005213
Episode 2706	Average Score: 0.01
epsilon: 2.7704510000005187
Episode 2707	Average Score: 0.01
epsilon: 2.769716000000516
Episode 2708	Average Score: 0.01
epsilon: 2.7690790000005134
Episode 2709	Average Score: 0.01
epsilon: 2.7675110000005074
Episode 2710	Average Score: 0.01
epsilon: 2.766825000000505
Episode 2711	Average Score: 0.01
epsilon: 2.766139000000502
Episode 2712	Average Score: 0.01
epsilon: 2.7654530000004995
Episode 2713	Average Score: 0.01
epsilon: 2.764767000000497
Episode 2714	Average Score: 0.01
epsilon: 2.763199000000491
Episode 2715	Average Score: 0.02
epsilon: 2.7615820000004847
Episode 2716	Average Score: 0.02
epsilon: 2.760847000000482
Episode 2717	Average Score: 0.02
epsilon: 2.7601610000004793
Episode 2718	Average Score: 0.02
epsilon: 2.7586910000004736
Episode 2719	Average Score: 0.01
epsilon: 2.758005000000471
Episode 2720	Average Score: 0.01
epsilon: 2.7573190000004684
Episode 2721	Average Score: 0.01
epsilon: 2.7565840000004656
Episode 2722	Average Score: 0.01
epsilon: 2.755898000000463
Episode 2723	Average Score: 0.01
epsilon: 2.7552120000004603
Episode 2724	Average Score: 0.01
epsilon: 2.7545260000004577
Episode 2725	Average Score: 0.01
epsilon: 2.753840000000455
Episode 2726	Average Score: 0.01
epsilon: 2.7531540000004524
Episode 2727	Average Score: 0.01
epsilon: 2.7524680000004498
Episode 2728	Average Score: 0.01
epsilon: 2.751733000000447
Episode 2729	Average Score: 0.01
epsilon: 2.7510470000004443
Episode 2730	Average Score: 0.01
epsilon: 2.7503610000004417
Episode 2731	Average Score: 0.01
epsilon: 2.749675000000439
Episode 2732	Average Score: 0.01
epsilon: 2.7481560000004333
Episode 2733	Average Score: 0.01
epsilon: 2.7474700000004306
Episode 2734	Average Score: 0.01
epsilon: 2.7466860000004276
Episode 2735	Average Score: 0.01
epsilon: 2.746049000000425
Episode 2736	Average Score: 0.01
epsilon: 2.7453630000004225
Episode 2737	Average Score: 0.01
epsilon: 2.74467700000042
Episode 2738	Average Score: 0.01
epsilon: 2.7439910000004173
Episode 2739	Average Score: 0.01
epsilon: 2.7433050000004147
Episode 2740	Average Score: 0.01
epsilon: 2.742619000000412
Episode 2741	Average Score: 0.01
epsilon: 2.7419330000004094
Episode 2742	Average Score: 0.01
epsilon: 2.7412470000004068
Episode 2743	Average Score: 0.01
epsilon: 2.740561000000404
Episode 2744	Average Score: 0.01
epsilon: 2.7398260000004013
Episode 2745	Average Score: 0.01
epsilon: 2.739189000000399
Episode 2746	Average Score: 0.01
epsilon: 2.7385030000003963
Episode 2747	Average Score: 0.02
epsilon: 2.7370330000003906
Episode 2748	Average Score: 0.02
epsilon: 2.736347000000388
Episode 2749	Average Score: 0.01
epsilon: 2.735612000000385
Episode 2750	Average Score: 0.01
epsilon: 2.7349260000003826
Episode 2751	Average Score: 0.01
epsilon: 2.73428900000038
Episode 2752	Average Score: 0.01
epsilon: 2.7335540000003773
Episode 2753	Average Score: 0.01
epsilon: 2.732917000000375
Episode 2754	Average Score: 0.01
epsilon: 2.7322310000003722
Episode 2755	Average Score: 0.01
epsilon: 2.7315450000003696
Episode 2756	Average Score: 0.01
epsilon: 2.7301730000003643
Episode 2757	Average Score: 0.01
epsilon: 2.7294380000003615
Episode 2758	Average Score: 0.01
epsilon: 2.7287030000003587
Episode 2759	Average Score: 0.01
epsilon: 2.728017000000356
Episode 2760	Average Score: 0.01
epsilon: 2.7273310000003534
Episode 2761	Average Score: 0.01
epsilon: 2.726694000000351
Episode 2762	Average Score: 0.01
epsilon: 2.7260080000003484
Episode 2763	Average Score: 0.01
epsilon: 2.7253220000003457
Episode 2764	Average Score: 0.01
epsilon: 2.724636000000343
Episode 2765	Average Score: 0.01
epsilon: 2.7239500000003405
Episode 2766	Average Score: 0.01
epsilon: 2.723264000000338
Episode 2767	Average Score: 0.01
epsilon: 2.722529000000335
Episode 2768	Average Score: 0.01
epsilon: 2.7218430000003324
Episode 2769	Average Score: 0.01
epsilon: 2.72115700000033
Episode 2770	Average Score: 0.01
epsilon: 2.720471000000327
Episode 2771	Average Score: 0.01
epsilon: 2.7197850000003245
Episode 2772	Average Score: 0.01
epsilon: 2.7190500000003217
Episode 2773	Average Score: 0.01
epsilon: 2.718364000000319
Episode 2774	Average Score: 0.01
epsilon: 2.7176780000003165
Episode 2775	Average Score: 0.01
epsilon: 2.716992000000314
Episode 2776	Average Score: 0.01
epsilon: 2.716306000000311
Episode 2777	Average Score: 0.01
epsilon: 2.7155710000003084
Episode 2778	Average Score: 0.01
epsilon: 2.712288000000296
Episode 2779	Average Score: 0.01
epsilon: 2.711602000000293
Episode 2780	Average Score: 0.01
epsilon: 2.7109160000002905
Episode 2781	Average Score: 0.01
epsilon: 2.7101810000002877
Episode 2782	Average Score: 0.01
epsilon: 2.709495000000285
Episode 2783	Average Score: 0.01
epsilon: 2.7088090000002825
Episode 2784	Average Score: 0.01
epsilon: 2.70812300000028
Episode 2785	Average Score: 0.01
epsilon: 2.707437000000277
Episode 2786	Average Score: 0.01
epsilon: 2.7067510000002746
Episode 2787	Average Score: 0.01
epsilon: 2.706065000000272
Episode 2788	Average Score: 0.01
epsilon: 2.7053790000002693
Episode 2789	Average Score: 0.01
epsilon: 2.7046930000002667
Episode 2790	Average Score: 0.01
epsilon: 2.703958000000264
Episode 2791	Average Score: 0.01
epsilon: 2.702439000000258
Episode 2792	Average Score: 0.01
epsilon: 2.7017530000002554
Episode 2793	Average Score: 0.01
epsilon: 2.701067000000253
Episode 2794	Average Score: 0.01
epsilon: 2.699548000000247
Episode 2795	Average Score: 0.01
epsilon: 2.6988620000002443
Episode 2796	Average Score: 0.01
epsilon: 2.6981270000002415
Episode 2797	Average Score: 0.01
epsilon: 2.697441000000239
Episode 2798	Average Score: 0.01
epsilon: 2.6967550000002363
Episode 2799	Average Score: 0.01
epsilon: 2.6951870000002303
Episode 2800	Average Score: 0.01
epsilon: 2.6945010000002276
Episode 2801	Average Score: 0.01
epsilon: 2.693815000000225
Episode 2802	Average Score: 0.01
epsilon: 2.6931290000002224
Episode 2803	Average Score: 0.01
epsilon: 2.6924430000002197
Episode 2804	Average Score: 0.01
epsilon: 2.691757000000217
Episode 2805	Average Score: 0.01
epsilon: 2.6910220000002143
Episode 2806	Average Score: 0.01
epsilon: 2.6903360000002117
Episode 2807	Average Score: 0.01
epsilon: 2.689650000000209
Episode 2808	Average Score: 0.01
epsilon: 2.6889640000002064
Episode 2809	Average Score: 0.01
epsilon: 2.688278000000204
Episode 2810	Average Score: 0.01
epsilon: 2.687592000000201
Episode 2811	Average Score: 0.01
epsilon: 2.6868570000001983
Episode 2812	Average Score: 0.01
epsilon: 2.6852890000001923
Episode 2813	Average Score: 0.01
epsilon: 2.68465200000019
Episode 2814	Average Score: 0.01
epsilon: 2.6839660000001873
Episode 2815	Average Score: 0.01
epsilon: 2.6832800000001846
Episode 2816	Average Score: 0.01
epsilon: 2.682594000000182
Episode 2817	Average Score: 0.01
epsilon: 2.681810000000179
Episode 2818	Average Score: 0.01
epsilon: 2.681075000000176
Episode 2819	Average Score: 0.01
epsilon: 2.6803400000001734
Episode 2820	Average Score: 0.01
epsilon: 2.6787720000001674
Episode 2821	Average Score: 0.01
epsilon: 2.6780860000001647
Episode 2822	Average Score: 0.01
epsilon: 2.677400000000162
Episode 2823	Average Score: 0.01
epsilon: 2.676616000000159
Episode 2824	Average Score: 0.01
epsilon: 2.675244000000154
Episode 2825	Average Score: 0.01
epsilon: 2.673725000000148
Episode 2826	Average Score: 0.01
epsilon: 2.6730390000001454
Episode 2827	Average Score: 0.01
epsilon: 2.671422000000139
Episode 2828	Average Score: 0.01
epsilon: 2.6707360000001366
Episode 2829	Average Score: 0.01
epsilon: 2.6692170000001307
Episode 2830	Average Score: 0.01
epsilon: 2.6665710000001206
Episode 2831	Average Score: 0.01
epsilon: 2.6650520000001148
Episode 2832	Average Score: 0.01
epsilon: 2.664366000000112
Episode 2833	Average Score: 0.01
epsilon: 2.6636800000001095
Episode 2834	Average Score: 0.01
epsilon: 2.662994000000107
Episode 2835	Average Score: 0.01
epsilon: 2.6623080000001043
Episode 2836	Average Score: 0.01
epsilon: 2.6615730000001014
Episode 2837	Average Score: 0.01
epsilon: 2.660887000000099
Episode 2838	Average Score: 0.01
epsilon: 2.660201000000096
Episode 2839	Average Score: 0.01
epsilon: 2.6595150000000936
Episode 2840	Average Score: 0.01
epsilon: 2.658829000000091
Episode 2841	Average Score: 0.01
epsilon: 2.658094000000088
Episode 2842	Average Score: 0.01
epsilon: 2.6574080000000855
Episode 2843	Average Score: 0.01
epsilon: 2.656722000000083
Episode 2844	Average Score: 0.01
epsilon: 2.655154000000077
Episode 2845	Average Score: 0.01
epsilon: 2.654468000000074
Episode 2846	Average Score: 0.02
epsilon: 2.6529980000000686
Episode 2847	Average Score: 0.01
epsilon: 2.652312000000066
Episode 2848	Average Score: 0.01
epsilon: 2.6516260000000633
Episode 2849	Average Score: 0.01
epsilon: 2.6508910000000605
Episode 2850	Average Score: 0.02
epsilon: 2.647608000000048
Episode 2851	Average Score: 0.02
epsilon: 2.6469220000000453
Episode 2852	Average Score: 0.02
epsilon: 2.6462360000000427
Episode 2853	Average Score: 0.02
epsilon: 2.6454520000000397
Episode 2854	Average Score: 0.02
epsilon: 2.644766000000037
Episode 2855	Average Score: 0.02
epsilon: 2.6440800000000344
Episode 2856	Average Score: 0.02
epsilon: 2.6433940000000318
Episode 2857	Average Score: 0.02
epsilon: 2.6402090000000196
Episode 2858	Average Score: 0.02
epsilon: 2.6386410000000136
Episode 2859	Average Score: 0.02
epsilon: 2.637955000000011
Episode 2860	Average Score: 0.02
epsilon: 2.6372690000000083
Episode 2861	Average Score: 0.02
epsilon: 2.6365340000000055
Episode 2862	Average Score: 0.02
epsilon: 2.635848000000003
Episode 2863	Average Score: 0.02
epsilon: 2.6351620000000002
Episode 2864	Average Score: 0.02
epsilon: 2.6344759999999976
Episode 2865	Average Score: 0.02
epsilon: 2.631486999999986
Episode 2866	Average Score: 0.02
epsilon: 2.6308009999999835
Episode 2867	Average Score: 0.02
epsilon: 2.6300659999999807
Episode 2868	Average Score: 0.02
epsilon: 2.629379999999978
Episode 2869	Average Score: 0.02
epsilon: 2.6286939999999754
Episode 2870	Average Score: 0.02
epsilon: 2.628007999999973
Episode 2871	Average Score: 0.02
epsilon: 2.62732199999997
Episode 2872	Average Score: 0.02
epsilon: 2.6265869999999674
Episode 2873	Average Score: 0.02
epsilon: 2.6250189999999614
Episode 2874	Average Score: 0.02
epsilon: 2.6242839999999585
Episode 2875	Average Score: 0.02
epsilon: 2.623646999999956
Episode 2876	Average Score: 0.02
epsilon: 2.622813999999953
Episode 2877	Average Score: 0.02
epsilon: 2.621294999999947
Episode 2878	Average Score: 0.02
epsilon: 2.6206089999999445
Episode 2879	Average Score: 0.02
epsilon: 2.619138999999939
Episode 2880	Average Score: 0.02
epsilon: 2.618452999999936
Episode 2881	Average Score: 0.02
epsilon: 2.6177669999999336
Episode 2882	Average Score: 0.02
epsilon: 2.6169339999999304
Episode 2883	Average Score: 0.02
epsilon: 2.6162479999999277
Episode 2884	Average Score: 0.02
epsilon: 2.612915999999915
Episode 2885	Average Score: 0.02
epsilon: 2.6122789999999125
Episode 2886	Average Score: 0.02
epsilon: 2.6115439999999097
Episode 2887	Average Score: 0.02
epsilon: 2.6109069999999073
Episode 2888	Average Score: 0.02
epsilon: 2.6102209999999046
Episode 2889	Average Score: 0.02
epsilon: 2.609534999999902
Episode 2890	Average Score: 0.02
epsilon: 2.6086039999998984
Episode 2891	Average Score: 0.02
epsilon: 2.607917999999896
Episode 2892	Average Score: 0.02
epsilon: 2.607182999999893
Episode 2893	Average Score: 0.02
epsilon: 2.605614999999887
Episode 2894	Average Score: 0.02
epsilon: 2.603997999999881
Episode 2895	Average Score: 0.02
epsilon: 2.6023809999998746
Episode 2896	Average Score: 0.02
epsilon: 2.601694999999872
Episode 2897	Average Score: 0.02
epsilon: 2.6010089999998693
Episode 2898	Average Score: 0.02
epsilon: 2.6003229999998667
Episode 2899	Average Score: 0.02
epsilon: 2.5987059999998605
Episode 2900	Average Score: 0.02
epsilon: 2.598019999999858
Episode 2901	Average Score: 0.02
epsilon: 2.5973339999998553
Episode 2902	Average Score: 0.02
epsilon: 2.5966479999998526
Episode 2903	Average Score: 0.02
epsilon: 2.5950309999998464
Episode 2904	Average Score: 0.02
epsilon: 2.594344999999844
Episode 2905	Average Score: 0.02
epsilon: 2.593658999999841
Episode 2906	Average Score: 0.02
epsilon: 2.5930219999998387
Episode 2907	Average Score: 0.02
epsilon: 2.592286999999836
Episode 2908	Average Score: 0.02
epsilon: 2.589493999999825
Episode 2909	Average Score: 0.02
epsilon: 2.588611999999822
Episode 2910	Average Score: 0.02
epsilon: 2.587925999999819
Episode 2911	Average Score: 0.02
epsilon: 2.5872399999998166
Episode 2912	Average Score: 0.02
epsilon: 2.5856719999998106
Episode 2913	Average Score: 0.02
epsilon: 2.5849369999998077
Episode 2914	Average Score: 0.02
epsilon: 2.5842999999998053
Episode 2915	Average Score: 0.02
epsilon: 2.5835649999998025
Episode 2916	Average Score: 0.02
epsilon: 2.581065999999793
Episode 2917	Average Score: 0.02
epsilon: 2.5803799999997903
Episode 2918	Average Score: 0.02
epsilon: 2.579546999999787
Episode 2919	Average Score: 0.02
epsilon: 2.5788609999997845
Episode 2920	Average Score: 0.02
epsilon: 2.578174999999782
Episode 2921	Average Score: 0.02
epsilon: 2.577488999999779
Episode 2922	Average Score: 0.02
epsilon: 2.5752839999997708
Episode 2923	Average Score: 0.02
epsilon: 2.574597999999768
Episode 2924	Average Score: 0.02
epsilon: 2.5739119999997655
Episode 2925	Average Score: 0.02
epsilon: 2.5723439999997595
Episode 2926	Average Score: 0.02
epsilon: 2.5707759999997535
Episode 2927	Average Score: 0.02
epsilon: 2.570089999999751
Episode 2928	Average Score: 0.02
epsilon: 2.5694039999997482
Episode 2929	Average Score: 0.02
epsilon: 2.567786999999742
Episode 2930	Average Score: 0.02
epsilon: 2.5671009999997394
Episode 2931	Average Score: 0.02
epsilon: 2.5663659999997366
Episode 2932	Average Score: 0.02
epsilon: 2.565728999999734
Episode 2933	Average Score: 0.02
epsilon: 2.5649939999997313
Episode 2934	Average Score: 0.02
epsilon: 2.5643079999997287
Episode 2935	Average Score: 0.02
epsilon: 2.5628869999997232
Episode 2936	Average Score: 0.02
epsilon: 2.5622009999997206
Episode 2937	Average Score: 0.02
epsilon: 2.561514999999718
Episode 2938	Average Score: 0.02
epsilon: 2.559995999999712
Episode 2939	Average Score: 0.02
epsilon: 2.5593099999997095
Episode 2940	Average Score: 0.02
epsilon: 2.5577909999997037
Episode 2941	Average Score: 0.02
epsilon: 2.557055999999701
Episode 2942	Average Score: 0.02
epsilon: 2.556320999999698
Episode 2943	Average Score: 0.02
epsilon: 2.5556349999996955
Episode 2944	Average Score: 0.02
epsilon: 2.554997999999693
Episode 2945	Average Score: 0.02
epsilon: 2.55426299999969
Episode 2946	Average Score: 0.02
epsilon: 2.5536259999996878
Episode 2947	Average Score: 0.02
epsilon: 2.552890999999685
Episode 2948	Average Score: 0.02
epsilon: 2.5514699999996795
Episode 2949	Average Score: 0.02
epsilon: 2.550783999999677
Episode 2950	Average Score: 0.02
epsilon: 2.5500979999996742
Episode 2951	Average Score: 0.02
epsilon: 2.5494119999996716
Episode 2952	Average Score: 0.02
epsilon: 2.548676999999669
Episode 2953	Average Score: 0.02
epsilon: 2.547990999999666
Episode 2954	Average Score: 0.02
epsilon: 2.5464719999996603
Episode 2955	Average Score: 0.02
epsilon: 2.5457859999996577
Episode 2956	Average Score: 0.02
epsilon: 2.545099999999655
Episode 2957	Average Score: 0.02
epsilon: 2.5444139999996525
Episode 2958	Average Score: 0.02
epsilon: 2.5428949999996466
Episode 2959	Average Score: 0.02
epsilon: 2.542159999999644
Episode 2960	Average Score: 0.02
epsilon: 2.5407879999996386
Episode 2961	Average Score: 0.02
epsilon: 2.540101999999636
Episode 2962	Average Score: 0.02
epsilon: 2.5394159999996333
Episode 2963	Average Score: 0.02
epsilon: 2.5379459999996277
Episode 2964	Average Score: 0.02
epsilon: 2.537259999999625
Episode 2965	Average Score: 0.02
epsilon: 2.5365739999996224
Episode 2966	Average Score: 0.02
epsilon: 2.5350059999996164
Episode 2967	Average Score: 0.02
epsilon: 2.5343199999996138
Episode 2968	Average Score: 0.02
epsilon: 2.533633999999611
Episode 2969	Average Score: 0.02
epsilon: 2.5327029999996076
Episode 2970	Average Score: 0.02
epsilon: 2.532016999999605
Episode 2971	Average Score: 0.02
epsilon: 2.5313309999996023
Episode 2972	Average Score: 0.02
epsilon: 2.5297629999995963
Episode 2973	Average Score: 0.02
epsilon: 2.5290769999995937
Episode 2974	Average Score: 0.02
epsilon: 2.528390999999591
Episode 2975	Average Score: 0.02
epsilon: 2.5277049999995884
Episode 2976	Average Score: 0.02
epsilon: 2.5260879999995822
Episode 2977	Average Score: 0.02
epsilon: 2.5254019999995796
Episode 2978	Average Score: 0.02
epsilon: 2.524764999999577
Episode 2979	Average Score: 0.02
epsilon: 2.5232949999995715
Episode 2980	Average Score: 0.02
epsilon: 2.522608999999569
Episode 2981	Average Score: 0.02
epsilon: 2.521040999999563
Episode 2982	Average Score: 0.02
epsilon: 2.5204039999995604
Episode 2983	Average Score: 0.02
epsilon: 2.5196689999995576
Episode 2984	Average Score: 0.02
epsilon: 2.518982999999555
Episode 2985	Average Score: 0.02
epsilon: 2.5182969999995524
Episode 2986	Average Score: 0.02
epsilon: 2.51765999999955
Episode 2987	Average Score: 0.02
epsilon: 2.516924999999547
Episode 2988	Average Score: 0.02
epsilon: 2.5162389999995445
Episode 2989	Average Score: 0.02
epsilon: 2.515552999999542
Episode 2990	Average Score: 0.02
epsilon: 2.514866999999539
Episode 2991	Average Score: 0.02
epsilon: 2.513249999999533
Episode 2992	Average Score: 0.02
epsilon: 2.51251499999953
Episode 2993	Average Score: 0.02
epsilon: 2.5118289999995276
Episode 2994	Average Score: 0.02
epsilon: 2.511142999999525
Episode 2995	Average Score: 0.02
epsilon: 2.509623999999519
Episode 2996	Average Score: 0.02
epsilon: 2.5081049999995133
Episode 2997	Average Score: 0.02
epsilon: 2.5073699999995105
Episode 2998	Average Score: 0.02
epsilon: 2.506732999999508
Episode 2999	Average Score: 0.02
epsilon: 2.5060469999995054
Episode 3000	Average Score: 0.02
epsilon: 2.5053119999995026
Episode 3001	Average Score: 0.02
epsilon: 2.5046259999995
Episode 3002	Average Score: 0.02
epsilon: 2.5039399999994973
Episode 3003	Average Score: 0.02
epsilon: 2.5032539999994947
Episode 3004	Average Score: 0.02
epsilon: 2.501734999999489
Episode 3005	Average Score: 0.02
epsilon: 2.500999999999486
Episode 3006	Average Score: 0.02
epsilon: 2.5003139999994834
Episode 3007	Average Score: 0.02
epsilon: 2.499627999999481
Episode 3008	Average Score: 0.02
epsilon: 2.4989909999994784
Episode 3009	Average Score: 0.02
epsilon: 2.4982559999994756
Episode 3010	Average Score: 0.02
epsilon: 2.49678599999947
Episode 3011	Average Score: 0.02
epsilon: 2.4960999999994673
Episode 3012	Average Score: 0.02
epsilon: 2.4954139999994647
Episode 3013	Average Score: 0.02
epsilon: 2.494678999999462
Episode 3014	Average Score: 0.02
epsilon: 2.4930619999994557
Episode 3015	Average Score: 0.02
epsilon: 2.492375999999453
Episode 3016	Average Score: 0.02
epsilon: 2.4916899999994504
Episode 3017	Average Score: 0.02
epsilon: 2.4901219999994444
Episode 3018	Average Score: 0.02
epsilon: 2.489484999999442
Episode 3019	Average Score: 0.02
epsilon: 2.488749999999439
Episode 3020	Average Score: 0.02
epsilon: 2.486152999999429
Episode 3021	Average Score: 0.02
epsilon: 2.4854179999994264
Episode 3022	Average Score: 0.02
epsilon: 2.4847319999994237
Episode 3023	Average Score: 0.02
epsilon: 2.483996999999421
Episode 3024	Average Score: 0.02
epsilon: 2.4823309999994145
Episode 3025	Average Score: 0.02
epsilon: 2.481644999999412
Episode 3026	Average Score: 0.02
epsilon: 2.4809589999994093
Episode 3027	Average Score: 0.02
epsilon: 2.4802729999994066
Episode 3028	Average Score: 0.02
epsilon: 2.479586999999404
Episode 3029	Average Score: 0.02
epsilon: 2.4789009999994014
Episode 3030	Average Score: 0.02
epsilon: 2.4782149999993988
Episode 3031	Average Score: 0.02
epsilon: 2.477479999999396
Episode 3032	Average Score: 0.02
epsilon: 2.47591199999939
Episode 3033	Average Score: 0.02
epsilon: 2.474343999999384
Episode 3034	Average Score: 0.02
epsilon: 2.4729229999993785
Episode 3035	Average Score: 0.02
epsilon: 2.4703259999993685
Episode 3036	Average Score: 0.02
epsilon: 2.4687089999993623
Episode 3037	Average Score: 0.03
epsilon: 2.4650339999993482
Episode 3038	Average Score: 0.03
epsilon: 2.4643479999993456
Episode 3039	Average Score: 0.03
epsilon: 2.463171999999341
Episode 3040	Average Score: 0.02
epsilon: 2.4624369999993383
Episode 3041	Average Score: 0.02
epsilon: 2.4617509999993357
Episode 3042	Average Score: 0.02
epsilon: 2.461015999999333
Episode 3043	Average Score: 0.02
epsilon: 2.4603789999993304
Episode 3044	Average Score: 0.03
epsilon: 2.458761999999324
Episode 3045	Average Score: 0.03
epsilon: 2.4580269999993214
Episode 3046	Average Score: 0.03
epsilon: 2.4565569999993158
Episode 3047	Average Score: 0.03
epsilon: 2.4549399999993096
Episode 3048	Average Score: 0.03
epsilon: 2.454253999999307
Episode 3049	Average Score: 0.03
epsilon: 2.4526369999993007
Episode 3050	Average Score: 0.03
epsilon: 2.451950999999298
Episode 3051	Average Score: 0.03
epsilon: 2.4512649999992955
Episode 3052	Average Score: 0.03
epsilon: 2.4489129999992865
Episode 3053	Average Score: 0.03
epsilon: 2.448275999999284
Episode 3054	Average Score: 0.03
epsilon: 2.447540999999281
Episode 3055	Average Score: 0.03
epsilon: 2.445972999999275
Episode 3056	Average Score: 0.03
epsilon: 2.4452869999992726
Episode 3057	Average Score: 0.03
epsilon: 2.44460099999927
Episode 3058	Average Score: 0.03
epsilon: 2.4439149999992673
Episode 3059	Average Score: 0.03
epsilon: 2.4423469999992613
Episode 3060	Average Score: 0.03
epsilon: 2.4416119999992585
Episode 3061	Average Score: 0.03
epsilon: 2.440925999999256
Episode 3062	Average Score: 0.03
epsilon: 2.4402399999992532
Episode 3063	Average Score: 0.03
epsilon: 2.4395539999992506
Episode 3064	Average Score: 0.03
epsilon: 2.438867999999248
Episode 3065	Average Score: 0.03
epsilon: 2.438132999999245
Episode 3066	Average Score: 0.03
epsilon: 2.4374469999992425
Episode 3067	Average Score: 0.03
epsilon: 2.43676099999924
Episode 3068	Average Score: 0.03
epsilon: 2.4351439999992337
Episode 3069	Average Score: 0.03
epsilon: 2.434457999999231
Episode 3070	Average Score: 0.03
epsilon: 2.4337719999992284
Episode 3071	Average Score: 0.03
epsilon: 2.433085999999226
Episode 3072	Average Score: 0.03
epsilon: 2.432399999999223
Episode 3073	Average Score: 0.03
epsilon: 2.4317139999992206
Episode 3074	Average Score: 0.03
epsilon: 2.4300969999992144
Episode 3075	Average Score: 0.03
epsilon: 2.4294109999992117
Episode 3076	Average Score: 0.03
epsilon: 2.428724999999209
Episode 3077	Average Score: 0.03
epsilon: 2.4262259999991995
Episode 3078	Average Score: 0.03
epsilon: 2.425539999999197
Episode 3079	Average Score: 0.03
epsilon: 2.4246579999991935
Episode 3080	Average Score: 0.03
epsilon: 2.423971999999191
Episode 3081	Average Score: 0.02
epsilon: 2.4232859999991883
Episode 3082	Average Score: 0.02
epsilon: 2.4225999999991856
Episode 3083	Average Score: 0.02
epsilon: 2.421913999999183
Episode 3084	Average Score: 0.02
epsilon: 2.4212279999991804
Episode 3085	Average Score: 0.03
epsilon: 2.419610999999174
Episode 3086	Average Score: 0.03
epsilon: 2.4189249999991715
Episode 3087	Average Score: 0.03
epsilon: 2.418238999999169
Episode 3088	Average Score: 0.03
epsilon: 2.4175529999991663
Episode 3089	Average Score: 0.03
epsilon: 2.4168669999991637
Episode 3090	Average Score: 0.03
epsilon: 2.416180999999161
Episode 3091	Average Score: 0.02
epsilon: 2.4154949999991584
Episode 3092	Average Score: 0.03
epsilon: 2.4139269999991524
Episode 3093	Average Score: 0.03
epsilon: 2.4132409999991498
Episode 3094	Average Score: 0.03
epsilon: 2.412554999999147
Episode 3095	Average Score: 0.02
epsilon: 2.4118689999991445
Episode 3096	Average Score: 0.02
epsilon: 2.411182999999142
Episode 3097	Average Score: 0.02
epsilon: 2.4104969999991392
Episode 3098	Average Score: 0.02
epsilon: 2.4097619999991364
Episode 3099	Average Score: 0.02
epsilon: 2.409075999999134
Episode 3100	Average Score: 0.02
epsilon: 2.4074589999991276
Episode 3101	Average Score: 0.02
epsilon: 2.406821999999125
Episode 3102	Average Score: 0.02
epsilon: 2.405204999999119
Episode 3103	Average Score: 0.02
epsilon: 2.404469999999116
Episode 3104	Average Score: 0.02
epsilon: 2.4037839999991135
Episode 3105	Average Score: 0.02
epsilon: 2.403097999999111
Episode 3106	Average Score: 0.02
epsilon: 2.4024119999991083
Episode 3107	Average Score: 0.02
epsilon: 2.4017259999991056
Episode 3108	Average Score: 0.02
epsilon: 2.401039999999103
Episode 3109	Average Score: 0.02
epsilon: 2.4003539999991004
Episode 3110	Average Score: 0.02
epsilon: 2.3988349999990946
Episode 3111	Average Score: 0.02
epsilon: 2.398148999999092
Episode 3112	Average Score: 0.02
epsilon: 2.397413999999089
Episode 3113	Average Score: 0.02
epsilon: 2.3967279999990865
Episode 3114	Average Score: 0.02
epsilon: 2.396041999999084
Episode 3115	Average Score: 0.02
epsilon: 2.395355999999081
Episode 3116	Average Score: 0.02
epsilon: 2.3946699999990786
Episode 3117	Average Score: 0.02
epsilon: 2.3939349999990758
Episode 3118	Average Score: 0.02
epsilon: 2.3923669999990698
Episode 3119	Average Score: 0.02
epsilon: 2.391680999999067
Episode 3120	Average Score: 0.02
epsilon: 2.3907989999990638
Episode 3121	Average Score: 0.02
epsilon: 2.390112999999061
Episode 3122	Average Score: 0.02
epsilon: 2.3894269999990585
Episode 3123	Average Score: 0.02
epsilon: 2.388740999999056
Episode 3124	Average Score: 0.02
epsilon: 2.3880549999990532
Episode 3125	Average Score: 0.02
epsilon: 2.3873199999990504
Episode 3126	Average Score: 0.02
epsilon: 2.386633999999048
Episode 3127	Average Score: 0.02
epsilon: 2.385114999999042
Episode 3128	Average Score: 0.02
epsilon: 2.3844289999990393
Episode 3129	Average Score: 0.02
epsilon: 2.3837429999990367
Episode 3130	Average Score: 0.02
epsilon: 2.383007999999034
Episode 3131	Average Score: 0.02
epsilon: 2.381488999999028
Episode 3132	Average Score: 0.02
epsilon: 2.3808029999990254
Episode 3133	Average Score: 0.02
epsilon: 2.380116999999023
Episode 3134	Average Score: 0.02
epsilon: 2.37943099999902
Episode 3135	Average Score: 0.02
epsilon: 2.3768829999990104
Episode 3136	Average Score: 0.02
epsilon: 2.373109999998996
Episode 3137	Average Score: 0.02
epsilon: 2.3724239999989933
Episode 3138	Average Score: 0.02
epsilon: 2.3717379999989907
Episode 3139	Average Score: 0.02
epsilon: 2.368650999998979
Episode 3140	Average Score: 0.02
epsilon: 2.3679649999989762
Episode 3141	Average Score: 0.02
epsilon: 2.3672789999989736
Episode 3142	Average Score: 0.02
epsilon: 2.366592999998971
Episode 3143	Average Score: 0.02
epsilon: 2.365024999998965
Episode 3144	Average Score: 0.02
epsilon: 2.3643389999989624
Episode 3145	Average Score: 0.02
epsilon: 2.362721999998956
Episode 3146	Average Score: 0.02
epsilon: 2.3612029999989503
Episode 3147	Average Score: 0.02
epsilon: 2.3605169999989477
Episode 3148	Average Score: 0.02
epsilon: 2.359830999998945
Episode 3149	Average Score: 0.02
epsilon: 2.3590959999989423
Episode 3150	Average Score: 0.02
epsilon: 2.35845899999894
Episode 3151	Average Score: 0.02
epsilon: 2.356939999998934
Episode 3152	Average Score: 0.02
epsilon: 2.3562539999989314
Episode 3153	Average Score: 0.02
epsilon: 2.3555679999989287
Episode 3154	Average Score: 0.02
epsilon: 2.354832999998926
Episode 3155	Average Score: 0.02
epsilon: 2.3541469999989233
Episode 3156	Average Score: 0.02
epsilon: 2.3534609999989207
Episode 3157	Average Score: 0.02
epsilon: 2.352774999998918
Episode 3158	Average Score: 0.02
epsilon: 2.3520889999989154
Episode 3159	Average Score: 0.02
epsilon: 2.3513539999989126
Episode 3160	Average Score: 0.02
epsilon: 2.35066799999891
Episode 3161	Average Score: 0.02
epsilon: 2.3499819999989073
Episode 3162	Average Score: 0.02
epsilon: 2.3492959999989047
Episode 3163	Average Score: 0.02
epsilon: 2.347776999998899
Episode 3164	Average Score: 0.02
epsilon: 2.3470909999988963
Episode 3165	Average Score: 0.02
epsilon: 2.3464049999988936
Episode 3166	Average Score: 0.02
epsilon: 2.345718999998891
Episode 3167	Average Score: 0.02
epsilon: 2.3450329999988884
Episode 3168	Average Score: 0.02
epsilon: 2.3443469999988857
Episode 3169	Average Score: 0.02
epsilon: 2.3432199999988814
Episode 3170	Average Score: 0.02
epsilon: 2.342533999998879
Episode 3171	Average Score: 0.02
epsilon: 2.3411129999988733
Episode 3172	Average Score: 0.02
epsilon: 2.34023099999887
Episode 3173	Average Score: 0.02
epsilon: 2.3393979999988668
Episode 3174	Average Score: 0.02
epsilon: 2.338711999998864
Episode 3175	Average Score: 0.02
epsilon: 2.3379769999988613
Episode 3176	Average Score: 0.02
epsilon: 2.3364089999988553
Episode 3177	Average Score: 0.02
epsilon: 2.3357229999988527
Episode 3178	Average Score: 0.02
epsilon: 2.33498799999885
Episode 3179	Average Score: 0.02
epsilon: 2.3343019999988472
Episode 3180	Average Score: 0.02
epsilon: 2.3336159999988446
Episode 3181	Average Score: 0.02
epsilon: 2.332929999998842
Episode 3182	Average Score: 0.02
epsilon: 2.3322439999988394
Episode 3183	Average Score: 0.02
epsilon: 2.3315579999988367
Episode 3184	Average Score: 0.02
epsilon: 2.32984299999883
Episode 3185	Average Score: 0.02
epsilon: 2.3291569999988275
Episode 3186	Average Score: 0.02
epsilon: 2.328470999998825
Episode 3187	Average Score: 0.02
epsilon: 2.327735999998822
Episode 3188	Average Score: 0.02
epsilon: 2.3270499999988195
Episode 3189	Average Score: 0.02
epsilon: 2.326363999998817
Episode 3190	Average Score: 0.02
epsilon: 2.325677999998814
Episode 3191	Average Score: 0.02
epsilon: 2.324060999998808
Episode 3192	Average Score: 0.02
epsilon: 2.323325999998805
Episode 3193	Average Score: 0.02
epsilon: 2.3226399999988026
Episode 3194	Average Score: 0.02
epsilon: 2.3219539999988
Episode 3195	Average Score: 0.02
epsilon: 2.3212679999987973
Episode 3196	Average Score: 0.02
epsilon: 2.3205819999987947
Episode 3197	Average Score: 0.02
epsilon: 2.319895999998792
Episode 3198	Average Score: 0.02
epsilon: 2.3192099999987894
Episode 3199	Average Score: 0.02
epsilon: 2.3184749999987866
Episode 3200	Average Score: 0.02
epsilon: 2.317788999998784
Episode 3201	Average Score: 0.02
epsilon: 2.3161719999987778
Episode 3202	Average Score: 0.02
epsilon: 2.315485999998775
Episode 3203	Average Score: 0.02
epsilon: 2.3147999999987725
Episode 3204	Average Score: 0.02
epsilon: 2.3140649999987697
Episode 3205	Average Score: 0.02
epsilon: 2.313378999998767
Episode 3206	Average Score: 0.02
epsilon: 2.3126929999987644
Episode 3207	Average Score: 0.02
epsilon: 2.312006999998762
Episode 3208	Average Score: 0.02
epsilon: 2.311320999998759
Episode 3209	Average Score: 0.02
epsilon: 2.3105859999987564
Episode 3210	Average Score: 0.02
epsilon: 2.3098999999987537
Episode 3211	Average Score: 0.02
epsilon: 2.309213999998751
Episode 3212	Average Score: 0.02
epsilon: 2.3085279999987485
Episode 3213	Average Score: 0.02
epsilon: 2.307841999998746
Episode 3214	Average Score: 0.02
epsilon: 2.307106999998743
Episode 3215	Average Score: 0.02
epsilon: 2.3064209999987404
Episode 3216	Average Score: 0.02
epsilon: 2.3057349999987378
Episode 3217	Average Score: 0.02
epsilon: 2.303235999998728
Episode 3218	Average Score: 0.02
epsilon: 2.3025499999987256
Episode 3219	Average Score: 0.02
epsilon: 2.3018149999987227
Episode 3220	Average Score: 0.02
epsilon: 2.300344999998717
Episode 3221	Average Score: 0.02
epsilon: 2.2988259999987113
Episode 3222	Average Score: 0.02
epsilon: 2.296179999998701
Episode 3223	Average Score: 0.02
epsilon: 2.294562999998695
Episode 3224	Average Score: 0.02
epsilon: 2.2938769999986923
Episode 3225	Average Score: 0.02
epsilon: 2.29323999999869
Episode 3226	Average Score: 0.02
epsilon: 2.292455999998687
Episode 3227	Average Score: 0.02
epsilon: 2.2917699999986842
Episode 3228	Average Score: 0.02
epsilon: 2.2910839999986816
Episode 3229	Average Score: 0.02
epsilon: 2.290397999998679
Episode 3230	Average Score: 0.02
epsilon: 2.289662999998676
Episode 3231	Average Score: 0.02
epsilon: 2.2889769999986735
Episode 3232	Average Score: 0.02
epsilon: 2.288290999998671
Episode 3233	Average Score: 0.02
epsilon: 2.2866249999986645
Episode 3234	Average Score: 0.02
epsilon: 2.285938999998662
Episode 3235	Average Score: 0.02
epsilon: 2.2852529999986593
Episode 3236	Average Score: 0.02
epsilon: 2.2845669999986566
Episode 3237	Average Score: 0.02
epsilon: 2.283831999998654
Episode 3238	Average Score: 0.02
epsilon: 2.283145999998651
Episode 3239	Average Score: 0.02
epsilon: 2.280450999998641
Episode 3240	Average Score: 0.02
epsilon: 2.2797649999986382
Episode 3241	Average Score: 0.02
epsilon: 2.2790789999986356
Episode 3242	Average Score: 0.02
epsilon: 2.2774619999986294
Episode 3243	Average Score: 0.02
epsilon: 2.276775999998627
Episode 3244	Average Score: 0.02
epsilon: 2.276089999998624
Episode 3245	Average Score: 0.02
epsilon: 2.2754039999986215
Episode 3246	Average Score: 0.02
epsilon: 2.2737869999986153
Episode 3247	Average Score: 0.02
epsilon: 2.2731009999986127
Episode 3248	Average Score: 0.02
epsilon: 2.27241499999861
Episode 3249	Average Score: 0.02
epsilon: 2.2708959999986043
Episode 3250	Average Score: 0.02
epsilon: 2.270258999998602
Episode 3251	Average Score: 0.02
epsilon: 2.269572999998599
Episode 3252	Average Score: 0.02
epsilon: 2.2688869999985966
Episode 3253	Average Score: 0.02
epsilon: 2.268200999998594
Episode 3254	Average Score: 0.02
epsilon: 2.267465999998591
Episode 3255	Average Score: 0.02
epsilon: 2.2667799999985885
Episode 3256	Average Score: 0.02
epsilon: 2.266093999998586
Episode 3257	Average Score: 0.02
epsilon: 2.265407999998583
Episode 3258	Average Score: 0.02
epsilon: 2.2639379999985776
Episode 3259	Average Score: 0.02
epsilon: 2.2623699999985716
Episode 3260	Average Score: 0.02
epsilon: 2.261683999998569
Episode 3261	Average Score: 0.02
epsilon: 2.260948999998566
Episode 3262	Average Score: 0.02
epsilon: 2.2602629999985635
Episode 3263	Average Score: 0.02
epsilon: 2.259576999998561
Episode 3264	Average Score: 0.02
epsilon: 2.2588909999985582
Episode 3265	Average Score: 0.02
epsilon: 2.2582049999985556
Episode 3266	Average Score: 0.02
epsilon: 2.257518999998553
Episode 3267	Average Score: 0.02
epsilon: 2.2568329999985504
Episode 3268	Average Score: 0.02
epsilon: 2.2555099999985453
Episode 3269	Average Score: 0.02
epsilon: 2.253892999998539
Episode 3270	Average Score: 0.02
epsilon: 2.2532069999985365
Episode 3271	Average Score: 0.02
epsilon: 2.2524229999985335
Episode 3272	Average Score: 0.02
epsilon: 2.251736999998531
Episode 3273	Average Score: 0.02
epsilon: 2.251050999998528
Episode 3274	Average Score: 0.02
epsilon: 2.2503649999985256
Episode 3275	Average Score: 0.02
epsilon: 2.249678999998523
Episode 3276	Average Score: 0.02
epsilon: 2.24894399999852
Episode 3277	Average Score: 0.02
epsilon: 2.2482579999985175
Episode 3278	Average Score: 0.02
epsilon: 2.2466899999985115
Episode 3279	Average Score: 0.02
epsilon: 2.2450729999985053
Episode 3280	Average Score: 0.02
epsilon: 2.2443869999985027
Episode 3281	Average Score: 0.02
epsilon: 2.2437009999985
Episode 3282	Average Score: 0.02
epsilon: 2.2430149999984974
Episode 3283	Average Score: 0.02
epsilon: 2.2423289999984948
Episode 3284	Average Score: 0.02
epsilon: 2.241642999998492
Episode 3285	Average Score: 0.02
epsilon: 2.2409569999984895
Episode 3286	Average Score: 0.02
epsilon: 2.2402219999984867
Episode 3287	Average Score: 0.02
epsilon: 2.2395849999984843
Episode 3288	Average Score: 0.02
epsilon: 2.2388499999984814
Episode 3289	Average Score: 0.02
epsilon: 2.238163999998479
Episode 3290	Average Score: 0.02
epsilon: 2.237428999998476
Episode 3291	Average Score: 0.02
epsilon: 2.2367429999984734
Episode 3292	Average Score: 0.02
epsilon: 2.2360569999984707
Episode 3293	Average Score: 0.02
epsilon: 2.235370999998468
Episode 3294	Average Score: 0.02
epsilon: 2.2346849999984655
Episode 3295	Average Score: 0.02
epsilon: 2.233998999998463
Episode 3296	Average Score: 0.02
epsilon: 2.232528999998457
Episode 3297	Average Score: 0.02
epsilon: 2.2317939999984544
Episode 3298	Average Score: 0.02
epsilon: 2.2311079999984518
Episode 3299	Average Score: 0.02
epsilon: 2.229588999998446
Episode 3300	Average Score: 0.02
epsilon: 2.2289029999984433
Episode 3301	Average Score: 0.02
epsilon: 2.2282169999984407
Episode 3302	Average Score: 0.02
epsilon: 2.2265509999984343
Episode 3303	Average Score: 0.02
epsilon: 2.2258649999984317
Episode 3304	Average Score: 0.02
epsilon: 2.225178999998429
Episode 3305	Average Score: 0.02
epsilon: 2.2244929999984264
Episode 3306	Average Score: 0.02
epsilon: 2.223806999998424
Episode 3307	Average Score: 0.02
epsilon: 2.223071999998421
Episode 3308	Average Score: 0.02
epsilon: 2.2223859999984183
Episode 3309	Average Score: 0.02
epsilon: 2.2216999999984157
Episode 3310	Average Score: 0.02
epsilon: 2.221013999998413
Episode 3311	Average Score: 0.02
epsilon: 2.2193479999984067
Episode 3312	Average Score: 0.02
epsilon: 2.218661999998404
Episode 3313	Average Score: 0.02
epsilon: 2.2179759999984014
Episode 3314	Average Score: 0.02
epsilon: 2.217289999998399
Episode 3315	Average Score: 0.02
epsilon: 2.216603999998396
Episode 3316	Average Score: 0.02
epsilon: 2.2159179999983936
Episode 3317	Average Score: 0.02
epsilon: 2.215231999998391
Episode 3318	Average Score: 0.02
epsilon: 2.214496999998388
Episode 3319	Average Score: 0.02
epsilon: 2.2138109999983855
Episode 3320	Average Score: 0.02
epsilon: 2.213124999998383
Episode 3321	Average Score: 0.02
epsilon: 2.2124389999983802
Episode 3322	Average Score: 0.01
epsilon: 2.2117529999983776
Episode 3323	Average Score: 0.01
epsilon: 2.211017999998375
Episode 3324	Average Score: 0.01
epsilon: 2.210331999998372
Episode 3325	Average Score: 0.01
epsilon: 2.2096459999983695
Episode 3326	Average Score: 0.01
epsilon: 2.2089109999983667
Episode 3327	Average Score: 0.01
epsilon: 2.208224999998364
Episode 3328	Average Score: 0.01
epsilon: 2.2075389999983615
Episode 3329	Average Score: 0.01
epsilon: 2.2060199999983556
Episode 3330	Average Score: 0.01
epsilon: 2.205333999998353
Episode 3331	Average Score: 0.01
epsilon: 2.2046479999983504
Episode 3332	Average Score: 0.01
epsilon: 2.2039619999983477
Episode 3333	Average Score: 0.01
epsilon: 2.203226999998345
Episode 3334	Average Score: 0.01
epsilon: 2.2025409999983423
Episode 3335	Average Score: 0.01
epsilon: 2.2018549999983397
Episode 3336	Average Score: 0.01
epsilon: 2.19930699999833
Episode 3337	Average Score: 0.01
epsilon: 2.1986209999983273
Episode 3338	Average Score: 0.01
epsilon: 2.1979349999983246
Episode 3339	Average Score: 0.01
epsilon: 2.197248999998322
Episode 3340	Average Score: 0.01
epsilon: 2.196513999998319
Episode 3341	Average Score: 0.01
epsilon: 2.1958279999983166
Episode 3342	Average Score: 0.01
epsilon: 2.195141999998314
Episode 3343	Average Score: 0.01
epsilon: 2.1944559999983113
Episode 3344	Average Score: 0.01
epsilon: 2.1937699999983087
Episode 3345	Average Score: 0.01
epsilon: 2.193034999998306
Episode 3346	Average Score: 0.01
epsilon: 2.1923489999983032
Episode 3347	Average Score: 0.01
epsilon: 2.1916629999983006
Episode 3348	Average Score: 0.01
epsilon: 2.190192999998295
Episode 3349	Average Score: 0.01
epsilon: 2.1895069999982923
Episode 3350	Average Score: 0.01
epsilon: 2.1888209999982897
Episode 3351	Average Score: 0.01
epsilon: 2.1871549999982833
Episode 3352	Average Score: 0.01
epsilon: 2.1856359999982775
Episode 3353	Average Score: 0.01
epsilon: 2.184949999998275
Episode 3354	Average Score: 0.01
epsilon: 2.1842639999982723
Episode 3355	Average Score: 0.01
epsilon: 2.1835779999982696
Episode 3356	Average Score: 0.01
epsilon: 2.182842999998267
Episode 3357	Average Score: 0.01
epsilon: 2.182156999998264
Episode 3358	Average Score: 0.01
epsilon: 2.1814709999982616
Episode 3359	Average Score: 0.01
epsilon: 2.180784999998259
Episode 3360	Average Score: 0.01
epsilon: 2.1800989999982563
Episode 3361	Average Score: 0.01
epsilon: 2.1793639999982535
Episode 3362	Average Score: 0.01
epsilon: 2.178677999998251
Episode 3363	Average Score: 0.01
epsilon: 2.177893999998248
Episode 3364	Average Score: 0.01
epsilon: 2.177207999998245
Episode 3365	Average Score: 0.01
epsilon: 2.1765219999982426
Episode 3366	Average Score: 0.01
epsilon: 2.17583599999824
Episode 3367	Average Score: 0.01
epsilon: 2.174316999998234
Episode 3368	Average Score: 0.01
epsilon: 2.1736309999982315
Episode 3369	Average Score: 0.01
epsilon: 2.1720139999982253
Episode 3370	Average Score: 0.01
epsilon: 2.171180999998222
Episode 3371	Average Score: 0.01
epsilon: 2.169612999998216
Episode 3372	Average Score: 0.01
epsilon: 2.1689269999982135
Episode 3373	Average Score: 0.01
epsilon: 2.168240999998211
Episode 3374	Average Score: 0.01
epsilon: 2.167554999998208
Episode 3375	Average Score: 0.01
epsilon: 2.166770999998205
Episode 3376	Average Score: 0.01
epsilon: 2.1660849999982026
Episode 3377	Average Score: 0.01
epsilon: 2.1653499999981998
Episode 3378	Average Score: 0.01
epsilon: 2.164663999998197
Episode 3379	Average Score: 0.01
epsilon: 2.1639779999981945
Episode 3380	Average Score: 0.01
epsilon: 2.163291999998192
Episode 3381	Average Score: 0.01
epsilon: 2.162556999998189
Episode 3382	Average Score: 0.01
epsilon: 2.1618709999981864
Episode 3383	Average Score: 0.01
epsilon: 2.161184999998184
Episode 3384	Average Score: 0.01
epsilon: 2.160498999998181
Episode 3385	Average Score: 0.01
epsilon: 2.1598129999981786
Episode 3386	Average Score: 0.01
epsilon: 2.159126999998176
Episode 3387	Average Score: 0.01
epsilon: 2.1584409999981733
Episode 3388	Average Score: 0.01
epsilon: 2.1577549999981707
Episode 3389	Average Score: 0.01
epsilon: 2.157068999998168
Episode 3390	Average Score: 0.01
epsilon: 2.1563829999981654
Episode 3391	Average Score: 0.01
epsilon: 2.1556479999981626
Episode 3392	Average Score: 0.01
epsilon: 2.15501099999816
Episode 3393	Average Score: 0.01
epsilon: 2.1543249999981575
Episode 3394	Average Score: 0.01
epsilon: 2.153638999998155
Episode 3395	Average Score: 0.01
epsilon: 2.152903999998152
Episode 3396	Average Score: 0.01
epsilon: 2.1513849999981463
Episode 3397	Average Score: 0.01
epsilon: 2.1506989999981436
Episode 3398	Average Score: 0.01
epsilon: 2.149963999998141
Episode 3399	Average Score: 0.01
epsilon: 2.149277999998138
Episode 3400	Average Score: 0.01
epsilon: 2.1485919999981355
Episode 3401	Average Score: 0.01
epsilon: 2.147905999998133
Episode 3402	Average Score: 0.01
epsilon: 2.1472199999981303
Episode 3403	Average Score: 0.01
epsilon: 2.1465339999981277
Episode 3404	Average Score: 0.01
epsilon: 2.145798999998125
Episode 3405	Average Score: 0.01
epsilon: 2.145112999998122
Episode 3406	Average Score: 0.01
epsilon: 2.1444269999981196
Episode 3407	Average Score: 0.01
epsilon: 2.142956999998114
Episode 3408	Average Score: 0.01
epsilon: 2.141437999998108
Episode 3409	Average Score: 0.01
epsilon: 2.1407519999981055
Episode 3410	Average Score: 0.01
epsilon: 2.140065999998103
Episode 3411	Average Score: 0.01
epsilon: 2.1393799999981002
Episode 3412	Average Score: 0.01
epsilon: 2.1378119999980942
Episode 3413	Average Score: 0.01
epsilon: 2.1371259999980916
Episode 3414	Average Score: 0.01
epsilon: 2.136439999998089
Episode 3415	Average Score: 0.01
epsilon: 2.135704999998086
Episode 3416	Average Score: 0.01
epsilon: 2.1350189999980835
Episode 3417	Average Score: 0.01
epsilon: 2.134332999998081
Episode 3418	Average Score: 0.01
epsilon: 2.1336469999980783
Episode 3419	Average Score: 0.01
epsilon: 2.1329609999980756
Episode 3420	Average Score: 0.01
epsilon: 2.132225999998073
Episode 3421	Average Score: 0.01
epsilon: 2.13153999999807
Episode 3422	Average Score: 0.01
epsilon: 2.1308539999980676
Episode 3423	Average Score: 0.01
epsilon: 2.130167999998065
Episode 3424	Average Score: 0.01
epsilon: 2.1294819999980623
Episode 3425	Average Score: 0.01
epsilon: 2.1287469999980595
Episode 3426	Average Score: 0.01
epsilon: 2.128060999998057
Episode 3427	Average Score: 0.01
epsilon: 2.1273749999980542
Episode 3428	Average Score: 0.01
epsilon: 2.1266889999980516
Episode 3429	Average Score: 0.01
epsilon: 2.1253169999980464
Episode 3430	Average Score: 0.01
epsilon: 2.1246309999980437
Episode 3431	Average Score: 0.01
epsilon: 2.123895999998041
Episode 3432	Average Score: 0.01
epsilon: 2.1230629999980377
Episode 3433	Average Score: 0.01
epsilon: 2.122327999998035
Episode 3434	Average Score: 0.01
epsilon: 2.1216419999980323
Episode 3435	Average Score: 0.01
epsilon: 2.119975999998026
Episode 3436	Average Score: 0.01
epsilon: 2.1192899999980233
Episode 3437	Average Score: 0.01
epsilon: 2.1186039999980206
Episode 3438	Average Score: 0.01
epsilon: 2.117917999998018
Episode 3439	Average Score: 0.01
epsilon: 2.1172319999980154
Episode 3440	Average Score: 0.01
epsilon: 2.1164969999980126
Episode 3441	Average Score: 0.01
epsilon: 2.11585999999801
Episode 3442	Average Score: 0.01
epsilon: 2.1151739999980075
Episode 3443	Average Score: 0.01
epsilon: 2.114487999998005
Episode 3444	Average Score: 0.01
epsilon: 2.1138019999980022
Episode 3445	Average Score: 0.01
epsilon: 2.112184999997996
Episode 3446	Average Score: 0.01
epsilon: 2.111449999997993
Episode 3447	Average Score: 0.01
epsilon: 2.1107639999979906
Episode 3448	Average Score: 0.01
epsilon: 2.110077999997988
Episode 3449	Average Score: 0.01
epsilon: 2.1093919999979853
Episode 3450	Average Score: 0.01
epsilon: 2.1078729999979795
Episode 3451	Average Score: 0.01
epsilon: 2.107186999997977
Episode 3452	Average Score: 0.01
epsilon: 2.1065009999979742
Episode 3453	Average Score: 0.01
epsilon: 2.104883999997968
Episode 3454	Average Score: 0.01
epsilon: 2.1041979999979654
Episode 3455	Average Score: 0.01
epsilon: 2.103511999997963
Episode 3456	Average Score: 0.01
epsilon: 2.10282599999796
Episode 3457	Average Score: 0.01
epsilon: 2.1010619999979534
Episode 3458	Average Score: 0.01
epsilon: 2.1003759999979508
Episode 3459	Average Score: 0.01
epsilon: 2.0979749999979416
Episode 3460	Average Score: 0.01
epsilon: 2.097337999997939
Episode 3461	Average Score: 0.01
epsilon: 2.0966029999979363
Episode 3462	Average Score: 0.01
epsilon: 2.0950839999979305
Episode 3463	Average Score: 0.02
epsilon: 2.0935159999979245
Episode 3464	Average Score: 0.02
epsilon: 2.0909679999979147
Episode 3465	Average Score: 0.02
epsilon: 2.0903309999979123
Episode 3466	Average Score: 0.02
epsilon: 2.0895959999979095
Episode 3467	Average Score: 0.02
epsilon: 2.088909999997907
Episode 3468	Average Score: 0.02
epsilon: 2.088223999997904
Episode 3469	Average Score: 0.02
epsilon: 2.0841079999978884
Episode 3470	Average Score: 0.02
epsilon: 2.083421999997886
Episode 3471	Average Score: 0.02
epsilon: 2.082735999997883
Episode 3472	Average Score: 0.02
epsilon: 2.0820009999978804
Episode 3473	Average Score: 0.02
epsilon: 2.0813149999978777
Episode 3474	Average Score: 0.02
epsilon: 2.080628999997875
Episode 3475	Average Score: 0.02
epsilon: 2.0799919999978727
Episode 3476	Average Score: 0.02
epsilon: 2.07925699999787
Episode 3477	Average Score: 0.02
epsilon: 2.078570999997867
Episode 3478	Average Score: 0.02
epsilon: 2.0778849999978646
Episode 3479	Average Score: 0.02
epsilon: 2.077247999997862
Episode 3480	Average Score: 0.02
epsilon: 2.0765129999978593
Episode 3481	Average Score: 0.02
epsilon: 2.0758269999978567
Episode 3482	Average Score: 0.02
epsilon: 2.075140999997854
Episode 3483	Average Score: 0.02
epsilon: 2.0732299999978467
Episode 3484	Average Score: 0.02
epsilon: 2.072494999997844
Episode 3485	Average Score: 0.02
epsilon: 2.0707799999978374
Episode 3486	Average Score: 0.02
epsilon: 2.0667129999978218
Episode 3487	Average Score: 0.02
epsilon: 2.066026999997819
Episode 3488	Average Score: 0.02
epsilon: 2.0653409999978165
Episode 3489	Average Score: 0.02
epsilon: 2.0646059999978137
Episode 3490	Average Score: 0.02
epsilon: 2.0639689999978112
Episode 3491	Average Score: 0.02
epsilon: 2.0632829999978086
Episode 3492	Average Score: 0.02
epsilon: 2.062547999997806
Episode 3493	Average Score: 0.02
epsilon: 2.061861999997803
Episode 3494	Average Score: 0.02
epsilon: 2.0603919999977975
Episode 3495	Average Score: 0.02
epsilon: 2.059705999997795
Episode 3496	Average Score: 0.02
epsilon: 2.0590199999977923
Episode 3497	Average Score: 0.02
epsilon: 2.0583339999977897
Episode 3498	Average Score: 0.02
epsilon: 2.057647999997787
Episode 3499	Average Score: 0.02
epsilon: 2.0569619999977844
Episode 3500	Average Score: 0.02
epsilon: 2.0547079999977758
Episode 3501	Average Score: 0.02
epsilon: 2.054021999997773
Episode 3502	Average Score: 0.02
epsilon: 2.0528459999977686
Episode 3503	Average Score: 0.02
epsilon: 2.051375999997763
Episode 3504	Average Score: 0.02
epsilon: 2.0506899999977604
Episode 3505	Average Score: 0.02
epsilon: 2.0500039999977577
Episode 3506	Average Score: 0.02
epsilon: 2.049317999997755
Episode 3507	Average Score: 0.02
epsilon: 2.0477989999977493
Episode 3508	Average Score: 0.02
epsilon: 2.0471129999977467
Episode 3509	Average Score: 0.02
epsilon: 2.046426999997744
Episode 3510	Average Score: 0.02
epsilon: 2.0457409999977414
Episode 3511	Average Score: 0.02
epsilon: 2.0450549999977388
Episode 3512	Average Score: 0.02
epsilon: 2.044368999997736
Episode 3513	Average Score: 0.02
epsilon: 2.0428989999977305
Episode 3514	Average Score: 0.02
epsilon: 2.042212999997728
Episode 3515	Average Score: 0.02
epsilon: 2.0415269999977252
Episode 3516	Average Score: 0.02
epsilon: 2.0390279999977157
Episode 3517	Average Score: 0.02
epsilon: 2.038292999997713
Episode 3518	Average Score: 0.02
epsilon: 2.0376069999977102
Episode 3519	Average Score: 0.02
epsilon: 2.0358429999977035
Episode 3520	Average Score: 0.02
epsilon: 2.035156999997701
Episode 3521	Average Score: 0.02
epsilon: 2.034470999997698
Episode 3522	Average Score: 0.02
epsilon: 2.0337849999976956
Episode 3523	Average Score: 0.02
epsilon: 2.033098999997693
Episode 3524	Average Score: 0.02
epsilon: 2.03236399999769
Episode 3525	Average Score: 0.02
epsilon: 2.0316779999976875
Episode 3526	Average Score: 0.02
epsilon: 2.030991999997685
Episode 3527	Average Score: 0.02
epsilon: 2.0303059999976822
Episode 3528	Average Score: 0.02
epsilon: 2.0296199999976796
Episode 3529	Average Score: 0.02
epsilon: 2.028884999997677
Episode 3530	Average Score: 0.02
epsilon: 2.028198999997674
Episode 3531	Average Score: 0.02
epsilon: 2.0275129999976715
Episode 3532	Average Score: 0.02
epsilon: 2.026826999997669
Episode 3533	Average Score: 0.02
epsilon: 2.0261409999976663
Episode 3534	Average Score: 0.02
epsilon: 2.023984999997658
Episode 3535	Average Score: 0.02
epsilon: 2.0232989999976554
Episode 3536	Average Score: 0.02
epsilon: 2.0219759999976503
Episode 3537	Average Score: 0.02
epsilon: 2.0212899999976477
Episode 3538	Average Score: 0.02
epsilon: 2.020603999997645
Episode 3539	Average Score: 0.02
epsilon: 2.0199179999976424
Episode 3540	Average Score: 0.02
epsilon: 2.0191829999976396
Episode 3541	Average Score: 0.02
epsilon: 2.017663999997634
Episode 3542	Average Score: 0.02
epsilon: 2.016144999997628
Episode 3543	Average Score: 0.02
epsilon: 2.0154589999976253
Episode 3544	Average Score: 0.02
epsilon: 2.0147729999976227
Episode 3545	Average Score: 0.02
epsilon: 2.013792999997619
Episode 3546	Average Score: 0.02
epsilon: 2.0123229999976133
Episode 3547	Average Score: 0.02
epsilon: 2.0115879999976105
Episode 3548	Average Score: 0.02
epsilon: 2.010950999997608
Episode 3549	Average Score: 0.03
epsilon: 2.008598999997599
Episode 3550	Average Score: 0.03
epsilon: 2.0071289999975934
Episode 3551	Average Score: 0.03
epsilon: 2.006442999997591
Episode 3552	Average Score: 0.03
epsilon: 2.005756999997588
Episode 3553	Average Score: 0.02
epsilon: 2.0050709999975855
Episode 3554	Average Score: 0.03
epsilon: 2.0035029999975795
Episode 3555	Average Score: 0.03
epsilon: 2.0024249999975754
Episode 3556	Average Score: 0.03
epsilon: 2.0017389999975728
Episode 3557	Average Score: 0.02
epsilon: 2.00105299999757
Episode 3558	Average Score: 0.02
epsilon: 2.0003669999975675
Episode 3559	Average Score: 0.02
epsilon: 1.998896999997567
Episode 3560	Average Score: 0.02
epsilon: 1.9982109999975675
Episode 3561	Average Score: 0.02
epsilon: 1.997524999997568
Episode 3562	Average Score: 0.02
epsilon: 1.9968389999975684
Episode 3563	Average Score: 0.02
epsilon: 1.996152999997569
Episode 3564	Average Score: 0.02
epsilon: 1.9954669999975694
Episode 3565	Average Score: 0.02
epsilon: 1.9947809999975699
Episode 3566	Average Score: 0.02
epsilon: 1.9924289999975715
Episode 3567	Average Score: 0.02
epsilon: 1.9909099999975726
Episode 3568	Average Score: 0.02
epsilon: 1.990223999997573
Episode 3569	Average Score: 0.02
epsilon: 1.988753999997574
Episode 3570	Average Score: 0.02
epsilon: 1.9880679999975746
Episode 3571	Average Score: 0.02
epsilon: 1.987381999997575
Episode 3572	Average Score: 0.02
epsilon: 1.985862999997576
Episode 3573	Average Score: 0.02
epsilon: 1.9851279999975766
Episode 3574	Average Score: 0.02
epsilon: 1.984441999997577
Episode 3575	Average Score: 0.02
epsilon: 1.9837559999975776
Episode 3576	Average Score: 0.02
epsilon: 1.983069999997578
Episode 3577	Average Score: 0.03
epsilon: 1.9792479999975807
Episode 3578	Average Score: 0.03
epsilon: 1.9785619999975812
Episode 3579	Average Score: 0.03
epsilon: 1.9770919999975822
Episode 3580	Average Score: 0.03
epsilon: 1.9764059999975827
Episode 3581	Average Score: 0.03
epsilon: 1.9757199999975832
Episode 3582	Average Score: 0.03
epsilon: 1.9733189999975849
Episode 3583	Average Score: 0.03
epsilon: 1.9725839999975854
Episode 3584	Average Score: 0.03
epsilon: 1.9718979999975859
Episode 3585	Average Score: 0.03
epsilon: 1.9712119999975863
Episode 3586	Average Score: 0.02
epsilon: 1.9705259999975868
Episode 3587	Average Score: 0.02
epsilon: 1.9698399999975873
Episode 3588	Average Score: 0.02
epsilon: 1.9691539999975878
Episode 3589	Average Score: 0.02
epsilon: 1.9684679999975883
Episode 3590	Average Score: 0.02
epsilon: 1.9677819999975887
Episode 3591	Average Score: 0.02
epsilon: 1.9670469999975893
Episode 3592	Average Score: 0.02
epsilon: 1.9663609999975897
Episode 3593	Average Score: 0.02
epsilon: 1.9656749999975902
Episode 3594	Average Score: 0.02
epsilon: 1.9649889999975907
Episode 3595	Average Score: 0.02
epsilon: 1.9634699999975918
Episode 3596	Average Score: 0.02
epsilon: 1.9627839999975922
Episode 3597	Average Score: 0.02
epsilon: 1.9604319999975939
Episode 3598	Average Score: 0.02
epsilon: 1.9597459999975944
Episode 3599	Average Score: 0.02
epsilon: 1.9590599999975948
Episode 3600	Average Score: 0.02
epsilon: 1.9566589999975965
Episode 3601	Average Score: 0.02
epsilon: 1.955874999997597
Episode 3602	Average Score: 0.02
epsilon: 1.9551889999975975
Episode 3603	Average Score: 0.02
epsilon: 1.9538169999975985
Episode 3604	Average Score: 0.02
epsilon: 1.953130999997599
Episode 3605	Average Score: 0.02
epsilon: 1.9524449999975995
Episode 3606	Average Score: 0.02
epsilon: 1.9517589999976
Episode 3607	Average Score: 0.02
epsilon: 1.9510239999976005
Episode 3608	Average Score: 0.02
epsilon: 1.950337999997601
Episode 3609	Average Score: 0.02
epsilon: 1.9496519999976014
Episode 3610	Average Score: 0.02
epsilon: 1.9481329999976025
Episode 3611	Average Score: 0.02
epsilon: 1.947446999997603
Episode 3612	Average Score: 0.02
epsilon: 1.9467609999976034
Episode 3613	Average Score: 0.02
epsilon: 1.946025999997604
Episode 3614	Average Score: 0.02
epsilon: 1.944408999997605
Episode 3615	Average Score: 0.03
epsilon: 1.9428409999976062
Episode 3616	Average Score: 0.02
epsilon: 1.9421549999976067
Episode 3617	Average Score: 0.02
epsilon: 1.9414199999976072
Episode 3618	Average Score: 0.02
epsilon: 1.9407339999976077
Episode 3619	Average Score: 0.02
epsilon: 1.9400479999976081
Episode 3620	Average Score: 0.02
epsilon: 1.9393619999976086
Episode 3621	Average Score: 0.02
epsilon: 1.938675999997609
Episode 3622	Average Score: 0.02
epsilon: 1.9379899999976096
Episode 3623	Average Score: 0.02
epsilon: 1.9363239999976107
Episode 3624	Average Score: 0.02
epsilon: 1.9356379999976112
Episode 3625	Average Score: 0.02
epsilon: 1.9349519999976117
Episode 3626	Average Score: 0.02
epsilon: 1.9342659999976122
Episode 3627	Average Score: 0.02
epsilon: 1.9335799999976127
Episode 3628	Average Score: 0.02
epsilon: 1.9328939999976131
Episode 3629	Average Score: 0.02
epsilon: 1.9322079999976136
Episode 3630	Average Score: 0.02
epsilon: 1.931521999997614
Episode 3631	Average Score: 0.03
epsilon: 1.9298559999976153
Episode 3632	Average Score: 0.03
epsilon: 1.9291699999976157
Episode 3633	Average Score: 0.03
epsilon: 1.9266219999976175
Episode 3634	Average Score: 0.03
epsilon: 1.925935999997618
Episode 3635	Average Score: 0.03
epsilon: 1.9252499999976185
Episode 3636	Average Score: 0.02
epsilon: 1.924563999997619
Episode 3637	Average Score: 0.02
epsilon: 1.9238289999976195
Episode 3638	Average Score: 0.02
epsilon: 1.92319199999762
Episode 3639	Average Score: 0.02
epsilon: 1.9224569999976204
Episode 3640	Average Score: 0.02
epsilon: 1.921770999997621
Episode 3641	Average Score: 0.02
epsilon: 1.9210849999976214
Episode 3642	Average Score: 0.02
epsilon: 1.9203989999976219
Episode 3643	Average Score: 0.02
epsilon: 1.9197129999976223
Episode 3644	Average Score: 0.02
epsilon: 1.9190269999976228
Episode 3645	Average Score: 0.02
epsilon: 1.917458999997624
Episode 3646	Average Score: 0.02
epsilon: 1.9167729999976244
Episode 3647	Average Score: 0.02
epsilon: 1.916037999997625
Episode 3648	Average Score: 0.02
epsilon: 1.9153519999976254
Episode 3649	Average Score: 0.02
epsilon: 1.9146659999976259
Episode 3650	Average Score: 0.02
epsilon: 1.9139309999976264
Episode 3651	Average Score: 0.02
epsilon: 1.9132449999976269
Episode 3652	Average Score: 0.02
epsilon: 1.911774999997628
Episode 3653	Average Score: 0.02
epsilon: 1.9110889999976284
Episode 3654	Average Score: 0.02
epsilon: 1.9085899999976301
Episode 3655	Average Score: 0.02
epsilon: 1.9079039999976306
Episode 3656	Average Score: 0.02
epsilon: 1.9060909999976319
Episode 3657	Average Score: 0.02
epsilon: 1.9054049999976324
Episode 3658	Average Score: 0.02
epsilon: 1.9047189999976328
Episode 3659	Average Score: 0.02
epsilon: 1.9040329999976333
Episode 3660	Average Score: 0.02
epsilon: 1.9032979999976338
Episode 3661	Average Score: 0.02
epsilon: 1.9026119999976343
Episode 3662	Average Score: 0.02
epsilon: 1.9019259999976348
Episode 3663	Average Score: 0.02
epsilon: 1.9003579999976359
Episode 3664	Average Score: 0.02
epsilon: 1.8996719999976364
Episode 3665	Average Score: 0.02
epsilon: 1.8982019999976374
Episode 3666	Average Score: 0.02
epsilon: 1.897466999997638
Episode 3667	Average Score: 0.02
epsilon: 1.8967809999976384
Episode 3668	Average Score: 0.02
epsilon: 1.89442899999764
Episode 3669	Average Score: 0.02
epsilon: 1.8928609999976411
Episode 3670	Average Score: 0.02
epsilon: 1.8913419999976422
Episode 3671	Average Score: 0.02
epsilon: 1.8906559999976427
Episode 3672	Average Score: 0.02
epsilon: 1.8894309999976435
Episode 3673	Average Score: 0.02
epsilon: 1.888744999997644
Episode 3674	Average Score: 0.02
epsilon: 1.8880589999976445
Episode 3675	Average Score: 0.02
epsilon: 1.887372999997645
Episode 3676	Average Score: 0.02
epsilon: 1.8866869999976454
Episode 3677	Average Score: 0.02
epsilon: 1.885951999997646
Episode 3678	Average Score: 0.02
epsilon: 1.8852659999976464
Episode 3679	Average Score: 0.02
epsilon: 1.884579999997647
Episode 3680	Average Score: 0.02
epsilon: 1.8838939999976474
Episode 3681	Average Score: 0.02
epsilon: 1.8832079999976479
Episode 3682	Average Score: 0.02
epsilon: 1.8824729999976484
Episode 3683	Average Score: 0.02
epsilon: 1.8817869999976489
Episode 3684	Average Score: 0.02
epsilon: 1.8811009999976493
Episode 3685	Average Score: 0.02
epsilon: 1.8795819999976504
Episode 3686	Average Score: 0.02
epsilon: 1.878895999997651
Episode 3687	Average Score: 0.02
epsilon: 1.8781609999976514
Episode 3688	Average Score: 0.02
epsilon: 1.8763969999976526
Episode 3689	Average Score: 0.02
epsilon: 1.8757109999976531
Episode 3690	Average Score: 0.02
epsilon: 1.8749759999976536
Episode 3691	Average Score: 0.02
epsilon: 1.874289999997654
Episode 3692	Average Score: 0.02
epsilon: 1.8736039999976546
Episode 3693	Average Score: 0.02
epsilon: 1.872917999997655
Episode 3694	Average Score: 0.02
epsilon: 1.871447999997656
Episode 3695	Average Score: 0.02
epsilon: 1.8699779999976571
Episode 3696	Average Score: 0.02
epsilon: 1.8692919999976576
Episode 3697	Average Score: 0.02
epsilon: 1.868605999997658
Episode 3698	Average Score: 0.02
epsilon: 1.8679199999976586
Episode 3699	Average Score: 0.02
epsilon: 1.8669889999976592
Episode 3700	Average Score: 0.02
epsilon: 1.8663029999976597
Episode 3701	Average Score: 0.02
epsilon: 1.8656169999976602
Episode 3702	Average Score: 0.02
epsilon: 1.8648819999976607
Episode 3703	Average Score: 0.02
epsilon: 1.8641959999976612
Episode 3704	Average Score: 0.02
epsilon: 1.8625789999976623
Episode 3705	Average Score: 0.02
epsilon: 1.8619419999976627
Episode 3706	Average Score: 0.02
epsilon: 1.8604229999976638
Episode 3707	Average Score: 0.02
epsilon: 1.8597369999976643
Episode 3708	Average Score: 0.02
epsilon: 1.8590509999976648
Episode 3709	Average Score: 0.02
epsilon: 1.8583649999976652
Episode 3710	Average Score: 0.02
epsilon: 1.8568459999976663
Episode 3711	Average Score: 0.02
epsilon: 1.8552779999976674
Episode 3712	Average Score: 0.02
epsilon: 1.8537099999976685
Episode 3713	Average Score: 0.02
epsilon: 1.853023999997669
Episode 3714	Average Score: 0.02
epsilon: 1.8523379999976695
Episode 3715	Average Score: 0.02
epsilon: 1.85165199999767
Episode 3716	Average Score: 0.02
epsilon: 1.850230999997671
Episode 3717	Average Score: 0.02
epsilon: 1.8495449999976714
Episode 3718	Average Score: 0.02
epsilon: 1.848858999997672
Episode 3719	Average Score: 0.02
epsilon: 1.8481239999976724
Episode 3720	Average Score: 0.02
epsilon: 1.8466049999976735
Episode 3721	Average Score: 0.02
epsilon: 1.845869999997674
Episode 3722	Average Score: 0.02
epsilon: 1.8451839999976745
Episode 3723	Average Score: 0.02
epsilon: 1.844497999997675
Episode 3724	Average Score: 0.02
epsilon: 1.842880999997676
Episode 3725	Average Score: 0.02
epsilon: 1.8421949999976766
Episode 3726	Average Score: 0.02
epsilon: 1.841508999997677
Episode 3727	Average Score: 0.02
epsilon: 1.8396959999976783
Episode 3728	Average Score: 0.02
epsilon: 1.8390099999976788
Episode 3729	Average Score: 0.03
epsilon: 1.83734399999768
Episode 3730	Average Score: 0.03
epsilon: 1.8366579999976804
Episode 3731	Average Score: 0.02
epsilon: 1.835971999997681
Episode 3732	Average Score: 0.02
epsilon: 1.8352859999976814
Episode 3733	Average Score: 0.02
epsilon: 1.834550999997682
Episode 3734	Average Score: 0.02
epsilon: 1.8338649999976824
Episode 3735	Average Score: 0.02
epsilon: 1.8331789999976829
Episode 3736	Average Score: 0.02
epsilon: 1.8324929999976833
Episode 3737	Average Score: 0.02
epsilon: 1.8318069999976838
Episode 3738	Average Score: 0.02
epsilon: 1.830140999997685
Episode 3739	Average Score: 0.02
epsilon: 1.8294549999976855
Episode 3740	Average Score: 0.02
epsilon: 1.828768999997686
Episode 3741	Average Score: 0.02
epsilon: 1.8280829999976864
Episode 3742	Average Score: 0.02
epsilon: 1.827396999997687
Episode 3743	Average Score: 0.03
epsilon: 1.825730999997688
Episode 3744	Average Score: 0.03
epsilon: 1.8250939999976885
Episode 3745	Average Score: 0.02
epsilon: 1.824358999997689
Episode 3746	Average Score: 0.02
epsilon: 1.8236729999976895
Episode 3747	Average Score: 0.02
epsilon: 1.82298699999769
Episode 3748	Average Score: 0.02
epsilon: 1.8223009999976905
Episode 3749	Average Score: 0.02
epsilon: 1.821614999997691
Episode 3750	Average Score: 0.02
epsilon: 1.8208799999976915
Episode 3751	Average Score: 0.02
epsilon: 1.820193999997692
Episode 3752	Average Score: 0.02
epsilon: 1.818576999997693
Episode 3753	Average Score: 0.02
epsilon: 1.8178909999976935
Episode 3754	Average Score: 0.02
epsilon: 1.817204999997694
Episode 3755	Average Score: 0.02
epsilon: 1.8164699999976945
Episode 3756	Average Score: 0.02
epsilon: 1.815783999997695
Episode 3757	Average Score: 0.02
epsilon: 1.8150979999976955
Episode 3758	Average Score: 0.02
epsilon: 1.814411999997696
Episode 3759	Average Score: 0.02
epsilon: 1.8137259999976965
Episode 3760	Average Score: 0.02
epsilon: 1.812990999997697
Episode 3761	Average Score: 0.02
epsilon: 1.8123049999976975
Episode 3762	Average Score: 0.02
epsilon: 1.811618999997698
Episode 3763	Average Score: 0.02
epsilon: 1.8109329999976984
Episode 3764	Average Score: 0.02
epsilon: 1.810246999997699
Episode 3765	Average Score: 0.02
epsilon: 1.8095119999976994
Episode 3766	Average Score: 0.02
epsilon: 1.8088749999976999
Episode 3767	Average Score: 0.02
epsilon: 1.8076989999977007
Episode 3768	Average Score: 0.02
epsilon: 1.8070129999977012
Episode 3769	Average Score: 0.02
epsilon: 1.8052979999977024
Episode 3770	Average Score: 0.02
epsilon: 1.804464999997703
Episode 3771	Average Score: 0.02
epsilon: 1.8037789999977034
Episode 3772	Average Score: 0.02
epsilon: 1.803092999997704
Episode 3773	Average Score: 0.02
epsilon: 1.8024069999977044
Episode 3774	Average Score: 0.02
epsilon: 1.8016719999977049
Episode 3775	Average Score: 0.02
epsilon: 1.8009859999977054
Episode 3776	Average Score: 0.02
epsilon: 1.8002999999977058
Episode 3777	Average Score: 0.02
epsilon: 1.7996139999977063
Episode 3778	Average Score: 0.02
epsilon: 1.7989279999977068
Episode 3779	Average Score: 0.02
epsilon: 1.7981929999977073
Episode 3780	Average Score: 0.02
epsilon: 1.7975069999977078
Episode 3781	Average Score: 0.02
epsilon: 1.7968699999977082
Episode 3782	Average Score: 0.02
epsilon: 1.7961349999977088
Episode 3783	Average Score: 0.02
epsilon: 1.7954489999977092
Episode 3784	Average Score: 0.02
epsilon: 1.7939299999977103
Episode 3785	Average Score: 0.02
epsilon: 1.7932439999977108
Episode 3786	Average Score: 0.02
epsilon: 1.7925579999977113
Episode 3787	Average Score: 0.02
epsilon: 1.791430999997712
Episode 3788	Average Score: 0.02
epsilon: 1.7906959999977126
Episode 3789	Average Score: 0.02
epsilon: 1.788490999997714
Episode 3790	Average Score: 0.02
epsilon: 1.7859919999977159
Episode 3791	Average Score: 0.02
epsilon: 1.7852079999977164
Episode 3792	Average Score: 0.02
epsilon: 1.7845219999977169
Episode 3793	Average Score: 0.02
epsilon: 1.7820719999977186
Episode 3794	Average Score: 0.02
epsilon: 1.781385999997719
Episode 3795	Average Score: 0.02
epsilon: 1.7806999999977196
Episode 3796	Average Score: 0.02
epsilon: 1.78001399999772
Episode 3797	Average Score: 0.02
epsilon: 1.7793279999977205
Episode 3798	Average Score: 0.02
epsilon: 1.7778089999977216
Episode 3799	Average Score: 0.02
epsilon: 1.777122999997722
Episode 3800	Average Score: 0.02
epsilon: 1.7755549999977231
Episode 3801	Average Score: 0.02
epsilon: 1.7748689999977236
Episode 3802	Average Score: 0.02
epsilon: 1.774182999997724
Episode 3803	Average Score: 0.02
epsilon: 1.7734969999977246
Episode 3804	Average Score: 0.02
epsilon: 1.772761999997725
Episode 3805	Average Score: 0.02
epsilon: 1.7712429999977262
Episode 3806	Average Score: 0.02
epsilon: 1.7705569999977266
Episode 3807	Average Score: 0.02
epsilon: 1.7698219999977272
Episode 3808	Average Score: 0.02
epsilon: 1.7691359999977276
Episode 3809	Average Score: 0.02
epsilon: 1.7684499999977281
Episode 3810	Average Score: 0.02
epsilon: 1.7678129999977286
Episode 3811	Average Score: 0.02
epsilon: 1.767077999997729
Episode 3812	Average Score: 0.02
epsilon: 1.7636479999977315
Episode 3813	Average Score: 0.02
epsilon: 1.762912999997732
Episode 3814	Average Score: 0.02
epsilon: 1.761442999997733
Episode 3815	Average Score: 0.02
epsilon: 1.759923999997734
Episode 3816	Average Score: 0.02
epsilon: 1.7567389999977363
Episode 3817	Average Score: 0.02
epsilon: 1.7559549999977369
Episode 3818	Average Score: 0.02
epsilon: 1.7552689999977373
Episode 3819	Average Score: 0.02
epsilon: 1.7545829999977378
Episode 3820	Average Score: 0.02
epsilon: 1.7538969999977383
Episode 3821	Average Score: 0.02
epsilon: 1.7532109999977388
Episode 3822	Average Score: 0.02
epsilon: 1.7516429999977399
Episode 3823	Average Score: 0.02
epsilon: 1.7509569999977403
Episode 3824	Average Score: 0.02
epsilon: 1.7502709999977408
Episode 3825	Average Score: 0.02
epsilon: 1.748555999997742
Episode 3826	Average Score: 0.02
epsilon: 1.7477229999977426
Episode 3827	Average Score: 0.02
epsilon: 1.7469879999977431
Episode 3828	Average Score: 0.02
epsilon: 1.7446849999977447
Episode 3829	Average Score: 0.02
epsilon: 1.7439499999977452
Episode 3830	Average Score: 0.02
epsilon: 1.7423819999977463
Episode 3831	Average Score: 0.02
epsilon: 1.7416469999977469
Episode 3832	Average Score: 0.02
epsilon: 1.7409609999977473
Episode 3833	Average Score: 0.02
epsilon: 1.7402749999977478
Episode 3834	Average Score: 0.02
epsilon: 1.738706999997749
Episode 3835	Average Score: 0.02
epsilon: 1.7380209999977494
Episode 3836	Average Score: 0.02
epsilon: 1.7373349999977499
Episode 3837	Average Score: 0.02
epsilon: 1.7366489999977504
Episode 3838	Average Score: 0.02
epsilon: 1.7351789999977514
Episode 3839	Average Score: 0.02
epsilon: 1.7344929999977519
Episode 3840	Average Score: 0.02
epsilon: 1.7322389999977534
Episode 3841	Average Score: 0.02
epsilon: 1.731552999997754
Episode 3842	Average Score: 0.02
epsilon: 1.7308179999977544
Episode 3843	Average Score: 0.02
epsilon: 1.730131999997755
Episode 3844	Average Score: 0.02
epsilon: 1.728612999997756
Episode 3845	Average Score: 0.02
epsilon: 1.7279269999977565
Episode 3846	Average Score: 0.02
epsilon: 1.727240999997757
Episode 3847	Average Score: 0.02
epsilon: 1.7265549999977574
Episode 3848	Average Score: 0.02
epsilon: 1.725868999997758
Episode 3849	Average Score: 0.02
epsilon: 1.724300999997759
Episode 3850	Average Score: 0.02
epsilon: 1.7226839999977601
Episode 3851	Average Score: 0.02
epsilon: 1.7219979999977606
Episode 3852	Average Score: 0.02
epsilon: 1.721311999997761
Episode 3853	Average Score: 0.02
epsilon: 1.7196949999977622
Episode 3854	Average Score: 0.03
epsilon: 1.7178329999977635
Episode 3855	Average Score: 0.03
epsilon: 1.714157999997766
Episode 3856	Average Score: 0.03
epsilon: 1.7134719999977666
Episode 3857	Average Score: 0.03
epsilon: 1.712785999997767
Episode 3858	Average Score: 0.03
epsilon: 1.7120999999977675
Episode 3859	Average Score: 0.03
epsilon: 1.711413999997768
Episode 3860	Average Score: 0.03
epsilon: 1.7106789999977685
Episode 3861	Average Score: 0.03
epsilon: 1.709992999997769
Episode 3862	Average Score: 0.03
epsilon: 1.7093069999977695
Episode 3863	Average Score: 0.03
epsilon: 1.70862099999777
Episode 3864	Average Score: 0.03
epsilon: 1.7079349999977704
Episode 3865	Average Score: 0.03
epsilon: 1.707248999997771
Episode 3866	Average Score: 0.03
epsilon: 1.7040149999977732
Episode 3867	Average Score: 0.03
epsilon: 1.7033289999977737
Episode 3868	Average Score: 0.03
epsilon: 1.7017119999977748
Episode 3869	Average Score: 0.03
epsilon: 1.7010259999977753
Episode 3870	Average Score: 0.03
epsilon: 1.7003399999977757
Episode 3871	Average Score: 0.03
epsilon: 1.6996539999977762
Episode 3872	Average Score: 0.03
epsilon: 1.6981839999977772
Episode 3873	Average Score: 0.03
epsilon: 1.6967139999977783
Episode 3874	Average Score: 0.03
epsilon: 1.6960279999977788
Episode 3875	Average Score: 0.03
epsilon: 1.6945089999977798
Episode 3876	Average Score: 0.03
epsilon: 1.6938229999977803
Episode 3877	Average Score: 0.03
epsilon: 1.6931369999977808
Episode 3878	Average Score: 0.03
epsilon: 1.6916179999977818
Episode 3879	Average Score: 0.03
epsilon: 1.6908829999977824
Episode 3880	Average Score: 0.03
epsilon: 1.6902459999977828
Episode 3881	Average Score: 0.03
epsilon: 1.6895109999977833
Episode 3882	Average Score: 0.03
epsilon: 1.6888249999977838
Episode 3883	Average Score: 0.03
epsilon: 1.6873059999977849
Episode 3884	Average Score: 0.03
epsilon: 1.6865709999977854
Episode 3885	Average Score: 0.03
epsilon: 1.6858849999977858
Episode 3886	Average Score: 0.03
epsilon: 1.6851989999977863
Episode 3887	Average Score: 0.03
epsilon: 1.6835819999977875
Episode 3888	Average Score: 0.03
epsilon: 1.6821119999977885
Episode 3889	Average Score: 0.03
epsilon: 1.6805929999977895
Episode 3890	Average Score: 0.03
epsilon: 1.67990699999779
Episode 3891	Average Score: 0.03
epsilon: 1.6792209999977905
Episode 3892	Average Score: 0.03
epsilon: 1.678534999997791
Episode 3893	Average Score: 0.03
epsilon: 1.677015999997792
Episode 3894	Average Score: 0.03
epsilon: 1.6744679999977938
Episode 3895	Average Score: 0.03
epsilon: 1.6737819999977943
Episode 3896	Average Score: 0.03
epsilon: 1.6730959999977948
Episode 3897	Average Score: 0.03
epsilon: 1.6724099999977953
Episode 3898	Average Score: 0.03
epsilon: 1.6716749999977958
Episode 3899	Average Score: 0.03
epsilon: 1.6709399999977963
Episode 3900	Average Score: 0.03
epsilon: 1.6703029999977967
Episode 3901	Average Score: 0.03
epsilon: 1.6695679999977973
Episode 3902	Average Score: 0.03
epsilon: 1.6688819999977977
Episode 3903	Average Score: 0.03
epsilon: 1.6651089999978004
Episode 3904	Average Score: 0.03
epsilon: 1.6631489999978017
Episode 3905	Average Score: 0.03
epsilon: 1.6624139999978023
Episode 3906	Average Score: 0.03
epsilon: 1.6602089999978038
Episode 3907	Average Score: 0.03
epsilon: 1.6595229999978043
Episode 3908	Average Score: 0.03
epsilon: 1.6588369999978048
Episode 3909	Average Score: 0.04
epsilon: 1.6573669999978058
Episode 3910	Average Score: 0.04
epsilon: 1.6558969999978068
Episode 3911	Average Score: 0.04
epsilon: 1.6552109999978073
Episode 3912	Average Score: 0.03
epsilon: 1.6545249999978078
Episode 3913	Average Score: 0.03
epsilon: 1.6538879999978082
Episode 3914	Average Score: 0.03
epsilon: 1.6522219999978094
Episode 3915	Average Score: 0.03
epsilon: 1.6506539999978105
Episode 3916	Average Score: 0.03
epsilon: 1.6490369999978116
Episode 3917	Average Score: 0.03
epsilon: 1.648350999997812
Episode 3918	Average Score: 0.03
epsilon: 1.6476159999978126
Episode 3919	Average Score: 0.03
epsilon: 1.646929999997813
Episode 3920	Average Score: 0.03
epsilon: 1.6444799999978148
Episode 3921	Average Score: 0.04
epsilon: 1.6430589999978158
Episode 3922	Average Score: 0.04
epsilon: 1.6423729999978163
Episode 3923	Average Score: 0.04
epsilon: 1.6407559999978174
Episode 3924	Average Score: 0.04
epsilon: 1.6400699999978179
Episode 3925	Average Score: 0.04
epsilon: 1.6365419999978204
Episode 3926	Average Score: 0.04
epsilon: 1.6358559999978208
Episode 3927	Average Score: 0.04
epsilon: 1.6330629999978228
Episode 3928	Average Score: 0.04
epsilon: 1.6323769999978233
Episode 3929	Average Score: 0.04
epsilon: 1.629828999997825
Episode 3930	Average Score: 0.04
epsilon: 1.6291429999978255
Episode 3931	Average Score: 0.04
epsilon: 1.628456999997826
Episode 3932	Average Score: 0.04
epsilon: 1.6277709999978265
Episode 3933	Average Score: 0.04
epsilon: 1.627084999997827
Episode 3934	Average Score: 0.04
epsilon: 1.6263499999978275
Episode 3935	Average Score: 0.04
epsilon: 1.6248799999978285
Episode 3936	Average Score: 0.04
epsilon: 1.624193999997829
Episode 3937	Average Score: 0.04
epsilon: 1.6235569999978294
Episode 3938	Average Score: 0.04
epsilon: 1.62287099999783
Episode 3939	Average Score: 0.04
epsilon: 1.6221849999978304
Episode 3940	Average Score: 0.04
epsilon: 1.6204699999978316
Episode 3941	Average Score: 0.04
epsilon: 1.619783999997832
Episode 3942	Average Score: 0.04
epsilon: 1.6182159999978332
Episode 3943	Average Score: 0.04
epsilon: 1.6174809999978337
Episode 3944	Average Score: 0.04
epsilon: 1.6167949999978342
Episode 3945	Average Score: 0.04
epsilon: 1.6161089999978346
Episode 3946	Average Score: 0.04
epsilon: 1.6136099999978364
Episode 3947	Average Score: 0.04
epsilon: 1.6129239999978369
Episode 3948	Average Score: 0.04
epsilon: 1.6121889999978374
Episode 3949	Average Score: 0.04
epsilon: 1.609836999997839
Episode 3950	Average Score: 0.04
epsilon: 1.6058679999978418
Episode 3951	Average Score: 0.04
epsilon: 1.6051819999978423
Episode 3952	Average Score: 0.04
epsilon: 1.6044959999978428
Episode 3953	Average Score: 0.04
epsilon: 1.6038099999978432
Episode 3954	Average Score: 0.04
epsilon: 1.6008699999978453
Episode 3955	Average Score: 0.04
epsilon: 1.6001839999978458
Episode 3956	Average Score: 0.04
epsilon: 1.5976359999978476
Episode 3957	Average Score: 0.04
epsilon: 1.5950389999978494
Episode 3958	Average Score: 0.04
epsilon: 1.5933729999978505
Episode 3959	Average Score: 0.04
epsilon: 1.592686999997851
Episode 3960	Average Score: 0.04
epsilon: 1.5920009999978515
Episode 3961	Average Score: 0.04
epsilon: 1.5894529999978533
Episode 3962	Average Score: 0.04
epsilon: 1.5887669999978538
Episode 3963	Average Score: 0.04
epsilon: 1.5880319999978543
Episode 3964	Average Score: 0.04
epsilon: 1.5873949999978547
Episode 3965	Average Score: 0.04
epsilon: 1.5857779999978558
Episode 3966	Average Score: 0.04
epsilon: 1.5851409999978563
Episode 3967	Average Score: 0.04
epsilon: 1.5844059999978568
Episode 3968	Average Score: 0.04
epsilon: 1.5819069999978586
Episode 3969	Average Score: 0.04
epsilon: 1.5803879999978596
Episode 3970	Average Score: 0.04
epsilon: 1.5788199999978607
Episode 3971	Average Score: 0.04
epsilon: 1.5780849999978612
Episode 3972	Average Score: 0.04
epsilon: 1.575536999997863
Episode 3973	Average Score: 0.04
epsilon: 1.5730379999978648
Episode 3974	Average Score: 0.04
epsilon: 1.5723519999978652
Episode 3975	Average Score: 0.04
epsilon: 1.5716659999978657
Episode 3976	Average Score: 0.04
epsilon: 1.5709799999978662
Episode 3977	Average Score: 0.04
epsilon: 1.5678929999978684
Episode 3978	Average Score: 0.04
epsilon: 1.56539399999787
Episode 3979	Average Score: 0.04
epsilon: 1.5647079999978706
Episode 3980	Average Score: 0.04
epsilon: 1.5629439999978718
Episode 3981	Average Score: 0.04
epsilon: 1.5622579999978723
Episode 3982	Average Score: 0.04
epsilon: 1.559758999997874
Episode 3983	Average Score: 0.04
epsilon: 1.5590239999978746
Episode 3984	Average Score: 0.04
epsilon: 1.558337999997875
Episode 3985	Average Score: 0.05
epsilon: 1.5524089999978792
Episode 3986	Average Score: 0.05
epsilon: 1.5517229999978797
Episode 3987	Average Score: 0.05
epsilon: 1.5510369999978801
Episode 3988	Average Score: 0.05
epsilon: 1.548488999997882
Episode 3989	Average Score: 0.04
epsilon: 1.5478029999978824
Episode 3990	Average Score: 0.05
epsilon: 1.5462349999978835
Episode 3991	Average Score: 0.05
epsilon: 1.5405999999978874
Episode 3992	Average Score: 0.05
epsilon: 1.539913999997888
Episode 3993	Average Score: 0.05
epsilon: 1.5392279999978884
Episode 3994	Average Score: 0.05
epsilon: 1.5376109999978895
Episode 3995	Average Score: 0.05
epsilon: 1.5356999999978909
Episode 3996	Average Score: 0.05
epsilon: 1.5350139999978913
Episode 3997	Average Score: 0.05
epsilon: 1.5342789999978919
Episode 3998	Average Score: 0.05
epsilon: 1.5328089999978929
Episode 3999	Average Score: 0.05
epsilon: 1.5321229999978934
Episode 4000	Average Score: 0.05
epsilon: 1.5314369999978938
Episode 4001	Average Score: 0.05
epsilon: 1.5307019999978944
Episode 4002	Average Score: 0.05
epsilon: 1.5281539999978961
Episode 4003	Average Score: 0.05
epsilon: 1.5275169999978966
Episode 4004	Average Score: 0.05
epsilon: 1.5234009999978995
Episode 4005	Average Score: 0.05
epsilon: 1.5208529999979012
Episode 4006	Average Score: 0.05
epsilon: 1.5201179999979018
Episode 4007	Average Score: 0.05
epsilon: 1.5194319999979022
Episode 4008	Average Score: 0.05
epsilon: 1.5187949999979027
Episode 4009	Average Score: 0.05
epsilon: 1.5181089999979032
Episode 4010	Average Score: 0.05
epsilon: 1.5174229999979036
Episode 4011	Average Score: 0.05
epsilon: 1.5167369999979041
Episode 4012	Average Score: 0.05
epsilon: 1.5160019999979046
Episode 4013	Average Score: 0.05
epsilon: 1.5145319999979057
Episode 4014	Average Score: 0.05
epsilon: 1.5138459999979061
Episode 4015	Average Score: 0.05
epsilon: 1.5114449999979078
Episode 4016	Average Score: 0.05
epsilon: 1.5088969999979096
Episode 4017	Average Score: 0.05
epsilon: 1.5073779999979107
Episode 4018	Average Score: 0.05
epsilon: 1.5059079999979117
Episode 4019	Average Score: 0.05
epsilon: 1.5044379999979127
Episode 4020	Average Score: 0.05
epsilon: 1.5037519999979132
Episode 4021	Average Score: 0.05
epsilon: 1.5022329999979143
Episode 4022	Average Score: 0.05
epsilon: 1.5015959999979147
Episode 4023	Average Score: 0.05
epsilon: 1.5001259999979157
Episode 4024	Average Score: 0.05
epsilon: 1.4994399999979162
Episode 4025	Average Score: 0.05
epsilon: 1.4978719999979173
Episode 4026	Average Score: 0.05
epsilon: 1.4971859999979178
Episode 4027	Average Score: 0.05
epsilon: 1.4965489999979182
Episode 4028	Average Score: 0.05
epsilon: 1.4958629999979187
Episode 4029	Average Score: 0.05
epsilon: 1.4943439999979198
Episode 4030	Average Score: 0.05
epsilon: 1.4908159999979222
Episode 4031	Average Score: 0.05
epsilon: 1.4892969999979233
Episode 4032	Average Score: 0.05
epsilon: 1.4886109999979238
Episode 4033	Average Score: 0.05
epsilon: 1.4878759999979243
Episode 4034	Average Score: 0.05
epsilon: 1.4871409999979248
Episode 4035	Average Score: 0.05
epsilon: 1.4864549999979253
Episode 4036	Average Score: 0.05
epsilon: 1.4858179999979257
Episode 4037	Average Score: 0.05
epsilon: 1.483955999997927
Episode 4038	Average Score: 0.05
epsilon: 1.4832699999979275
Episode 4039	Average Score: 0.05
epsilon: 1.482583999997928
Episode 4040	Average Score: 0.05
epsilon: 1.481113999997929
Episode 4041	Average Score: 0.05
epsilon: 1.476948999997932
Episode 4042	Average Score: 0.05
epsilon: 1.4762629999979324
Episode 4043	Average Score: 0.05
epsilon: 1.475576999997933
Episode 4044	Average Score: 0.05
epsilon: 1.4748909999979334
Episode 4045	Average Score: 0.05
epsilon: 1.471215999997936
Episode 4046	Average Score: 0.05
epsilon: 1.469696999997937
Episode 4047	Average Score: 0.05
epsilon: 1.4671489999979388
Episode 4048	Average Score: 0.06
epsilon: 1.4646009999979406
Episode 4049	Average Score: 0.06
epsilon: 1.4606809999979433
Episode 4050	Average Score: 0.05
epsilon: 1.4599949999979438
Episode 4051	Average Score: 0.06
epsilon: 1.4574469999979456
Episode 4052	Average Score: 0.06
epsilon: 1.4558299999979467
Episode 4053	Average Score: 0.06
epsilon: 1.4550459999979473
Episode 4054	Average Score: 0.06
epsilon: 1.4537719999979481
Episode 4055	Average Score: 0.06
epsilon: 1.4500969999979507
Episode 4056	Average Score: 0.06
epsilon: 1.4494109999979512
Episode 4057	Average Score: 0.06
epsilon: 1.4486269999979517
Episode 4058	Average Score: 0.05
epsilon: 1.4479899999979522
Episode 4059	Average Score: 0.06
epsilon: 1.445441999997954
Episode 4060	Average Score: 0.06
epsilon: 1.4447069999979545
Episode 4061	Average Score: 0.06
epsilon: 1.442599999997956
Episode 4062	Average Score: 0.06
epsilon: 1.4419629999979564
Episode 4063	Average Score: 0.06
epsilon: 1.4412769999979569
Episode 4064	Average Score: 0.06
epsilon: 1.4384839999979588
Episode 4065	Average Score: 0.06
epsilon: 1.4377979999979593
Episode 4066	Average Score: 0.06
epsilon: 1.4371119999979598
Episode 4067	Average Score: 0.06
epsilon: 1.4345639999979616
Episode 4068	Average Score: 0.06
epsilon: 1.433877999997962
Episode 4069	Average Score: 0.06
epsilon: 1.432407999997963
Episode 4070	Average Score: 0.06
epsilon: 1.4309379999979641
Episode 4071	Average Score: 0.06
epsilon: 1.4302519999979646
Episode 4072	Average Score: 0.06
epsilon: 1.4286349999979657
Episode 4073	Average Score: 0.06
epsilon: 1.4278999999979662
Episode 4074	Average Score: 0.06
epsilon: 1.4262829999979674
Episode 4075	Average Score: 0.06
epsilon: 1.4255969999979679
Episode 4076	Average Score: 0.06
epsilon: 1.4229509999979697
Episode 4077	Average Score: 0.06
epsilon: 1.4222649999979702
Episode 4078	Average Score: 0.06
epsilon: 1.4178549999979733
Episode 4079	Average Score: 0.06
epsilon: 1.4171689999979737
Episode 4080	Average Score: 0.06
epsilon: 1.4164829999979742
Episode 4081	Average Score: 0.06
epsilon: 1.4157969999979747
Episode 4082	Average Score: 0.05
epsilon: 1.4151109999979752
Episode 4083	Average Score: 0.05
epsilon: 1.4144249999979757
Episode 4084	Average Score: 0.06
epsilon: 1.4128079999979768
Episode 4085	Average Score: 0.05
epsilon: 1.4121219999979773
Episode 4086	Average Score: 0.05
epsilon: 1.4113869999979778
Episode 4087	Average Score: 0.05
epsilon: 1.4088389999979796
Episode 4088	Average Score: 0.05
epsilon: 1.40810399999798
Episode 4089	Average Score: 0.05
epsilon: 1.4065849999979811
Episode 4090	Average Score: 0.05
epsilon: 1.402468999997984
Episode 4091	Average Score: 0.05
epsilon: 1.4009009999979851
Episode 4092	Average Score: 0.05
epsilon: 1.3993819999979862
Episode 4093	Average Score: 0.05
epsilon: 1.396735999997988
Episode 4094	Average Score: 0.05
epsilon: 1.3960499999979885
Episode 4095	Average Score: 0.05
epsilon: 1.3935019999979903
Episode 4096	Average Score: 0.05
epsilon: 1.390953999997992
Episode 4097	Average Score: 0.06
epsilon: 1.389483999997993
Episode 4098	Average Score: 0.05
epsilon: 1.3887489999979936
Episode 4099	Average Score: 0.06
epsilon: 1.3870339999979948
Episode 4100	Average Score: 0.06
epsilon: 1.3863969999979953
Episode 4101	Average Score: 0.06
epsilon: 1.3857109999979957
Episode 4102	Average Score: 0.05
epsilon: 1.3831629999979975
Episode 4103	Average Score: 0.05
epsilon: 1.382476999997998
Episode 4104	Average Score: 0.05
epsilon: 1.3817419999979985
Episode 4105	Average Score: 0.05
epsilon: 1.375273999998003
Episode 4106	Average Score: 0.06
epsilon: 1.3719419999980054
Episode 4107	Average Score: 0.06
epsilon: 1.3712559999980058
Episode 4108	Average Score: 0.06
epsilon: 1.3679729999980081
Episode 4109	Average Score: 0.06
epsilon: 1.3673359999980086
Episode 4110	Average Score: 0.06
epsilon: 1.366649999998009
Episode 4111	Average Score: 0.06
epsilon: 1.3651309999980101
Episode 4112	Average Score: 0.06
epsilon: 1.362533999998012
Episode 4113	Average Score: 0.06
epsilon: 1.3618479999980124
Episode 4114	Average Score: 0.06
epsilon: 1.3592509999980142
Episode 4115	Average Score: 0.06
epsilon: 1.3559189999980166
Episode 4116	Average Score: 0.06
epsilon: 1.3543999999980176
Episode 4117	Average Score: 0.06
epsilon: 1.3536649999980181
Episode 4118	Average Score: 0.06
epsilon: 1.3521949999980192
Episode 4119	Average Score: 0.06
epsilon: 1.3515089999980197
Episode 4120	Average Score: 0.06
epsilon: 1.3508229999980201
Episode 4121	Average Score: 0.06
epsilon: 1.3501369999980206
Episode 4122	Average Score: 0.06
epsilon: 1.345187999998024
Episode 4123	Average Score: 0.06
epsilon: 1.342590999998026
Episode 4124	Average Score: 0.06
epsilon: 1.3419049999980264
Episode 4125	Average Score: 0.06
epsilon: 1.3412679999980268
Episode 4126	Average Score: 0.06
epsilon: 1.3387199999980286
Episode 4127	Average Score: 0.06
epsilon: 1.3371519999980297
Episode 4128	Average Score: 0.06
epsilon: 1.3364659999980302
Episode 4129	Average Score: 0.06
epsilon: 1.3349469999980312
Episode 4130	Average Score: 0.06
epsilon: 1.332398999998033
Episode 4131	Average Score: 0.06
epsilon: 1.3317129999980335
Episode 4132	Average Score: 0.06
epsilon: 1.331026999998034
Episode 4133	Average Score: 0.06
epsilon: 1.3303409999980345
Episode 4134	Average Score: 0.06
epsilon: 1.3288219999980355
Episode 4135	Average Score: 0.06
epsilon: 1.328135999998036
Episode 4136	Average Score: 0.06
epsilon: 1.3255879999980378
Episode 4137	Average Score: 0.06
epsilon: 1.3240689999980388
Episode 4138	Average Score: 0.06
epsilon: 1.3214229999980407
Episode 4139	Average Score: 0.06
epsilon: 1.3207369999980412
Episode 4140	Average Score: 0.06
epsilon: 1.3200509999980417
Episode 4141	Average Score: 0.06
epsilon: 1.3174049999980435
Episode 4142	Average Score: 0.06
epsilon: 1.3159349999980445
Episode 4143	Average Score: 0.06
epsilon: 1.313925999998046
Episode 4144	Average Score: 0.06
epsilon: 1.312504999998047
Episode 4145	Average Score: 0.06
epsilon: 1.310936999998048
Episode 4146	Average Score: 0.06
epsilon: 1.3102509999980485
Episode 4147	Average Score: 0.06
epsilon: 1.3087319999980496
Episode 4148	Average Score: 0.06
epsilon: 1.3042239999980527
Episode 4149	Average Score: 0.06
epsilon: 1.3035379999980532
Episode 4150	Average Score: 0.06
epsilon: 1.300940999998055
Episode 4151	Average Score: 0.06
epsilon: 1.3002549999980555
Episode 4152	Average Score: 0.06
epsilon: 1.296628999998058
Episode 4153	Average Score: 0.06
epsilon: 1.2959429999980585
Episode 4154	Average Score: 0.06
epsilon: 1.295256999998059
Episode 4155	Average Score: 0.06
epsilon: 1.2945709999980595
Episode 4156	Average Score: 0.06
epsilon: 1.2929049999980606
Episode 4157	Average Score: 0.06
epsilon: 1.2904059999980624
Episode 4158	Average Score: 0.06
epsilon: 1.2878089999980642
Episode 4159	Average Score: 0.06
epsilon: 1.2869759999980648
Episode 4160	Average Score: 0.06
epsilon: 1.2862409999980653
Episode 4161	Average Score: 0.06
epsilon: 1.2847219999980664
Episode 4162	Average Score: 0.06
epsilon: 1.2839869999980669
Episode 4163	Average Score: 0.06
epsilon: 1.282467999998068
Episode 4164	Average Score: 0.06
epsilon: 1.280948999998069
Episode 4165	Average Score: 0.06
epsilon: 1.2784009999980708
Episode 4166	Average Score: 0.06
epsilon: 1.2777639999980712
Episode 4167	Average Score: 0.06
epsilon: 1.2770289999980717
Episode 4168	Average Score: 0.06
epsilon: 1.2754119999980729
Episode 4169	Average Score: 0.06
epsilon: 1.2747259999980733
Episode 4170	Average Score: 0.06
epsilon: 1.2740399999980738
Episode 4171	Average Score: 0.06
epsilon: 1.2733049999980743
Episode 4172	Average Score: 0.06
epsilon: 1.2726189999980748
Episode 4173	Average Score: 0.06
epsilon: 1.2719329999980753
Episode 4174	Average Score: 0.06
epsilon: 1.2701199999980766
Episode 4175	Average Score: 0.06
epsilon: 1.2660039999980794
Episode 4176	Average Score: 0.06
epsilon: 1.2643379999980806
Episode 4177	Average Score: 0.06
epsilon: 1.263651999998081
Episode 4178	Average Score: 0.06
epsilon: 1.2629659999980816
Episode 4179	Average Score: 0.06
epsilon: 1.259388999998084
Episode 4180	Average Score: 0.06
epsilon: 1.2567919999980859
Episode 4181	Average Score: 0.06
epsilon: 1.2561059999980864
Episode 4182	Average Score: 0.06
epsilon: 1.2554199999980868
Episode 4183	Average Score: 0.06
epsilon: 1.2547339999980873
Episode 4184	Average Score: 0.06
epsilon: 1.2539989999980878
Episode 4185	Average Score: 0.06
epsilon: 1.2533619999980883
Episode 4186	Average Score: 0.06
epsilon: 1.25086299999809
Episode 4187	Average Score: 0.06
epsilon: 1.2470409999980927
Episode 4188	Average Score: 0.07
epsilon: 1.243708999998095
Episode 4189	Average Score: 0.06
epsilon: 1.2430229999980955
Episode 4190	Average Score: 0.06
epsilon: 1.242336999998096
Episode 4191	Average Score: 0.06
epsilon: 1.2373389999980995
Episode 4192	Average Score: 0.06
epsilon: 1.2347909999981013
Episode 4193	Average Score: 0.06
epsilon: 1.232438999998103
Episode 4194	Average Score: 0.06
epsilon: 1.230870999998104
Episode 4195	Average Score: 0.06
epsilon: 1.2301359999981045
Episode 4196	Average Score: 0.06
epsilon: 1.229449999998105
Episode 4197	Average Score: 0.06
epsilon: 1.2287639999981055
Episode 4198	Average Score: 0.06
epsilon: 1.2262159999981073
Episode 4199	Average Score: 0.06
epsilon: 1.2245989999981084
Episode 4200	Average Score: 0.06
epsilon: 1.2229329999981096
Episode 4201	Average Score: 0.06
epsilon: 1.22224699999811
Episode 4202	Average Score: 0.06
epsilon: 1.2196989999981118
Episode 4203	Average Score: 0.06
epsilon: 1.2189639999981123
Episode 4204	Average Score: 0.06
epsilon: 1.2163179999981142
Episode 4205	Average Score: 0.06
epsilon: 1.2147989999981152
Episode 4206	Average Score: 0.06
epsilon: 1.212250999998117
Episode 4207	Average Score: 0.06
epsilon: 1.210731999998118
Episode 4208	Average Score: 0.06
epsilon: 1.2082819999981198
Episode 4209	Average Score: 0.06
epsilon: 1.2075959999981203
Episode 4210	Average Score: 0.06
epsilon: 1.2060279999981214
Episode 4211	Average Score: 0.06
epsilon: 1.2053419999981219
Episode 4212	Average Score: 0.06
epsilon: 1.2046559999981223
Episode 4213	Average Score: 0.06
epsilon: 1.2032839999981233
Episode 4214	Average Score: 0.06
epsilon: 1.2025979999981238
Episode 4215	Average Score: 0.06
epsilon: 1.1996579999981258
Episode 4216	Average Score: 0.06
epsilon: 1.1989719999981263
Episode 4217	Average Score: 0.06
epsilon: 1.1982859999981268
Episode 4218	Average Score: 0.06
epsilon: 1.1975999999981273
Episode 4219	Average Score: 0.06
epsilon: 1.1932879999981303
Episode 4220	Average Score: 0.06
epsilon: 1.1915239999981315
Episode 4221	Average Score: 0.06
epsilon: 1.1871139999981346
Episode 4222	Average Score: 0.06
epsilon: 1.1827039999981377
Episode 4223	Average Score: 0.06
epsilon: 1.1801559999981395
Episode 4224	Average Score: 0.06
epsilon: 1.176627999998142
Episode 4225	Average Score: 0.06
epsilon: 1.1759909999981424
Episode 4226	Average Score: 0.06
epsilon: 1.1743249999981435
Episode 4227	Average Score: 0.06
epsilon: 1.1728059999981446
Episode 4228	Average Score: 0.06
epsilon: 1.172168999998145
Episode 4229	Average Score: 0.06
epsilon: 1.1714829999981455
Episode 4230	Average Score: 0.06
epsilon: 1.167905999998148
Episode 4231	Average Score: 0.07
epsilon: 1.166386999998149
Episode 4232	Average Score: 0.07
epsilon: 1.1657009999981496
Episode 4233	Average Score: 0.07
epsilon: 1.16501499999815
Episode 4234	Average Score: 0.07
epsilon: 1.1624179999981519
Episode 4235	Average Score: 0.07
epsilon: 1.1616829999981524
Episode 4236	Average Score: 0.06
epsilon: 1.160947999998153
Episode 4237	Average Score: 0.06
epsilon: 1.1603109999981533
Episode 4238	Average Score: 0.06
epsilon: 1.1596249999981538
Episode 4239	Average Score: 0.06
epsilon: 1.158056999998155
Episode 4240	Average Score: 0.06
epsilon: 1.1573709999981554
Episode 4241	Average Score: 0.06
epsilon: 1.1566849999981559
Episode 4242	Average Score: 0.06
epsilon: 1.155165999998157
Episode 4243	Average Score: 0.06
epsilon: 1.1500699999981605
Episode 4244	Average Score: 0.06
epsilon: 1.149383999998161
Episode 4245	Average Score: 0.06
epsilon: 1.1461989999981632
Episode 4246	Average Score: 0.06
epsilon: 1.1455619999981637
Episode 4247	Average Score: 0.06
epsilon: 1.1448269999981642
Episode 4248	Average Score: 0.06
epsilon: 1.1434059999981652
Episode 4249	Average Score: 0.06
epsilon: 1.1427199999981656
Episode 4250	Average Score: 0.06
epsilon: 1.1401719999981674
Episode 4251	Average Score: 0.06
epsilon: 1.1386529999981685
Episode 4252	Average Score: 0.06
epsilon: 1.1361049999981703
Episode 4253	Average Score: 0.06
epsilon: 1.130665999998174
Episode 4254	Average Score: 0.07
epsilon: 1.1289999999981752
Episode 4255	Average Score: 0.07
epsilon: 1.1254719999981777
Episode 4256	Average Score: 0.07
epsilon: 1.1247859999981782
Episode 4257	Average Score: 0.07
epsilon: 1.1240999999981787
Episode 4258	Average Score: 0.07
epsilon: 1.1204739999981812
Episode 4259	Average Score: 0.07
epsilon: 1.1174849999981833
Episode 4260	Average Score: 0.07
epsilon: 1.114985999998185
Episode 4261	Average Score: 0.07
epsilon: 1.1133689999981862
Episode 4262	Average Score: 0.07
epsilon: 1.110673999998188
Episode 4263	Average Score: 0.07
epsilon: 1.1083709999981897
Episode 4264	Average Score: 0.07
epsilon: 1.1076849999981901
Episode 4265	Average Score: 0.07
epsilon: 1.1039609999981927
Episode 4266	Average Score: 0.07
epsilon: 1.0995509999981958
Episode 4267	Average Score: 0.07
epsilon: 1.0988159999981963
Episode 4268	Average Score: 0.07
epsilon: 1.0972479999981974
Episode 4269	Average Score: 0.07
epsilon: 1.0938179999981998
Episode 4270	Average Score: 0.07
epsilon: 1.0912699999982016
Episode 4271	Average Score: 0.08
epsilon: 1.0886239999982035
Episode 4272	Average Score: 0.08
epsilon: 1.0833319999982072
Episode 4273	Average Score: 0.08
epsilon: 1.0818129999982082
Episode 4274	Average Score: 0.08
epsilon: 1.0802939999982093
Episode 4275	Average Score: 0.08
epsilon: 1.0796079999982098
Episode 4276	Average Score: 0.08
epsilon: 1.0780889999982108
Episode 4277	Average Score: 0.08
epsilon: 1.0754919999982127
Episode 4278	Average Score: 0.08
epsilon: 1.0748059999982131
Episode 4279	Average Score: 0.08
epsilon: 1.0723559999982148
Episode 4280	Average Score: 0.08
epsilon: 1.067945999998218
Episode 4281	Average Score: 0.08
epsilon: 1.0662309999982191
Episode 4282	Average Score: 0.08
epsilon: 1.0610369999982228
Episode 4283	Average Score: 0.08
epsilon: 1.0595179999982238
Episode 4284	Average Score: 0.08
epsilon: 1.0588319999982243
Episode 4285	Average Score: 0.08
epsilon: 1.0573129999982254
Episode 4286	Average Score: 0.08
epsilon: 1.0558429999982264
Episode 4287	Average Score: 0.08
epsilon: 1.052118999998229
Episode 4288	Average Score: 0.08
epsilon: 1.0514329999982295
Episode 4289	Average Score: 0.08
epsilon: 1.05079599999823
Episode 4290	Average Score: 0.08
epsilon: 1.0476109999982322
Episode 4291	Average Score: 0.08
epsilon: 1.0461409999982332
Episode 4292	Average Score: 0.08
epsilon: 1.0454549999982337
Episode 4293	Average Score: 0.08
epsilon: 1.0439849999982347
Episode 4294	Average Score: 0.08
epsilon: 1.0433479999982351
Episode 4295	Average Score: 0.08
epsilon: 1.0418289999982362
Episode 4296	Average Score: 0.08
epsilon: 1.0384969999982385
Episode 4297	Average Score: 0.08
epsilon: 1.0368799999982397
Episode 4298	Average Score: 0.08
epsilon: 1.0338909999982417
Episode 4299	Average Score: 0.08
epsilon: 1.030705999998244
Episode 4300	Average Score: 0.08
epsilon: 1.0297749999982446
Episode 4301	Average Score: 0.08
epsilon: 1.0283049999982457
Episode 4302	Average Score: 0.08
epsilon: 1.0276189999982461
Episode 4303	Average Score: 0.08
epsilon: 1.0261489999982472
Episode 4304	Average Score: 0.08
epsilon: 1.0252179999982478
Episode 4305	Average Score: 0.08
epsilon: 1.023600999998249
Episode 4306	Average Score: 0.08
epsilon: 1.0229639999982494
Episode 4307	Average Score: 0.08
epsilon: 1.020758999998251
Episode 4308	Average Score: 0.09
epsilon: 1.0110079999982577
Episode 4309	Average Score: 0.09
epsilon: 1.0079699999982599
Episode 4310	Average Score: 0.09
epsilon: 1.0031189999982633
Episode 4311	Average Score: 0.09
epsilon: 1.0024329999982637
Episode 4312	Average Score: 0.09
epsilon: 0.9999339999982655
Episode 4313	Average Score: 0.09
epsilon: 0.999198999998266
Episode 4314	Average Score: 0.09
epsilon: 0.9985619999982664
Episode 4315	Average Score: 0.09
epsilon: 0.9941029999982696
Episode 4316	Average Score: 0.09
epsilon: 0.99341699999827
Episode 4317	Average Score: 0.09
epsilon: 0.9927309999982705
Episode 4318	Average Score: 0.09
epsilon: 0.992044999998271
Episode 4319	Average Score: 0.09
epsilon: 0.9914079999982714
Episode 4320	Average Score: 0.09
epsilon: 0.9891049999982731
Episode 4321	Average Score: 0.09
epsilon: 0.9884189999982735
Episode 4322	Average Score: 0.09
epsilon: 0.9858219999982754
Episode 4323	Average Score: 0.09
epsilon: 0.9851359999982758
Episode 4324	Average Score: 0.08
epsilon: 0.983518999998277
Episode 4325	Average Score: 0.09
epsilon: 0.9786189999982804
Episode 4326	Average Score: 0.09
epsilon: 0.976315999998282
Episode 4327	Average Score: 0.09
epsilon: 0.9756299999982825
Episode 4328	Average Score: 0.09
epsilon: 0.974943999998283
Episode 4329	Average Score: 0.09
epsilon: 0.9742579999982834
Episode 4330	Average Score: 0.08
epsilon: 0.9727389999982845
Episode 4331	Average Score: 0.08
epsilon: 0.9701909999982863
Episode 4332	Average Score: 0.09
epsilon: 0.9669079999982886
Episode 4333	Average Score: 0.09
epsilon: 0.9609789999982927
Episode 4334	Average Score: 0.09
epsilon: 0.9564709999982959
Episode 4335	Average Score: 0.09
epsilon: 0.9557849999982964
Episode 4336	Average Score: 0.09
epsilon: 0.9532369999982981
Episode 4337	Average Score: 0.09
epsilon: 0.9503459999983002
Episode 4338	Average Score: 0.09
epsilon: 0.9488759999983012
Episode 4339	Average Score: 0.09
epsilon: 0.9454459999983036
Episode 4340	Average Score: 0.10
epsilon: 0.9437309999983048
Episode 4341	Average Score: 0.10
epsilon: 0.9430449999983053
Episode 4342	Average Score: 0.09
epsilon: 0.9423589999983057
Episode 4343	Average Score: 0.09
epsilon: 0.9416729999983062
Episode 4344	Average Score: 0.09
epsilon: 0.9409869999983067
Episode 4345	Average Score: 0.09
epsilon: 0.9403499999983072
Episode 4346	Average Score: 0.09
epsilon: 0.937752999998309
Episode 4347	Average Score: 0.09
epsilon: 0.9370669999983094
Episode 4348	Average Score: 0.09
epsilon: 0.9363809999983099
Episode 4349	Average Score: 0.09
epsilon: 0.9357439999983104
Episode 4350	Average Score: 0.09
epsilon: 0.933342999998312
Episode 4351	Average Score: 0.09
epsilon: 0.9308439999983138
Episode 4352	Average Score: 0.09
epsilon: 0.9281979999983156
Episode 4353	Average Score: 0.09
epsilon: 0.9237879999983187
Episode 4354	Average Score: 0.09
epsilon: 0.9193779999983218
Episode 4355	Average Score: 0.09
epsilon: 0.9122239999983268
Episode 4356	Average Score: 0.09
epsilon: 0.910508999998328
Episode 4357	Average Score: 0.09
epsilon: 0.9098229999983285
Episode 4358	Average Score: 0.09
epsilon: 0.906245999998331
Episode 4359	Average Score: 0.09
epsilon: 0.9056089999983314
Episode 4360	Average Score: 0.09
epsilon: 0.904873999998332
Episode 4361	Average Score: 0.09
epsilon: 0.900561999998335
Episode 4362	Average Score: 0.09
epsilon: 0.8998759999983355
Episode 4363	Average Score: 0.09
epsilon: 0.8984059999983365
Episode 4364	Average Score: 0.09
epsilon: 0.897719999998337
Episode 4365	Average Score: 0.09
epsilon: 0.8961029999983381
Episode 4366	Average Score: 0.09
epsilon: 0.8935549999983399
Episode 4367	Average Score: 0.09
epsilon: 0.8928689999983404
Episode 4368	Average Score: 0.09
epsilon: 0.8886549999983433
Episode 4369	Average Score: 0.09
epsilon: 0.8879689999983438
Episode 4370	Average Score: 0.09
epsilon: 0.8864499999983448
Episode 4371	Average Score: 0.09
epsilon: 0.8824809999983476
Episode 4372	Average Score: 0.09
epsilon: 0.8810109999983486
Episode 4373	Average Score: 0.08
epsilon: 0.8803249999983491
Episode 4374	Average Score: 0.08
epsilon: 0.8771399999983513
Episode 4375	Average Score: 0.09
epsilon: 0.8750329999983528
Episode 4376	Average Score: 0.09
epsilon: 0.8735629999983539
Episode 4377	Average Score: 0.08
epsilon: 0.8728769999983543
Episode 4378	Average Score: 0.09
epsilon: 0.8701819999983562
Episode 4379	Average Score: 0.09
epsilon: 0.8609209999983627
Episode 4380	Average Score: 0.09
epsilon: 0.8594509999983637
Episode 4381	Average Score: 0.09
epsilon: 0.8587649999983642
Episode 4382	Average Score: 0.08
epsilon: 0.8580789999983647
Episode 4383	Average Score: 0.08
epsilon: 0.8573929999983652
Episode 4384	Average Score: 0.08
epsilon: 0.854795999998367
Episode 4385	Average Score: 0.08
epsilon: 0.8530809999983682
Episode 4386	Average Score: 0.08
epsilon: 0.8506309999983699
Episode 4387	Average Score: 0.08
epsilon: 0.8480339999983717
Episode 4388	Average Score: 0.08
epsilon: 0.8473479999983722
Episode 4389	Average Score: 0.08
epsilon: 0.8458289999983732
Episode 4390	Average Score: 0.08
epsilon: 0.8425949999983755
Episode 4391	Average Score: 0.08
epsilon: 0.841908999998376
Episode 4392	Average Score: 0.09
epsilon: 0.8384789999983784
Episode 4393	Average Score: 0.09
epsilon: 0.83622499999838
Episode 4394	Average Score: 0.09
epsilon: 0.8336769999983817
Episode 4395	Average Score: 0.09
epsilon: 0.831814999998383
Episode 4396	Average Score: 0.09
epsilon: 0.8301489999983842
Episode 4397	Average Score: 0.08
epsilon: 0.8294629999983847
Episode 4398	Average Score: 0.09
epsilon: 0.826130999998387
Episode 4399	Average Score: 0.09
epsilon: 0.8181439999983926
Episode 4400	Average Score: 0.09
epsilon: 0.8142729999983953
Episode 4401	Average Score: 0.09
epsilon: 0.8116759999983971
Episode 4402	Average Score: 0.09
epsilon: 0.8101569999983982
Episode 4403	Average Score: 0.09
epsilon: 0.804717999998402
Episode 4404	Average Score: 0.09
epsilon: 0.7997199999984055
Episode 4405	Average Score: 0.09
epsilon: 0.7971229999984073
Episode 4406	Average Score: 0.10
epsilon: 0.7927619999984103
Episode 4407	Average Score: 0.10
epsilon: 0.7891359999984129
Episode 4408	Average Score: 0.09
epsilon: 0.7864899999984147
Episode 4409	Average Score: 0.09
epsilon: 0.7858039999984152
Episode 4410	Average Score: 0.09
epsilon: 0.7851179999984157
Episode 4411	Average Score: 0.09
epsilon: 0.7770819999984213
Episode 4412	Average Score: 0.09
epsilon: 0.7755629999984224
Episode 4413	Average Score: 0.09
epsilon: 0.7740439999984234
Episode 4414	Average Score: 0.10
epsilon: 0.770319999998426
Episode 4415	Average Score: 0.10
epsilon: 0.7671349999984283
Episode 4416	Average Score: 0.10
epsilon: 0.7622349999984317
Episode 4417	Average Score: 0.10
epsilon: 0.7587559999984341
Episode 4418	Average Score: 0.10
epsilon: 0.7581189999984346
Episode 4419	Average Score: 0.10
epsilon: 0.7553749999984365
Episode 4420	Average Score: 0.10
epsilon: 0.7538559999984376
Episode 4421	Average Score: 0.10
epsilon: 0.753169999998438
Episode 4422	Average Score: 0.10
epsilon: 0.7525329999984385
Episode 4423	Average Score: 0.10
epsilon: 0.751846999998439
Episode 4424	Average Score: 0.10
epsilon: 0.75032799999844
Episode 4425	Average Score: 0.10
epsilon: 0.7496909999984405
Episode 4426	Average Score: 0.10
epsilon: 0.749004999998441
Episode 4427	Average Score: 0.10
epsilon: 0.744643999998444
Episode 4428	Average Score: 0.10
epsilon: 0.7411649999984464
Episode 4429	Average Score: 0.10
epsilon: 0.7404789999984469
Episode 4430	Average Score: 0.10
epsilon: 0.7397929999984474
Episode 4431	Average Score: 0.10
epsilon: 0.7391069999984479
Episode 4432	Average Score: 0.10
epsilon: 0.7356279999984503
Episode 4433	Average Score: 0.10
epsilon: 0.7321979999984527
Episode 4434	Average Score: 0.09
epsilon: 0.7315609999984531
Episode 4435	Average Score: 0.09
epsilon: 0.7308749999984536
Episode 4436	Average Score: 0.09
epsilon: 0.7301889999984541
Episode 4437	Average Score: 0.09
epsilon: 0.7295029999984546
Episode 4438	Average Score: 0.09
epsilon: 0.7288169999984551
Episode 4439	Average Score: 0.09
epsilon: 0.7281309999984555
Episode 4440	Average Score: 0.09
epsilon: 0.727444999998456
Episode 4441	Average Score: 0.09
epsilon: 0.7258769999984571
Episode 4442	Average Score: 0.09
epsilon: 0.7252399999984576
Episode 4443	Average Score: 0.09
epsilon: 0.724553999998458
Episode 4444	Average Score: 0.09
epsilon: 0.7230349999984591
Episode 4445	Average Score: 0.09
epsilon: 0.718820999998462
Episode 4446	Average Score: 0.09
epsilon: 0.7181349999984625
Episode 4447	Average Score: 0.09
epsilon: 0.7166159999984636
Episode 4448	Average Score: 0.09
epsilon: 0.7135779999984657
Episode 4449	Average Score: 0.09
epsilon: 0.7110299999984675
Episode 4450	Average Score: 0.09
epsilon: 0.7099519999984683
Episode 4451	Average Score: 0.09
epsilon: 0.7082369999984695
Episode 4452	Average Score: 0.09
epsilon: 0.7057379999984712
Episode 4453	Average Score: 0.09
epsilon: 0.7050519999984717
Episode 4454	Average Score: 0.09
epsilon: 0.7043659999984722
Episode 4455	Average Score: 0.09
epsilon: 0.7003969999984749
Episode 4456	Average Score: 0.09
epsilon: 0.6997109999984754
Episode 4457	Average Score: 0.09
epsilon: 0.6965749999984776
Episode 4458	Average Score: 0.09
epsilon: 0.6921649999984807
Episode 4459	Average Score: 0.09
epsilon: 0.6914789999984812
Episode 4460	Average Score: 0.09
epsilon: 0.6894699999984826
Episode 4461	Average Score: 0.09
epsilon: 0.688783999998483
Episode 4462	Average Score: 0.09
epsilon: 0.6880979999984835
Episode 4463	Average Score: 0.09
epsilon: 0.6838839999984865
Episode 4464	Average Score: 0.09
epsilon: 0.683197999998487
Episode 4465	Average Score: 0.09
epsilon: 0.6750149999984927
Episode 4466	Average Score: 0.09
epsilon: 0.6698699999984963
Episode 4467	Average Score: 0.09
epsilon: 0.6691839999984968
Episode 4468	Average Score: 0.09
epsilon: 0.6677139999984978
Episode 4469	Average Score: 0.10
epsilon: 0.6634019999985008
Episode 4470	Average Score: 0.10
epsilon: 0.6597759999985033
Episode 4471	Average Score: 0.10
epsilon: 0.6535529999985077
Episode 4472	Average Score: 0.10
epsilon: 0.6509559999985095
Episode 4473	Average Score: 0.10
epsilon: 0.65031899999851
Episode 4474	Average Score: 0.10
epsilon: 0.6460559999985129
Episode 4475	Average Score: 0.10
epsilon: 0.6435569999985147
Episode 4476	Average Score: 0.10
epsilon: 0.6419889999985158
Episode 4477	Average Score: 0.10
epsilon: 0.6404209999985169
Episode 4478	Average Score: 0.10
epsilon: 0.6378239999985187
Episode 4479	Average Score: 0.10
epsilon: 0.6363539999985197
Episode 4480	Average Score: 0.10
epsilon: 0.6348349999985208
Episode 4481	Average Score: 0.10
epsilon: 0.625867999998527
Episode 4482	Average Score: 0.10
epsilon: 0.6224379999985294
Episode 4483	Average Score: 0.10
epsilon: 0.6217519999985299
Episode 4484	Average Score: 0.10
epsilon: 0.6192529999985317
Episode 4485	Average Score: 0.10
epsilon: 0.6158229999985341
Episode 4486	Average Score: 0.10
epsilon: 0.6151859999985345
Episode 4487	Average Score: 0.10
epsilon: 0.614499999998535
Episode 4488	Average Score: 0.10
epsilon: 0.6114129999985372
Episode 4489	Average Score: 0.10
epsilon: 0.6082279999985394
Episode 4490	Average Score: 0.10
epsilon: 0.6021519999985436
Episode 4491	Average Score: 0.10
epsilon: 0.6015149999985441
Episode 4492	Average Score: 0.10
epsilon: 0.5989669999985459
Episode 4493	Average Score: 0.10
epsilon: 0.5982809999985463
Episode 4494	Average Score: 0.10
epsilon: 0.5956349999985482
Episode 4495	Average Score: 0.10
epsilon: 0.5883829999985533
Episode 4496	Average Score: 0.11
epsilon: 0.5848549999985557
Episode 4497	Average Score: 0.11
epsilon: 0.5822579999985575
Episode 4498	Average Score: 0.10
epsilon: 0.581571999998558
Episode 4499	Average Score: 0.10
epsilon: 0.5766719999985614
Episode 4500	Average Score: 0.10
epsilon: 0.5735359999985636
Episode 4501	Average Score: 0.10
epsilon: 0.5728499999985641
Episode 4502	Average Score: 0.10
epsilon: 0.570203999998566
Episode 4503	Average Score: 0.10
epsilon: 0.5695179999985664
Episode 4504	Average Score: 0.10
epsilon: 0.5641279999985702
Episode 4505	Average Score: 0.10
epsilon: 0.5634909999985707
Episode 4506	Average Score: 0.10
epsilon: 0.5608939999985725
Episode 4507	Average Score: 0.10
epsilon: 0.5566309999985755
Episode 4508	Average Score: 0.10
epsilon: 0.5540829999985772
Episode 4509	Average Score: 0.10
epsilon: 0.5507999999985795
Episode 4510	Average Score: 0.10
epsilon: 0.548643999998581
Episode 4511	Average Score: 0.10
epsilon: 0.5479579999985815
Episode 4512	Average Score: 0.10
epsilon: 0.5434009999985847
Episode 4513	Average Score: 0.10
epsilon: 0.5419309999985857
Episode 4514	Average Score: 0.10
epsilon: 0.5391869999985877
Episode 4515	Average Score: 0.10
epsilon: 0.5376679999985887
Episode 4516	Average Score: 0.10
epsilon: 0.534433999998591
Episode 4517	Average Score: 0.10
epsilon: 0.5318369999985928
Episode 4518	Average Score: 0.10
epsilon: 0.5247319999985978
Episode 4519	Average Score: 0.10
epsilon: 0.5240459999985982
Episode 4520	Average Score: 0.10
epsilon: 0.5208609999986005
Episode 4521	Average Score: 0.10
epsilon: 0.520174999998601
Episode 4522	Average Score: 0.10
epsilon: 0.5167449999986033
Episode 4523	Average Score: 0.10
epsilon: 0.5141479999986052
Episode 4524	Average Score: 0.10
epsilon: 0.5097869999986082
Episode 4525	Average Score: 0.10
epsilon: 0.50718999999861
Episode 4526	Average Score: 0.10
epsilon: 0.5065039999986105
Episode 4527	Average Score: 0.11
epsilon: 0.49841899999861433
Episode 4528	Average Score: 0.10
epsilon: 0.4946949999986127
Episode 4529	Average Score: 0.11
epsilon: 0.4895989999986105
Episode 4530	Average Score: 0.11
epsilon: 0.4889129999986102
Episode 4531	Average Score: 0.11
epsilon: 0.48558099999860876
Episode 4532	Average Score: 0.11
epsilon: 0.47955399999860615
Episode 4533	Average Score: 0.11
epsilon: 0.47700599999860505
Episode 4534	Average Score: 0.11
epsilon: 0.474555999998604
Episode 4535	Average Score: 0.11
epsilon: 0.4738699999986037
Episode 4536	Average Score: 0.11
epsilon: 0.4687739999986015
Episode 4537	Average Score: 0.12
epsilon: 0.46637299999860043
Episode 4538	Average Score: 0.12
epsilon: 0.4614729999985983
Episode 4539	Average Score: 0.12
epsilon: 0.4600029999985977
Episode 4540	Average Score: 0.12
epsilon: 0.45853299999859704
Episode 4541	Average Score: 0.12
epsilon: 0.4570139999985964
Episode 4542	Average Score: 0.12
epsilon: 0.4517709999985941
Episode 4543	Average Score: 0.12
epsilon: 0.44912499999859296
Episode 4544	Average Score: 0.13
epsilon: 0.44290199999859026
Episode 4545	Average Score: 0.12
epsilon: 0.44221599999858996
Episode 4546	Average Score: 0.12
epsilon: 0.44152999999858966
Episode 4547	Average Score: 0.12
epsilon: 0.4393739999985887
Episode 4548	Average Score: 0.12
epsilon: 0.43873699999858845
Episode 4549	Average Score: 0.12
epsilon: 0.4314849999985853
Episode 4550	Average Score: 0.12
epsilon: 0.4299169999985846
Episode 4551	Average Score: 0.13
epsilon: 0.42300799999858163
Episode 4552	Average Score: 0.13
epsilon: 0.4114439999985766
Episode 4553	Average Score: 0.13
epsilon: 0.40879799999857547
Episode 4554	Average Score: 0.13
epsilon: 0.40811199999857517
Episode 4555	Average Score: 0.13
epsilon: 0.40556399999857407
Episode 4556	Average Score: 0.13
epsilon: 0.4029179999985729
Episode 4557	Average Score: 0.13
epsilon: 0.40228099999857264
Episode 4558	Average Score: 0.13
epsilon: 0.3977729999985707
Episode 4559	Average Score: 0.13
epsilon: 0.39620499999857
Episode 4560	Average Score: 0.13
epsilon: 0.3946369999985693
Episode 4561	Average Score: 0.13
epsilon: 0.3914029999985679
Episode 4562	Average Score: 0.14
epsilon: 0.3883159999985666
Episode 4563	Average Score: 0.14
epsilon: 0.38498399999856514
Episode 4564	Average Score: 0.14
epsilon: 0.38155399999856365
Episode 4565	Average Score: 0.14
epsilon: 0.3809169999985634
Episode 4566	Average Score: 0.13
epsilon: 0.377731999998562
Episode 4567	Average Score: 0.14
epsilon: 0.37435099999856053
Episode 4568	Average Score: 0.13
epsilon: 0.37366499999856023
Episode 4569	Average Score: 0.13
epsilon: 0.37023499999855874
Episode 4570	Average Score: 0.13
epsilon: 0.3687649999985581
Episode 4571	Average Score: 0.13
epsilon: 0.36543299999855666
Episode 4572	Average Score: 0.13
epsilon: 0.3621009999985552
Episode 4573	Average Score: 0.13
epsilon: 0.35891599999855384
Episode 4574	Average Score: 0.14
epsilon: 0.3459309999985482
Episode 4575	Average Score: 0.14
epsilon: 0.34254999999854674
Episode 4576	Average Score: 0.14
epsilon: 0.3391689999985453
Episode 4577	Average Score: 0.14
epsilon: 0.3351019999985435
Episode 4578	Average Score: 0.14
epsilon: 0.33196599999854215
Episode 4579	Average Score: 0.15
epsilon: 0.3171189999985357
Episode 4580	Average Score: 0.15
epsilon: 0.31510999999853484
Episode 4581	Average Score: 0.15
epsilon: 0.3136399999985342
Episode 4582	Average Score: 0.15
epsilon: 0.3090829999985322
Episode 4583	Average Score: 0.15
epsilon: 0.30638799999853106
Episode 4584	Average Score: 0.15
epsilon: 0.30295799999852957
Episode 4585	Average Score: 0.15
epsilon: 0.30129199999852885
Episode 4586	Average Score: 0.15
epsilon: 0.30065499999852857
Episode 4587	Average Score: 0.15
epsilon: 0.29531399999852626
Episode 4588	Average Score: 0.15
epsilon: 0.29090399999852434
Episode 4589	Average Score: 0.15
epsilon: 0.28840499999852326
Episode 4590	Average Score: 0.15
epsilon: 0.28585699999852215
Episode 4591	Average Score: 0.15
epsilon: 0.2824759999985207
Episode 4592	Average Score: 0.15
epsilon: 0.2817899999985204
Episode 4593	Average Score: 0.15
epsilon: 0.2766939999985182
Episode 4594	Average Score: 0.15
epsilon: 0.27409699999851705
Episode 4595	Average Score: 0.15
epsilon: 0.27341099999851676
Episode 4596	Average Score: 0.15
epsilon: 0.2656689999985134
Episode 4597	Average Score: 0.15
epsilon: 0.26414999999851274
Episode 4598	Average Score: 0.15
epsilon: 0.26052399999851117
Episode 4599	Average Score: 0.15
epsilon: 0.2598869999985109
Episode 4600	Average Score: 0.15
epsilon: 0.25836799999851023
Episode 4601	Average Score: 0.15
epsilon: 0.2548889999985087
Episode 4602	Average Score: 0.15
epsilon: 0.2534189999985081
Episode 4603	Average Score: 0.16
epsilon: 0.2367099999985084
Episode 4604	Average Score: 0.16
epsilon: 0.232201999998509
Episode 4605	Average Score: 0.16
epsilon: 0.22960499999850933
Episode 4606	Average Score: 0.16
epsilon: 0.22328399999851017
Episode 4607	Average Score: 0.16
epsilon: 0.2207359999985105
Episode 4608	Average Score: 0.16
epsilon: 0.2162279999985111
Episode 4609	Average Score: 0.16
epsilon: 0.21167099999851172
Episode 4610	Average Score: 0.16
epsilon: 0.2057419999985125
Episode 4611	Average Score: 0.16
epsilon: 0.2026549999985129
Episode 4612	Average Score: 0.16
epsilon: 0.20005799999851326
Episode 4613	Average Score: 0.16
epsilon: 0.19853899999851346
Episode 4614	Average Score: 0.16
epsilon: 0.19701999999851366
Episode 4615	Average Score: 0.16
epsilon: 0.194520999998514
Episode 4616	Average Score: 0.16
epsilon: 0.19383499999851408
Episode 4617	Average Score: 0.16
epsilon: 0.18859199999851478
Episode 4618	Average Score: 0.16
epsilon: 0.1839859999985154
Episode 4619	Average Score: 0.16
epsilon: 0.17957599999851598
Episode 4620	Average Score: 0.17
epsilon: 0.1741859999985167
Episode 4621	Average Score: 0.17
epsilon: 0.1688449999985174
Episode 4622	Average Score: 0.17
epsilon: 0.1673749999985176
Episode 4623	Average Score: 0.17
epsilon: 0.16477799999851794
Episode 4624	Average Score: 0.17
epsilon: 0.1613969999985184
Episode 4625	Average Score: 0.17
epsilon: 0.15203799999851964
Episode 4626	Average Score: 0.18
epsilon: 0.14390399999852072
Episode 4627	Average Score: 0.17
epsilon: 0.13861199999852142
Episode 4628	Average Score: 0.18
epsilon: 0.12494099999852323
Episode 4629	Average Score: 0.18
epsilon: 0.12043299999852383
Episode 4630	Average Score: 0.18
epsilon: 0.11499399999852455
Episode 4631	Average Score: 0.18
epsilon: 0.11435699999852464
Episode 4632	Average Score: 0.18
epsilon: 0.10524299999852585
Episode 4633	Average Score: 0.18
epsilon: 0.1026459999985262
Episode 4634	Average Score: 0.18
epsilon: 0.1
Episode 4635	Average Score: 0.19
epsilon: 0.1
Episode 4636	Average Score: 0.18
epsilon: 0.1
Episode 4637	Average Score: 0.18
epsilon: 0.1
Episode 4638	Average Score: 0.18
epsilon: 0.1
Episode 4639	Average Score: 0.19
epsilon: 0.1
Episode 4640	Average Score: 0.19
epsilon: 0.1
Episode 4641	Average Score: 0.19
epsilon: 0.1
Episode 4642	Average Score: 0.19
epsilon: 0.1
Episode 4643	Average Score: 0.18
epsilon: 0.1
Episode 4644	Average Score: 0.18
epsilon: 0.1
Episode 4645	Average Score: 0.18
epsilon: 0.1
Episode 4646	Average Score: 0.19
epsilon: 0.1
Episode 4647	Average Score: 0.19
epsilon: 0.1
Episode 4648	Average Score: 0.19
epsilon: 0.1
Episode 4649	Average Score: 0.19
epsilon: 0.1
Episode 4650	Average Score: 0.19
epsilon: 0.1
Episode 4651	Average Score: 0.19
epsilon: 0.1
Episode 4652	Average Score: 0.18
epsilon: 0.1
Episode 4653	Average Score: 0.19
epsilon: 0.1
Episode 4654	Average Score: 0.19
epsilon: 0.1
Episode 4655	Average Score: 0.19
epsilon: 0.1
Episode 4656	Average Score: 0.19
epsilon: 0.1
Episode 4657	Average Score: 0.19
epsilon: 0.1
Episode 4658	Average Score: 0.19
epsilon: 0.1
Episode 4659	Average Score: 0.19
epsilon: 0.1
Episode 4660	Average Score: 0.19
epsilon: 0.1
Episode 4661	Average Score: 0.19
epsilon: 0.1
Episode 4662	Average Score: 0.19
epsilon: 0.1
Episode 4663	Average Score: 0.19
epsilon: 0.1
Episode 4664	Average Score: 0.19
epsilon: 0.1
Episode 4665	Average Score: 0.19
epsilon: 0.1
Episode 4666	Average Score: 0.19
epsilon: 0.1
Episode 4667	Average Score: 0.19
epsilon: 0.1
Episode 4668	Average Score: 0.19
epsilon: 0.1
Episode 4669	Average Score: 0.19
epsilon: 0.1
Episode 4670	Average Score: 0.19
epsilon: 0.1
Episode 4671	Average Score: 0.19
epsilon: 0.1
Episode 4672	Average Score: 0.20
epsilon: 0.1
Episode 4673	Average Score: 0.20
epsilon: 0.1
Episode 4674	Average Score: 0.20
epsilon: 0.1
Episode 4675	Average Score: 0.20
epsilon: 0.1
Episode 4676	Average Score: 0.20
epsilon: 0.1
Episode 4677	Average Score: 0.19
epsilon: 0.1
Episode 4678	Average Score: 0.20
epsilon: 0.1
Episode 4679	Average Score: 0.19
epsilon: 0.1
Episode 4680	Average Score: 0.19
epsilon: 0.1
Episode 4681	Average Score: 0.19
epsilon: 0.1
Episode 4682	Average Score: 0.19
epsilon: 0.1
Episode 4683	Average Score: 0.19
epsilon: 0.1
Episode 4684	Average Score: 0.19
epsilon: 0.1
Episode 4685	Average Score: 0.19
epsilon: 0.1
Episode 4686	Average Score: 0.20
epsilon: 0.1
Episode 4687	Average Score: 0.20
epsilon: 0.1
Episode 4688	Average Score: 0.20
epsilon: 0.1
Episode 4689	Average Score: 0.20
epsilon: 0.1
Episode 4690	Average Score: 0.20
epsilon: 0.1
Episode 4691	Average Score: 0.20
epsilon: 0.1
Episode 4692	Average Score: 0.20
epsilon: 0.1
Episode 4693	Average Score: 0.20
epsilon: 0.1
Episode 4694	Average Score: 0.20
epsilon: 0.1
Episode 4695	Average Score: 0.20
epsilon: 0.1
Episode 4696	Average Score: 0.20
epsilon: 0.1
Episode 4697	Average Score: 0.20
epsilon: 0.1
Episode 4698	Average Score: 0.20
epsilon: 0.1
Episode 4699	Average Score: 0.20
epsilon: 0.1
Episode 4700	Average Score: 0.21
epsilon: 0.1
Episode 4701	Average Score: 0.20
epsilon: 0.1
Episode 4702	Average Score: 0.21
epsilon: 0.1
Episode 4703	Average Score: 0.20
epsilon: 0.1
Episode 4704	Average Score: 0.20
epsilon: 0.1
Episode 4705	Average Score: 0.20
epsilon: 0.1
Episode 4706	Average Score: 0.20
epsilon: 0.1
Episode 4707	Average Score: 0.20
epsilon: 0.1
Episode 4708	Average Score: 0.20
epsilon: 0.1
Episode 4709	Average Score: 0.20
epsilon: 0.1
Episode 4710	Average Score: 0.20
epsilon: 0.1
Episode 4711	Average Score: 0.20
epsilon: 0.1
Episode 4712	Average Score: 0.20
epsilon: 0.1
Episode 4713	Average Score: 0.20
epsilon: 0.1
Episode 4714	Average Score: 0.20
epsilon: 0.1
Episode 4715	Average Score: 0.21
epsilon: 0.1
Episode 4716	Average Score: 0.21
epsilon: 0.1
Episode 4717	Average Score: 0.21
epsilon: 0.1
Episode 4718	Average Score: 0.21
epsilon: 0.1
Episode 4719	Average Score: 0.21
epsilon: 0.1
Episode 4720	Average Score: 0.21
epsilon: 0.1
Episode 4721	Average Score: 0.21
epsilon: 0.1
Episode 4722	Average Score: 0.22
epsilon: 0.1
Episode 4723	Average Score: 0.22
epsilon: 0.1
Episode 4724	Average Score: 0.22
epsilon: 0.1
Episode 4725	Average Score: 0.22
epsilon: 0.1
Episode 4726	Average Score: 0.21
epsilon: 0.1
Episode 4727	Average Score: 0.21
epsilon: 0.1
Episode 4728	Average Score: 0.21
epsilon: 0.1
Episode 4729	Average Score: 0.21
epsilon: 0.1
Episode 4730	Average Score: 0.21
epsilon: 0.1
Episode 4731	Average Score: 0.21
epsilon: 0.1
Episode 4732	Average Score: 0.21
epsilon: 0.1
Episode 4733	Average Score: 0.21
epsilon: 0.1
Episode 4734	Average Score: 0.21
epsilon: 0.1
Episode 4735	Average Score: 0.21
epsilon: 0.1
Episode 4736	Average Score: 0.21
epsilon: 0.1
Episode 4737	Average Score: 0.21
epsilon: 0.1
Episode 4738	Average Score: 0.21
epsilon: 0.1
Episode 4739	Average Score: 0.21
epsilon: 0.1
Episode 4740	Average Score: 0.21
epsilon: 0.1
Episode 4741	Average Score: 0.21
epsilon: 0.1
Episode 4742	Average Score: 0.21
epsilon: 0.1
Episode 4743	Average Score: 0.22
epsilon: 0.1
Episode 4744	Average Score: 0.22
epsilon: 0.1
Episode 4745	Average Score: 0.22
epsilon: 0.1
Episode 4746	Average Score: 0.22
epsilon: 0.1
Episode 4747	Average Score: 0.22
epsilon: 0.1
Episode 4748	Average Score: 0.22
epsilon: 0.1
Episode 4749	Average Score: 0.22
epsilon: 0.1
Episode 4750	Average Score: 0.22
epsilon: 0.1
Episode 4751	Average Score: 0.22
epsilon: 0.1
Episode 4752	Average Score: 0.21
epsilon: 0.1
Episode 4753	Average Score: 0.21
epsilon: 0.1
Episode 4754	Average Score: 0.21
epsilon: 0.1
Episode 4755	Average Score: 0.21
epsilon: 0.1
Episode 4756	Average Score: 0.21
epsilon: 0.1
Episode 4757	Average Score: 0.21
epsilon: 0.1
Episode 4758	Average Score: 0.21
epsilon: 0.1
Episode 4759	Average Score: 0.21
epsilon: 0.1
Episode 4760	Average Score: 0.22
epsilon: 0.1
Episode 4761	Average Score: 0.22
epsilon: 0.1
Episode 4762	Average Score: 0.22
epsilon: 0.1
Episode 4763	Average Score: 0.22
epsilon: 0.1
Episode 4764	Average Score: 0.22
epsilon: 0.1
Episode 4765	Average Score: 0.22
epsilon: 0.1
Episode 4766	Average Score: 0.21
epsilon: 0.1
Episode 4767	Average Score: 0.21
epsilon: 0.1
Episode 4768	Average Score: 0.22
epsilon: 0.1
Episode 4769	Average Score: 0.22
epsilon: 0.1
Episode 4770	Average Score: 0.21
epsilon: 0.1
Episode 4771	Average Score: 0.21
epsilon: 0.1
Episode 4772	Average Score: 0.21
epsilon: 0.1
Episode 4773	Average Score: 0.22
epsilon: 0.1
Episode 4774	Average Score: 0.21
epsilon: 0.1
Episode 4775	Average Score: 0.21
epsilon: 0.1
Episode 4776	Average Score: 0.21
epsilon: 0.1
Episode 4777	Average Score: 0.21
epsilon: 0.1
Episode 4778	Average Score: 0.21
epsilon: 0.1
Episode 4779	Average Score: 0.21
epsilon: 0.1
Episode 4780	Average Score: 0.21
epsilon: 0.1
Episode 4781	Average Score: 0.21
epsilon: 0.1
Episode 4782	Average Score: 0.21
epsilon: 0.1
Episode 4783	Average Score: 0.21
epsilon: 0.1
Episode 4784	Average Score: 0.21
epsilon: 0.1
Episode 4785	Average Score: 0.21
epsilon: 0.1
Episode 4786	Average Score: 0.21
epsilon: 0.1
Episode 4787	Average Score: 0.21
epsilon: 0.1
Episode 4788	Average Score: 0.21
epsilon: 0.1
Episode 4789	Average Score: 0.21
epsilon: 0.1
Episode 4790	Average Score: 0.21
epsilon: 0.1
Episode 4791	Average Score: 0.21
epsilon: 0.1
Episode 4792	Average Score: 0.21
epsilon: 0.1
Episode 4793	Average Score: 0.20
epsilon: 0.1
Episode 4794	Average Score: 0.20
epsilon: 0.1
Episode 4795	Average Score: 0.20
epsilon: 0.1
Episode 4796	Average Score: 0.20
epsilon: 0.1
Episode 4797	Average Score: 0.21
epsilon: 0.1
Episode 4798	Average Score: 0.21
epsilon: 0.1
Episode 4799	Average Score: 0.21
epsilon: 0.1
Episode 4800	Average Score: 0.21
epsilon: 0.1
Episode 4801	Average Score: 0.21
epsilon: 0.1
Episode 4802	Average Score: 0.21
epsilon: 0.1
Episode 4803	Average Score: 0.21
epsilon: 0.1
Episode 4804	Average Score: 0.21
epsilon: 0.1
Episode 4805	Average Score: 0.21
epsilon: 0.1
Episode 4806	Average Score: 0.20
epsilon: 0.1
Episode 4807	Average Score: 0.21
epsilon: 0.1
Episode 4808	Average Score: 0.20
epsilon: 0.1
Episode 4809	Average Score: 0.21
epsilon: 0.1
Episode 4810	Average Score: 0.21
epsilon: 0.1
Episode 4811	Average Score: 0.21
epsilon: 0.1
Episode 4812	Average Score: 0.22
epsilon: 0.1
Episode 4813	Average Score: 0.22
epsilon: 0.1
Episode 4814	Average Score: 0.22
epsilon: 0.1
Episode 4815	Average Score: 0.21
epsilon: 0.1
Episode 4816	Average Score: 0.21
epsilon: 0.1
Episode 4817	Average Score: 0.21
epsilon: 0.1
Episode 4818	Average Score: 0.20
epsilon: 0.1
Episode 4819	Average Score: 0.21
epsilon: 0.1
Episode 4820	Average Score: 0.21
epsilon: 0.1
Episode 4821	Average Score: 0.20
epsilon: 0.1
Episode 4822	Average Score: 0.20
epsilon: 0.1
Episode 4823	Average Score: 0.19
epsilon: 0.1
Episode 4824	Average Score: 0.19
epsilon: 0.1
Episode 4825	Average Score: 0.19
epsilon: 0.1
Episode 4826	Average Score: 0.19
epsilon: 0.1
Episode 4827	Average Score: 0.20
epsilon: 0.1
Episode 4828	Average Score: 0.20
epsilon: 0.1
Episode 4829	Average Score: 0.20
epsilon: 0.1
Episode 4830	Average Score: 0.19
epsilon: 0.1
Episode 4831	Average Score: 0.19
epsilon: 0.1
Episode 4832	Average Score: 0.20
epsilon: 0.1
Episode 4833	Average Score: 0.21
epsilon: 0.1
Episode 4834	Average Score: 0.21
epsilon: 0.1
Episode 4835	Average Score: 0.22
epsilon: 0.1
Episode 4836	Average Score: 0.21
epsilon: 0.1
Episode 4837	Average Score: 0.21
epsilon: 0.1
Episode 4838	Average Score: 0.21
epsilon: 0.1
Episode 4839	Average Score: 0.21
epsilon: 0.1
Episode 4840	Average Score: 0.21
epsilon: 0.1
Episode 4841	Average Score: 0.21
epsilon: 0.1
Episode 4842	Average Score: 0.21
epsilon: 0.1
Episode 4843	Average Score: 0.20
epsilon: 0.1
Episode 4844	Average Score: 0.20
epsilon: 0.1
Episode 4845	Average Score: 0.20
epsilon: 0.1
Episode 4846	Average Score: 0.20
epsilon: 0.1
Episode 4847	Average Score: 0.20
epsilon: 0.1
Episode 4848	Average Score: 0.20
epsilon: 0.1
Episode 4849	Average Score: 0.20
epsilon: 0.1
Episode 4850	Average Score: 0.20
epsilon: 0.1
Episode 4851	Average Score: 0.20
epsilon: 0.1
Episode 4852	Average Score: 0.21
epsilon: 0.1
Episode 4853	Average Score: 0.21
epsilon: 0.1
Episode 4854	Average Score: 0.21
epsilon: 0.1
Episode 4855	Average Score: 0.21
epsilon: 0.1
Episode 4856	Average Score: 0.22
epsilon: 0.1
Episode 4857	Average Score: 0.22
epsilon: 0.1
Episode 4858	Average Score: 0.22
epsilon: 0.1
Episode 4859	Average Score: 0.23
epsilon: 0.1
Episode 4860	Average Score: 0.22
epsilon: 0.1
Episode 4861	Average Score: 0.23
epsilon: 0.1
Episode 4862	Average Score: 0.23
epsilon: 0.1
Episode 4863	Average Score: 0.23
epsilon: 0.1
Episode 4864	Average Score: 0.24
epsilon: 0.1
Episode 4865	Average Score: 0.24
epsilon: 0.1
Episode 4866	Average Score: 0.25
epsilon: 0.1
Episode 4867	Average Score: 0.25
epsilon: 0.1
Episode 4868	Average Score: 0.25
epsilon: 0.1
Episode 4869	Average Score: 0.25
epsilon: 0.1
Episode 4870	Average Score: 0.25
epsilon: 0.1
Episode 4871	Average Score: 0.25
epsilon: 0.1
Episode 4872	Average Score: 0.25
epsilon: 0.1
Episode 4873	Average Score: 0.24
epsilon: 0.1
Episode 4874	Average Score: 0.24
epsilon: 0.1
Episode 4875	Average Score: 0.24
epsilon: 0.1
Episode 4876	Average Score: 0.24
epsilon: 0.1
Episode 4877	Average Score: 0.25
epsilon: 0.1
Episode 4878	Average Score: 0.25
epsilon: 0.1
Episode 4879	Average Score: 0.24
epsilon: 0.1
Episode 4880	Average Score: 0.25
epsilon: 0.1
Episode 4881	Average Score: 0.25
epsilon: 0.1
Episode 4882	Average Score: 0.24
epsilon: 0.1
Episode 4883	Average Score: 0.24
epsilon: 0.1
Episode 4884	Average Score: 0.24
epsilon: 0.1
Episode 4885	Average Score: 0.24
epsilon: 0.1
Episode 4886	Average Score: 0.24
epsilon: 0.1
Episode 4887	Average Score: 0.24
epsilon: 0.1
Episode 4888	Average Score: 0.25
epsilon: 0.1
Episode 4889	Average Score: 0.25
epsilon: 0.1
Episode 4890	Average Score: 0.25
epsilon: 0.1
Episode 4891	Average Score: 0.25
epsilon: 0.1
Episode 4892	Average Score: 0.24
epsilon: 0.1
Episode 4893	Average Score: 0.25
epsilon: 0.1
Episode 4894	Average Score: 0.25
epsilon: 0.1
Episode 4895	Average Score: 0.25
epsilon: 0.1
Episode 4896	Average Score: 0.25
epsilon: 0.1
Episode 4897	Average Score: 0.25
epsilon: 0.1
Episode 4898	Average Score: 0.25
epsilon: 0.1
Episode 4899	Average Score: 0.24
epsilon: 0.1
Episode 4900	Average Score: 0.25
epsilon: 0.1
Episode 4901	Average Score: 0.25
epsilon: 0.1
Episode 4902	Average Score: 0.25
epsilon: 0.1
Episode 4903	Average Score: 0.25
epsilon: 0.1
Episode 4904	Average Score: 0.25
epsilon: 0.1
Episode 4905	Average Score: 0.25
epsilon: 0.1
Episode 4906	Average Score: 0.25
epsilon: 0.1
Episode 4907	Average Score: 0.25
epsilon: 0.1
Episode 4908	Average Score: 0.25
epsilon: 0.1
Episode 4909	Average Score: 0.25
epsilon: 0.1
Episode 4910	Average Score: 0.25
epsilon: 0.1
Episode 4911	Average Score: 0.24
epsilon: 0.1
Episode 4912	Average Score: 0.24
epsilon: 0.1
Episode 4913	Average Score: 0.24
epsilon: 0.1
Episode 4914	Average Score: 0.24
epsilon: 0.1
Episode 4915	Average Score: 0.24
epsilon: 0.1
Episode 4916	Average Score: 0.24
epsilon: 0.1
Episode 4917	Average Score: 0.24
epsilon: 0.1
Episode 4918	Average Score: 0.24
epsilon: 0.1
Episode 4919	Average Score: 0.24
epsilon: 0.1
Episode 4920	Average Score: 0.24
epsilon: 0.1
Episode 4921	Average Score: 0.23
epsilon: 0.1
Episode 4922	Average Score: 0.23
epsilon: 0.1
Episode 4923	Average Score: 0.24
epsilon: 0.1
Episode 4924	Average Score: 0.24
epsilon: 0.1
Episode 4925	Average Score: 0.24
epsilon: 0.1
Episode 4926	Average Score: 0.24
epsilon: 0.1
Episode 4927	Average Score: 0.24
epsilon: 0.1
Episode 4928	Average Score: 0.24
epsilon: 0.1
Episode 4929	Average Score: 0.24
epsilon: 0.1
Episode 4930	Average Score: 0.24
epsilon: 0.1
Episode 4931	Average Score: 0.24
epsilon: 0.1
Episode 4932	Average Score: 0.23
epsilon: 0.1
Episode 4933	Average Score: 0.24
epsilon: 0.1
Episode 4934	Average Score: 0.24
epsilon: 0.1
Episode 4935	Average Score: 0.24
epsilon: 0.1
Episode 4936	Average Score: 0.24
epsilon: 0.1
Episode 4937	Average Score: 0.24
epsilon: 0.1
Episode 4938	Average Score: 0.24
epsilon: 0.1
Episode 4939	Average Score: 0.25
epsilon: 0.1
Episode 4940	Average Score: 0.25
epsilon: 0.1
Episode 4941	Average Score: 0.25
epsilon: 0.1
Episode 4942	Average Score: 0.25
epsilon: 0.1
Episode 4943	Average Score: 0.25
epsilon: 0.1
Episode 4944	Average Score: 0.25
epsilon: 0.1
Episode 4945	Average Score: 0.26
epsilon: 0.1
Episode 4946	Average Score: 0.26
epsilon: 0.1
Episode 4947	Average Score: 0.26
epsilon: 0.1
Episode 4948	Average Score: 0.26
epsilon: 0.1
Episode 4949	Average Score: 0.26
epsilon: 0.1
Episode 4950	Average Score: 0.27
epsilon: 0.1
Episode 4951	Average Score: 0.27
epsilon: 0.1
Episode 4952	Average Score: 0.26
epsilon: 0.1
Episode 4953	Average Score: 0.26
epsilon: 0.1
Episode 4954	Average Score: 0.25
epsilon: 0.1
Episode 4955	Average Score: 0.26
epsilon: 0.1
Episode 4956	Average Score: 0.25
epsilon: 0.1
Episode 4957	Average Score: 0.25
epsilon: 0.1
Episode 4958	Average Score: 0.25
epsilon: 0.1
Episode 4959	Average Score: 0.25
epsilon: 0.1
Episode 4960	Average Score: 0.25
epsilon: 0.1
Episode 4961	Average Score: 0.25
epsilon: 0.1
Episode 4962	Average Score: 0.25
epsilon: 0.1
Episode 4963	Average Score: 0.24
epsilon: 0.1
Episode 4964	Average Score: 0.24
epsilon: 0.1
Episode 4965	Average Score: 0.24
epsilon: 0.1
Episode 4966	Average Score: 0.23
epsilon: 0.1
Episode 4967	Average Score: 0.23
epsilon: 0.1
Episode 4968	Average Score: 0.23
epsilon: 0.1
Episode 4969	Average Score: 0.23
epsilon: 0.1
Episode 4970	Average Score: 0.23
epsilon: 0.1
Episode 4971	Average Score: 0.23
epsilon: 0.1
Episode 4972	Average Score: 0.23
epsilon: 0.1
Episode 4973	Average Score: 0.23
epsilon: 0.1
Episode 4974	Average Score: 0.23
epsilon: 0.1
Episode 4975	Average Score: 0.23
epsilon: 0.1
Episode 4976	Average Score: 0.23
epsilon: 0.1
Episode 4977	Average Score: 0.23
epsilon: 0.1
Episode 4978	Average Score: 0.23
epsilon: 0.1
Episode 4979	Average Score: 0.23
epsilon: 0.1
Episode 4980	Average Score: 0.23
epsilon: 0.1
Episode 4981	Average Score: 0.22
epsilon: 0.1
Episode 4982	Average Score: 0.23
epsilon: 0.1
Episode 4983	Average Score: 0.23
epsilon: 0.1
Episode 4984	Average Score: 0.23
epsilon: 0.1
Episode 4985	Average Score: 0.23
epsilon: 0.1
Episode 4986	Average Score: 0.23
epsilon: 0.1
Episode 4987	Average Score: 0.23
epsilon: 0.1
Episode 4988	Average Score: 0.23
epsilon: 0.1
Episode 4989	Average Score: 0.23
epsilon: 0.1
Episode 4990	Average Score: 0.23
epsilon: 0.1
Episode 4991	Average Score: 0.23
epsilon: 0.1
Episode 4992	Average Score: 0.23
epsilon: 0.1
Episode 4993	Average Score: 0.23
epsilon: 0.1
Episode 4994	Average Score: 0.23
epsilon: 0.1
Episode 4995	Average Score: 0.23
epsilon: 0.1
Episode 4996	Average Score: 0.23
epsilon: 0.1
Episode 4997	Average Score: 0.23
epsilon: 0.1
Episode 4998	Average Score: 0.23
epsilon: 0.1
Episode 4999	Average Score: 0.24
epsilon: 0.1
Episode 5000	Average Score: 0.23
epsilon: 0.1
Episode 5001	Average Score: 0.23
epsilon: 0.1
Episode 5002	Average Score: 0.24
epsilon: 0.1
Episode 5003	Average Score: 0.24
epsilon: 0.1
Episode 5004	Average Score: 0.24
epsilon: 0.1
Episode 5005	Average Score: 0.25
epsilon: 0.1
Episode 5006	Average Score: 0.25
epsilon: 0.1
Episode 5007	Average Score: 0.25
epsilon: 0.1
Episode 5008	Average Score: 0.25
epsilon: 0.1
Episode 5009	Average Score: 0.25
epsilon: 0.1
Episode 5010	Average Score: 0.25
epsilon: 0.1
Episode 5011	Average Score: 0.25
epsilon: 0.1
Episode 5012	Average Score: 0.25
epsilon: 0.1
Episode 5013	Average Score: 0.25
epsilon: 0.1
Episode 5014	Average Score: 0.25
epsilon: 0.1
Episode 5015	Average Score: 0.25
epsilon: 0.1
Episode 5016	Average Score: 0.25
epsilon: 0.1
Episode 5017	Average Score: 0.25
epsilon: 0.1
Episode 5018	Average Score: 0.25
epsilon: 0.1
Episode 5019	Average Score: 0.25
epsilon: 0.1
Episode 5020	Average Score: 0.25
epsilon: 0.1
Episode 5021	Average Score: 0.26
epsilon: 0.1
Episode 5022	Average Score: 0.26
epsilon: 0.1
Episode 5023	Average Score: 0.26
epsilon: 0.1
Episode 5024	Average Score: 0.26
epsilon: 0.1
Episode 5025	Average Score: 0.26
epsilon: 0.1
Episode 5026	Average Score: 0.26
epsilon: 0.1
Episode 5027	Average Score: 0.26
epsilon: 0.1
Episode 5028	Average Score: 0.26
epsilon: 0.1
Episode 5029	Average Score: 0.26
epsilon: 0.1
Episode 5030	Average Score: 0.27
epsilon: 0.1
Episode 5031	Average Score: 0.27
epsilon: 0.1
Episode 5032	Average Score: 0.27
epsilon: 0.1
Episode 5033	Average Score: 0.25
epsilon: 0.1
Episode 5034	Average Score: 0.25
epsilon: 0.1
Episode 5035	Average Score: 0.25
epsilon: 0.1
Episode 5036	Average Score: 0.26
epsilon: 0.1
Episode 5037	Average Score: 0.27
epsilon: 0.1
Episode 5038	Average Score: 0.27
epsilon: 0.1
Episode 5039	Average Score: 0.25
epsilon: 0.1
Episode 5040	Average Score: 0.25
epsilon: 0.1
Episode 5041	Average Score: 0.25
epsilon: 0.1
Episode 5042	Average Score: 0.25
epsilon: 0.1
Episode 5043	Average Score: 0.25
epsilon: 0.1
Episode 5044	Average Score: 0.25
epsilon: 0.1
Episode 5045	Average Score: 0.25
epsilon: 0.1
Episode 5046	Average Score: 0.25
epsilon: 0.1
Episode 5047	Average Score: 0.25
epsilon: 0.1
Episode 5048	Average Score: 0.25
epsilon: 0.1
Episode 5049	Average Score: 0.25
epsilon: 0.1
Episode 5050	Average Score: 0.25
epsilon: 0.1
Episode 5051	Average Score: 0.24
epsilon: 0.1
Episode 5052	Average Score: 0.24
epsilon: 0.1
Episode 5053	Average Score: 0.24
epsilon: 0.1
Episode 5054	Average Score: 0.24
epsilon: 0.1
Episode 5055	Average Score: 0.24
epsilon: 0.1
Episode 5056	Average Score: 0.24
epsilon: 0.1
Episode 5057	Average Score: 0.24
epsilon: 0.1
Episode 5058	Average Score: 0.24
epsilon: 0.1
Episode 5059	Average Score: 0.23
epsilon: 0.1
Episode 5060	Average Score: 0.23
epsilon: 0.1
Episode 5061	Average Score: 0.23
epsilon: 0.1
Episode 5062	Average Score: 0.23
epsilon: 0.1
Episode 5063	Average Score: 0.24
epsilon: 0.1
Episode 5064	Average Score: 0.24
epsilon: 0.1
Episode 5065	Average Score: 0.24
epsilon: 0.1
Episode 5066	Average Score: 0.23
epsilon: 0.1
Episode 5067	Average Score: 0.24
epsilon: 0.1
Episode 5068	Average Score: 0.24
epsilon: 0.1
Episode 5069	Average Score: 0.24
epsilon: 0.1
Episode 5070	Average Score: 0.24
epsilon: 0.1
Episode 5071	Average Score: 0.24
epsilon: 0.1
Episode 5072	Average Score: 0.24
epsilon: 0.1
Episode 5073	Average Score: 0.24
epsilon: 0.1
Episode 5074	Average Score: 0.24
epsilon: 0.1
Episode 5075	Average Score: 0.25
epsilon: 0.1
Episode 5076	Average Score: 0.25
epsilon: 0.1
Episode 5077	Average Score: 0.25
epsilon: 0.1
Episode 5078	Average Score: 0.24
epsilon: 0.1
Episode 5079	Average Score: 0.24
epsilon: 0.1
Episode 5080	Average Score: 0.24
epsilon: 0.1
Episode 5081	Average Score: 0.24
epsilon: 0.1
Episode 5082	Average Score: 0.24
epsilon: 0.1
Episode 5083	Average Score: 0.24
epsilon: 0.1
Episode 5084	Average Score: 0.24
epsilon: 0.1
Episode 5085	Average Score: 0.24
epsilon: 0.1
Episode 5086	Average Score: 0.24
epsilon: 0.1
Episode 5087	Average Score: 0.24
epsilon: 0.1
Episode 5088	Average Score: 0.24
epsilon: 0.1
Episode 5089	Average Score: 0.24
epsilon: 0.1
Episode 5090	Average Score: 0.24
epsilon: 0.1
Episode 5091	Average Score: 0.24
epsilon: 0.1
Episode 5092	Average Score: 0.24
epsilon: 0.1
Episode 5093	Average Score: 0.24
epsilon: 0.1
Episode 5094	Average Score: 0.24
epsilon: 0.1
Episode 5095	Average Score: 0.24
epsilon: 0.1
Episode 5096	Average Score: 0.23
epsilon: 0.1
Episode 5097	Average Score: 0.23
epsilon: 0.1
Episode 5098	Average Score: 0.23
epsilon: 0.1
Episode 5099	Average Score: 0.23
epsilon: 0.1
Episode 5100	Average Score: 0.23
epsilon: 0.1
Episode 5101	Average Score: 0.22
epsilon: 0.1
Episode 5102	Average Score: 0.22
epsilon: 0.1
Episode 5103	Average Score: 0.22
epsilon: 0.1
Episode 5104	Average Score: 0.22
epsilon: 0.1
Episode 5105	Average Score: 0.21
epsilon: 0.1
Episode 5106	Average Score: 0.21
epsilon: 0.1
Episode 5107	Average Score: 0.21
epsilon: 0.1
Episode 5108	Average Score: 0.21
epsilon: 0.1
Episode 5109	Average Score: 0.21
epsilon: 0.1
Episode 5110	Average Score: 0.21
epsilon: 0.1
Episode 5111	Average Score: 0.21
epsilon: 0.1
Episode 5112	Average Score: 0.21
epsilon: 0.1
Episode 5113	Average Score: 0.21
epsilon: 0.1
Episode 5114	Average Score: 0.21
epsilon: 0.1
Episode 5115	Average Score: 0.21
epsilon: 0.1
Episode 5116	Average Score: 0.21
epsilon: 0.1
Episode 5117	Average Score: 0.21
epsilon: 0.1
Episode 5118	Average Score: 0.21
epsilon: 0.1
Episode 5119	Average Score: 0.21
epsilon: 0.1
Episode 5120	Average Score: 0.21
epsilon: 0.1
Episode 5121	Average Score: 0.21
epsilon: 0.1
Episode 5122	Average Score: 0.21
epsilon: 0.1
Episode 5123	Average Score: 0.20
epsilon: 0.1
Episode 5124	Average Score: 0.21
epsilon: 0.1
Episode 5125	Average Score: 0.21
epsilon: 0.1
Episode 5126	Average Score: 0.20
epsilon: 0.1
Episode 5127	Average Score: 0.20
epsilon: 0.1
Episode 5128	Average Score: 0.20
epsilon: 0.1
Episode 5129	Average Score: 0.20
epsilon: 0.1
Episode 5130	Average Score: 0.20
epsilon: 0.1
Episode 5131	Average Score: 0.20
epsilon: 0.1
Episode 5132	Average Score: 0.20
epsilon: 0.1
Episode 5133	Average Score: 0.20
epsilon: 0.1
Episode 5134	Average Score: 0.21
epsilon: 0.1
Episode 5135	Average Score: 0.21
epsilon: 0.1
Episode 5136	Average Score: 0.21
epsilon: 0.1
Episode 5137	Average Score: 0.20
epsilon: 0.1
Episode 5138	Average Score: 0.20
epsilon: 0.1
Episode 5139	Average Score: 0.20
epsilon: 0.1
Episode 5140	Average Score: 0.22
epsilon: 0.1
Episode 5141	Average Score: 0.22
epsilon: 0.1
Episode 5142	Average Score: 0.22
epsilon: 0.1
Episode 5143	Average Score: 0.22
epsilon: 0.1
Episode 5144	Average Score: 0.23
epsilon: 0.1
Episode 5145	Average Score: 0.23
epsilon: 0.1
Episode 5146	Average Score: 0.24
epsilon: 0.1
Episode 5147	Average Score: 0.24
epsilon: 0.1
Episode 5148	Average Score: 0.24
epsilon: 0.1
Episode 5149	Average Score: 0.24
epsilon: 0.1
Episode 5150	Average Score: 0.24
epsilon: 0.1
Episode 5151	Average Score: 0.24
epsilon: 0.1
Episode 5152	Average Score: 0.25
epsilon: 0.1
Episode 5153	Average Score: 0.24
epsilon: 0.1
Episode 5154	Average Score: 0.24
epsilon: 0.1
Episode 5155	Average Score: 0.25
epsilon: 0.1
Episode 5156	Average Score: 0.25
epsilon: 0.1
Episode 5157	Average Score: 0.25
epsilon: 0.1
Episode 5158	Average Score: 0.25
epsilon: 0.1
Episode 5159	Average Score: 0.25
epsilon: 0.1
Episode 5160	Average Score: 0.26
epsilon: 0.1
Episode 5161	Average Score: 0.27
epsilon: 0.1
Episode 5162	Average Score: 0.27
epsilon: 0.1
Episode 5163	Average Score: 0.27
epsilon: 0.1
Episode 5164	Average Score: 0.27
epsilon: 0.1
Episode 5165	Average Score: 0.27
epsilon: 0.1
Episode 5166	Average Score: 0.27
epsilon: 0.1
Episode 5167	Average Score: 0.27
epsilon: 0.1
Episode 5168	Average Score: 0.27
epsilon: 0.1
Episode 5169	Average Score: 0.27
epsilon: 0.1
Episode 5170	Average Score: 0.27
epsilon: 0.1
Episode 5171	Average Score: 0.27
epsilon: 0.1
Episode 5172	Average Score: 0.27
epsilon: 0.1
Episode 5173	Average Score: 0.27
epsilon: 0.1
Episode 5174	Average Score: 0.27
epsilon: 0.1
Episode 5175	Average Score: 0.27
epsilon: 0.1
Episode 5176	Average Score: 0.28
epsilon: 0.1
Episode 5177	Average Score: 0.28
epsilon: 0.1
Episode 5178	Average Score: 0.28
epsilon: 0.1
Episode 5179	Average Score: 0.28
epsilon: 0.1
Episode 5180	Average Score: 0.29
epsilon: 0.1
Episode 5181	Average Score: 0.30
epsilon: 0.1
Episode 5182	Average Score: 0.30
epsilon: 0.1
Episode 5183	Average Score: 0.30
epsilon: 0.1
Episode 5184	Average Score: 0.30
epsilon: 0.1
Episode 5185	Average Score: 0.30
epsilon: 0.1
Episode 5186	Average Score: 0.31
epsilon: 0.1
Episode 5187	Average Score: 0.31
epsilon: 0.1
Episode 5188	Average Score: 0.31
epsilon: 0.1
Episode 5189	Average Score: 0.31
epsilon: 0.1
Episode 5190	Average Score: 0.31
epsilon: 0.1
Episode 5191	Average Score: 0.31
epsilon: 0.1
Episode 5192	Average Score: 0.33
epsilon: 0.1
Episode 5193	Average Score: 0.33
epsilon: 0.1
Episode 5194	Average Score: 0.33
epsilon: 0.1
Episode 5195	Average Score: 0.33
epsilon: 0.1
Episode 5196	Average Score: 0.34
epsilon: 0.1
Episode 5197	Average Score: 0.35
epsilon: 0.1
Episode 5198	Average Score: 0.34
epsilon: 0.1
Episode 5199	Average Score: 0.36
epsilon: 0.1
Episode 5200	Average Score: 0.36
epsilon: 0.1
Episode 5201	Average Score: 0.37
epsilon: 0.1
Episode 5202	Average Score: 0.37
epsilon: 0.1
Episode 5203	Average Score: 0.40
epsilon: 0.1
Episode 5204	Average Score: 0.40
epsilon: 0.1
Episode 5205	Average Score: 0.41
epsilon: 0.1
Episode 5206	Average Score: 0.43
epsilon: 0.1
Episode 5207	Average Score: 0.43
epsilon: 0.1
Episode 5208	Average Score: 0.44
epsilon: 0.1
Episode 5209	Average Score: 0.44
epsilon: 0.1
Episode 5210	Average Score: 0.44
epsilon: 0.1
Episode 5211	Average Score: 0.44
epsilon: 0.1
Episode 5212	Average Score: 0.45
epsilon: 0.1
Episode 5213	Average Score: 0.45
epsilon: 0.1
Episode 5214	Average Score: 0.45
epsilon: 0.1
Episode 5215	Average Score: 0.45
epsilon: 0.1
Episode 5216	Average Score: 0.45
epsilon: 0.1
Episode 5217	Average Score: 0.45
epsilon: 0.1
Episode 5218	Average Score: 0.47
epsilon: 0.1
Episode 5219	Average Score: 0.47
epsilon: 0.1
Episode 5220	Average Score: 0.47
epsilon: 0.1
Episode 5221	Average Score: 0.48
epsilon: 0.1
Episode 5222	Average Score: 0.48
epsilon: 0.1
Episode 5223	Average Score: 0.49
epsilon: 0.1
Episode 5224	Average Score: 0.49
epsilon: 0.1
Episode 5225	Average Score: 0.49
epsilon: 0.1
Episode 5226	Average Score: 0.49
epsilon: 0.1
Episode 5227	Average Score: 0.50
epsilon: 0.1
Episode 5228	Average Score: 0.50
epsilon: 0.1
Episode 5229	Average Score: 0.50
epsilon: 0.1
Episode 5230	Average Score: 0.51
epsilon: 0.1

Environment solved in 5130 episodes!	Average Score: 0.51
```

### Ideas for Future Work

Future ideas to improve the agent are to incorporate:

#### Prioritize Experience Replay

In MADDPG,the agent interacts with the environment,and a transition (s,a,r,s'), is inserted to an experience replay. To update the parameters of the neural network, a sample of transitions is drawn uniformly at random from the experience replay buffer. Some transitions might be more important to the learning process than others, however,uniform sampling policy treat transitions by the amount of their occurrences. Prioritized experience replay, aims to deal with this issue by assigning each transition a priority, P ,which is proportional to the TD-error. 

#### Delayed Policy Updates

In order to allow for better state-value estimation and have a more stable actor, we can update the policy less frequently compared to the critic.

### Learned Policy

The solution can be found in files checkpoint_actor_0.pth, checkpoint_actor_1.pth, checkpoint_critic_0.pth, checkpoint_critic_1.pth .