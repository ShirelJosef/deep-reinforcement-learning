from unityagents import UnityEnvironment
import numpy as np

env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

print('The state space shape: '+str(states.shape))
print('The state space looks like: '+str(states))
actions = np.random.randn(num_agents, action_size)
print('The action space shape: '+str(actions.shape))
print('The action space looks like: '+str(actions))


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

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


import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed, number_of_agents):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size * number_of_agents, action_size * number_of_agents, random_seed).to(
            device)
        self.critic_target = Critic(state_size * number_of_agents, action_size * number_of_agents, random_seed).to(
            device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)  # , weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        #self.noise = OrnsteinUhlenbeckProcess(action_size)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # number of agents
        self.number_of_agents = number_of_agents

        self.current_eps = START_EPS

    def step(self, state, action, reward, next_state, done, agent_number, agents):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        assert len(state) == (self.number_of_agents)
        # Save experience / reward
        # for i in range(len(state)):
        #    self.memory.add(state[i], action[i], reward[i], next_state[i], done)
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA, agent_number, agents)

    def act(self, state, step, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            self.current_eps = max(END_EPS, self.current_eps - (START_EPS - END_EPS) / 100000)#10000)
            # print(self.current_eps)
            action += (self.noise.sample() * self.current_eps)
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, agent_number, agents):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        minibatch_size = states.shape[0]
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = torch.zeros((minibatch_size, num_agents, action_size)).float().to(device)
        for i, agent in enumerate(agents):
            actions_next[:,i] = agent.actor_target(next_states[:,i])
        #actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states.view(minibatch_size,-1), actions_next.view(minibatch_size,-1))
        # Compute Q targets for current states (y_i)
        Q_targets = rewards[:,agent_number].unsqueeze(-1) + (gamma * Q_targets_next * (1 - dones[:,agent_number].unsqueeze(-1)))
        # Compute critic loss
        Q_expected = self.critic_local(states.view(minibatch_size,-1), actions.view(minibatch_size,-1))
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        #actions_pred = self.actor_local(states[:,agent_number])
        actions_pred = torch.zeros((minibatch_size, num_agents, action_size)).float().to(device)
        for i, agent in enumerate(agents):
            if i != agent_number:
                actions_pred[:, i] = agent.actor_local(next_states[:, i])
            else:
                actions_pred[:, i] = self.actor_local(next_states[:, i])
        actor_loss = -self.critic_local(states.view(minibatch_size,-1), actions_pred.view(minibatch_size,-1)).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.17, sigma=0.4):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        #self.seed = random.seed(seed)
        self.reset()
        print("OUNoise params - Mu:{} , theta:{} sigma:{} ".format(self.mu,self.theta,self.sigma))
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        #Standard normal distribution for the action size
        dx = self.theta * (self.mu - x) + self.sigma *  np.random.randn(self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([np.expand_dims(e.state,0) for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([np.expand_dims(e.action,0) for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([np.expand_dims(e.reward,0) for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([np.expand_dims(e.next_state,0) for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([np.expand_dims(e.done,0) for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# trianing loop

import matplotlib.pyplot as plt

agents = [Agent(state_size=24, action_size=2, random_seed=2,number_of_agents=2),
          Agent(state_size=24, action_size=2, random_seed=2,number_of_agents=2)]

print(env.brain_names)
brain_name = env.brain_names[0]
print(env.brains)
brain = env.brains[brain_name]

#def ddpg(n_episodes=1, max_t=300, print_every=100):
def maddpg(n_episodes=10000, max_t=300, print_every=1):
    scores_window = deque(maxlen=100)  # last 100 scores
    scores = []
    step_counter = 0
    #env_info = env.reset(train_mode=True)[brain_name]
    #max_states = env_info.vector_observations
    max_states= np.asarray([[ 0.          ,0.92426121  ,30.2304821   ,6.80380011 ,11.56614876  ,6.051126,
  30.2304821   ,6.80380011  , 0.          ,0.92426121 ,30.2304821   ,6.80380011,
  11.56614876  ,6.051126    ,30.2304821   ,6.80380011 ,-0.39999998  ,1.09286952,
  30.2304821   ,6.80380011  ,11.88199425  ,6.051126   ,30.2304821   ,6.80380011],
 [ 0.          ,0.92129707  ,30.          ,6.60760021 ,10.87809944  ,6.051126,
  30.          ,6.60760021  , 0.          ,1.17840075 ,30.          ,7.1561985,
  11.42092896  ,6.051126    ,30.          ,7.1561985  ,-0.39999998  ,1.68351305,
  30.          ,7.1561985   ,11.9212265   ,6.051126   ,30.          ,7.1561985 ]])
    min_states = np.asarray([[-11.47814369 , -1.97238433 ,-30.         ,-10.46334839 ,-10.87809944,
   -2.10424662 ,-30.         ,-10.46334839 ,-11.47814369 , -1.97238433,
  -30.         ,-10.46334839 ,-11.42092896 , -2.36308932 ,-30.,
  -10.46334839 ,-11.47814369 , -1.97238433 ,-30.         ,-10.46334839,
  -11.9212265  , -3.25464821 ,-30.         ,-10.46334839],
 [-11.45977211 , -1.95873082 ,-30.         ,-11.60156822 ,-11.56614876,
   -2.10424662 ,-30.         ,-11.60156822 ,-11.45977211 , -1.95873082,
  -30.         ,-11.60156822 ,-11.56614876 , -2.36308932 ,-30.,
  -11.60156822 ,-11.45977211 , -1.95873082 ,-30.         ,-11.60156822,
  -11.88199425 , -3.25464821 ,-30.         ,-11.60156822]])
    #min_states = max_states
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        states = (states-min_states) / (max_states-min_states)
        #print("states shape:" +str(states.shape))
        for agent in agents:
            agent.reset()
        current_scores = np.zeros(num_agents)
        while True:
            step_counter = step_counter + 1
            actions = np.zeros((num_agents, action_size))
            for i, agent in enumerate(agents):
                actions[i] = agent.act(states[i],step_counter)
            #print(actions)
            actions = np.clip(actions, -1, 1)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations         # get next state (for each agent)
            next_states = (next_states - min_states) / (max_states - min_states)
            #max_states = np.maximum(next_states,max_states)
            #min_states = np.minimum(next_states, min_states)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            for i, agent in enumerate(agents):
                agent.step(states, actions, rewards, next_states, dones,i,agents)
            states = next_states
            #print(agent.memory.__len__())
            current_scores += env_info.rewards
            #print(t)
            if np.any(dones):                                  # exit loop if episode finished
                break
            #next_state, reward, done, _ = env.step(action)
            #agent.step(state, action, reward, next_state, done)
            #state = next_state
            #score += reward
            #if done:
            #    break
        print(current_scores)
        #print(max_states)
        #print(min_states)
        scores_window.append(np.max(current_scores))
        scores.append(np.max(current_scores))              # save most recent score
        #print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        #print("epsilon: "+str(agent.current_eps))
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            print("epsilon: "+str(agent.current_eps))
        if np.mean(scores_window)>=0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agents[0].actor_local.state_dict(), 'checkpoint_actor_0.pth')
            torch.save(agents[0].critic_local.state_dict(), 'checkpoint_critic_0.pth')
            torch.save(agents[1].actor_local.state_dict(), 'checkpoint_actor_1.pth')
            torch.save(agents[1].critic_local.state_dict(), 'checkpoint_critic_1.pth')
            break
    return scores

scores = maddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()