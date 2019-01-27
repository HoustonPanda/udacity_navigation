import numpy as np
import random
from collections import namedtuple, deque

from NNet import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

# constants
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LR = 1e-3
BUFFER_SIZE = int(1e6)
UPDATE_CYCLE = 8
batch_size = 64
GAMMA = 0.99
TAU = 1e-3


class Agent():
    """
    agent that interact with the environment and learn
    """

    def __init__(self, state_size, action_size, seed, filename=None):
        """
        initialize the agent
        :param state_size:
        :param action_size:
        :param seed:
        :param filename:
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # initialize Q-network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr = LR)

        # load parameters from file
        if filename:
            try:
                weights = torch.load(filename)
                self.qnetwork_local.load_state_dict(weights)
                self.qnetwork_target.load_state_dict(weights)
                print('weight load success from file')
            except:
                print('No available weight file')

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, batch_size, seed)

        # time step
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """
        save the experience into memory
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        # save experience into memory
        self.memory.add(state, action, reward, next_state, done)

        # learn every UPDATE_CYCLE time steps
        self.t_step = (self.t_step + 1) % UPDATE_CYCLE
        if self.t_step == 0 and len(self.memory) > batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)


    def act(self, state, eps=0):
        """
        return the action based on given state and current policy
        :param state:
        :param eps:
        :return:
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # epsilon greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        update value parameters using given batch of experience tuples
        :param experiences:
        :param gamma:
        :return:
        """
        states, actions, rewards, next_states, dones = experiences

        Q_target_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_target = rewards + (gamma * Q_target_next * (1-dones))
        #print('action',len(actions))
        #print('predictq', len(self.qnetwork_local(states)))
        Q_local = self.qnetwork_local(states).gather(1, actions.long())
        #Q_local = self.qnetwork_local(states).gather(1, actions.unsqueeze(1))
        loss = F.mse_loss(Q_local, Q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        soft update the target model
        :param local_model:
        :param target_model:
        :param tau:
        :return:
        """
        for target_parameter, local_parameter in zip(target_model.parameters(), local_model.parameters()):
            target_parameter.data.copy_(tau*local_parameter.data + (1.0-tau)*target_parameter.data)

class ReplayBuffer:
    """
    fixed-sized buffer to store experience tuples
    """
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        initialize a replaybuffer object
        :param action_size:
        :param buffer_size:
        :param batch_size:
        :param seed:
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
        self.seed = seed

    def add(self, state, action, reward, next_state, done):
        """
        add a new experience to memory
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """
        randomly sample a batch of experiences from memory buffer
        :return:
        """
        experiences = random.sample(self.memory, k = self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """
        return the current size of internal memory
        :return:
        """
        return len(self.memory)
