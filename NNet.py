import torch
import torch.nn as nn
import torch.nn.functional as function

class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, seed, fc1_dim=128, fc2_dim=64):
        """
        :param state_dim: dimension of the state
        :param action_dim: dimension of the action
        :param seed: Random seed
        :param fc1_dim: dimension of first hidden layer
        :param fc2_dim: dimension of second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = seed
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, action_dim)

    def forward(self, state):
        """
        :param state: (self.state_dim x batch size) state data by batch size
        :return: (self.action_dim x batch size) action data by batch size
        """
        x = function.relu(self.fc1(state))
        x = function.relu(self.fc2(x))
        x = self.fc3(x)
        return x