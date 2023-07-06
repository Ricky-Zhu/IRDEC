import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import numpy as np


def glorot_init(p):
    if isinstance(p, nn.Linear):
        nn.init.xavier_normal_(p.weight.data, gain=1.)
        nn.init.zeros_(p.bias)


class tanh_gaussian_actor(nn.Module):
    def __init__(self, input_dims, action_dims, hidden_size):
        super(tanh_gaussian_actor, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_dims)
        self.log_std = nn.Linear(hidden_size, action_dims)

        # init the networks
        self.apply(glorot_init)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        std = F.softplus(self.log_std(x))

        return (mean, std)


class Critic(nn.Module):
    """
    construct a classifier C(s,a) -> [0,1]
    """

    def __init__(self, obs_dim, action_dim, hidden_size, loss_type='q'):
        super(Critic, self).__init__()
        self.loss_type = loss_type
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q = nn.Linear(hidden_size, 1)

        self.apply(glorot_init)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q = self.q(x)
        if self.loss_type == 'c':
            q = torch.sigmoid(q)
        return q


class ICMModel(nn.Module):
    def __init__(self, input_size, output_size, use_action_embedding=False, device='cpu'):
        '''

        :param input_size: state dimension
        :param output_size: action dimension
        '''
        super(ICMModel, self).__init__()
        self.use_action_embedding = use_action_embedding

        self.input_size = input_size
        if self.use_action_embedding:
            self.action_embedding_layer = nn.Linear(output_size, 32)
            output_size = 32
        self.output_size = output_size

        self.feature = nn.Linear(input_size, 512)
        self.inverse_net = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

        self.residual = [nn.Sequential(
            nn.Linear(output_size + 512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
        ).to(device)] * 4

        self.forward_net_1 = nn.Sequential(
            nn.Linear(output_size + 512, 512),
            nn.LeakyReLU()
        )
        self.forward_net_2 = nn.Sequential(
            nn.Linear(output_size + 512, 512),
        )

        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, inputs):
        state, next_state, action = inputs

        if self.use_action_embedding:
            action = self.action_embedding_layer(action)

        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)

        ## get the impact diff
        with torch.no_grad():
            impact_bonus = F.mse_loss(encode_state, encode_next_state, reduction='none').mean(-1).detach()

        # get pred action
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action)
        # ---------------------

        # get pred next state
        pred_next_state_feature_orig = torch.cat((encode_state, action), 1)
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)

        # residual
        for i in range(2):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig

        pred_next_state_feature = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1))

        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action, action, impact_bonus
