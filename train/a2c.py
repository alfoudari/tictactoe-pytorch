# Advantage Actor Critic (A2C)

import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import sys
import torch
import gym
import env # module init needs to run
from prop.algorithms.a2c import Agent
from prop.net.feed_forward import FeedForward

class CNNNet(FeedForward):
    """ compute action probability distribution and state value """
    def __init__(self, obs_size, n_actions):
        # model is initiated in parent class, set params early.
        self.obs_size = obs_size
        self.n_actions = n_actions
        super(CNNNet, self).__init__()

    def model(self):
        common = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding=1), # 3 channels -> `out_channels` different kernels/feature maps
            nn.ReLU(), # negative numbers -> 0
            nn.MaxPool2d(kernel_size=2, stride=1), # deformation invariance; subtle changes are captured
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten(),
            nn.Linear(256 * 4, 64),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(64, self.n_actions)
        self.critic_head = nn.Linear(64, 1)
        return common

    def forward(self, x, mask):
        # transfrom from one tensor of shape (9) into 3 tensors of shape (3,3) each
        empty = torch.zeros(x.size()).masked_scatter_((x == 0), torch.ones(x.size())).view(-1, 3, 3)
        player1 = torch.zeros(x.size()).masked_scatter_((x == 1), torch.ones(x.size())).view(-1, 3, 3)
        player2 = torch.zeros(x.size()).masked_scatter_((x == 2), torch.ones(x.size())).view(-1, 3, 3)
        cnn_input = torch.stack((empty, player1, player2), dim=1)

        # shared layers among actor and critic
        common = self.net(cnn_input)

        # actor layer
        actions = self.actor_head(common)
        if mask is not None:
            actions = self.mask_actions(actions, mask)
        action_dist = F.softmax(actions, dim=1)

        # critic layer
        value = self.critic_head(common)

        return action_dist, value

if __name__ == "__main__":
    # flags
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', action="store", dest="device", type=str)
    args = parser.parse_args()

    # device
    if args.device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        if args.device not in ["cpu", "cuda"]:
            sys.exit('device must be "cpu" or "cuda"')
        device = args.device

    device = torch.device(device)
    print(f"using: {device}")

    # setup env and agent and start training
    env = gym.make('TicTacToeRandomPlayer-v0')
    agent = Agent(
        env=env, 
        net=CNNNet,
        name="a2c-cnn",
        learning_rate=3e-5,
        optimizer=optim.Adam,
        discount=1,
        dev=device)
    agent.train()
