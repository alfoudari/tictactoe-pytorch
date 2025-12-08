# Advantage Actor Critic (A2C)
# ============================
# Actor:  Optimize a neural network to produce a probability distribution of actions.
# Critic: Optimize a neural network to produce V(s) which allows us to compute the
#         advantage of an action-state pair A(s,a). A(s,a) is used to scale gradients
#         computed by the actor, hence acting as a critic. Scaling gradients in this
#         way allows the critic to reinforce actor actions.

import torch
import torch.optim as optim
import torch.nn.functional as F 
import numpy as np
import random
import math
import copy
from torch import nn
from collections import namedtuple
from itertools import count
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Agent:
    def __init__(self, 
                env, 
                net, 
                name="", 
                learning_rate=3e-4, 
                optimizer=optim.Adam,
                discount=0.99, 
                eval_episodes_count=100, 
                logdir='', 
                dev=None):
        global device
        device = dev

        self.name = name
        self.learning_rate = learning_rate              # alpha
        self.optimizer = optimizer
        self.discount = discount                        # gamma
        self.eval_episodes_count = eval_episodes_count  # number of episodes for evaluation
        self.env = env
        self.net = net(self.env.unwrapped.observation_space_n, self.env.unwrapped.action_space_n).to(device)
        self.logdir = logdir

    def train(self):
        writer = SummaryWriter(logdir=self.logdir, comment=f"-{self.name}" if self.name else "")
        
        ep_idx = 1
        running_reward = 0

        # infinite episodes until threshold is met off
        while True:
            ep_reward, step_rewards, saved_actions, entropy = self.run_episode()

            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

            # returns
            returns = self.calculate_returns(step_rewards)
            returns = self.standardize_returns(returns)

            # optimize policy_net
            loss = self.optimize(returns, saved_actions, entropy)

            # tensorboard metrics
            writer.add_scalar("train/loss", loss, ep_idx)
            writer.add_scalar("train/running_reward", running_reward, ep_idx)

            # evaluate policy
            if ep_idx % 500 == 0:
                stop, avg_rewards = self.evaluate_policy(running_reward)

                writer.add_scalar("train/avg_rewards", avg_rewards, ep_idx)

                if stop:
                    break
            
            ep_idx = ep_idx + 1

        # save model
        policy_name = self.name if self.name else "a2c"
        torch.save(self.net.state_dict(), f"policies/{policy_name}")
        writer.close()

    def run_episode(self):
        state, info = self.env.reset()
        step_rewards = []
        saved_actions = []
        entropy = 0

        # run a single episode
        while True:
            # choose an action
            action, action_dist, dist_entropy = self.select_action(state, self.env.unwrapped.legal_actions, saved_actions)
            # take a step in env
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # calculate entropy
            entropy += dist_entropy

            # accumulate rewards
            step_rewards.append(reward)

            state = next_state

            if done:
                return sum(step_rewards), step_rewards, saved_actions, entropy

    def select_action(self, state, legal_actions, saved_actions):
        mask = torch.zeros(self.env.unwrapped.action_space_n).index_fill(0, torch.LongTensor(legal_actions),  1)
        action_dist, value = self.net(torch.Tensor(state).to(device), mask)

        m = Categorical(action_dist)
        action = m.sample()

        saved_actions.append(SavedAction(m.log_prob(action), value.squeeze(0)))

        return action.item(), action_dist, m.entropy()

    def calculate_returns(self, step_rewards):
        R = 0
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in step_rewards[::-1]:
            # calculate the discounted value
            R = r + R * self.discount
            returns.insert(0, R)

        return returns

    def standardize_returns(self, returns):
        # smallest positive number such that 1.0 + eps != 1.0
        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(returns)
        # calculate z-scores; standardize the distribution
        return (returns - returns.mean()) / (returns.std() + eps)

    def optimize(self, returns, saved_actions, entropy):
        """
        Calculates actor and critic loss and performs backprop.
        """
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss.
            # scale probabilities by advantage
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum() + 0.001 * entropy

        # reset gradients
        optimizer = self.optimizer(params=self.net.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        # perform backprop; compute gradient
        loss.backward()
        # clip gradients
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        # update net parameters
        optimizer.step()

        return loss

    def evaluate_policy(self, running_reward):
        rewards = []
        for _ in range(self.eval_episodes_count):
            state, info = self.env.reset()
            ep_reward = 0
            while True:
                mask = torch.zeros(self.env.unwrapped.action_space_n).index_fill(0, torch.LongTensor(self.env.unwrapped.legal_actions),  1)
                action_dist, value = self.net(torch.Tensor(state).to(device), mask)
                action = torch.argmax(action_dist).item()
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                rewards.append(reward)

                state = next_state

                if done:
                    break

        avg_rewards = sum(rewards)/self.eval_episodes_count if self.eval_episodes_count > 0 else 0

        # stop training if thresholds are met
        running_reward_achieved = running_reward >= self.env.spec.reward_threshold
        avg_rewards_achieved = avg_rewards >= self.env.spec.reward_threshold

        return running_reward_achieved and avg_rewards_achieved, avg_rewards
            