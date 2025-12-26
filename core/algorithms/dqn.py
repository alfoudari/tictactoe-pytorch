import torch
import torch.optim as optim
import torch.nn.functional as F 
import numpy as np
import random
import math
import copy
from collections import namedtuple
from itertools import count, compress
from tensorboardX import SummaryWriter
from core.buffers.priority_replay_buffer import PrioritizedReplayBuffer

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'mask'))

class Agent:
    def __init__(self, 
                 env, 
                 net, 
                 name="", 
                 double=True, 
                 learning_rate=3e-4, 
                 batch_size=128,
                 optimizer=optim.Adam,
                 loss_cutoff=0.1,
                 max_std_dev=-1,
                 epsilon_start=1, 
                 epsilon_end=0.1, 
                 epsilon_decay=1000, 
                 discount=0.99, 
                 target_net_update=5000,
                 eval_episodes_count=1000, 
                 eval_every=1000,
                 replay_buffer=PrioritizedReplayBuffer,
                 replay_buffer_capacity=1000000, 
                 extra_metrics=None,
                 logdir=None, 
                 dev=None):
        global device
        device = dev

        self.name = name
        self.double = double                # double q learning
        self.loss_cutoff = loss_cutoff      # training stops at loss_cutoff
        self.max_std_dev = max_std_dev      # max std deviation allowed to stop training; >= 0 to activate
        self.learning_rate = learning_rate  # alpha
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.epsilon_start = epsilon_start  # start with 100% exploration
        self.epsilon_end = epsilon_end      # end with 10% exploration
        self.epsilon_decay = epsilon_decay  # higher value = slower decay
        self.discount = discount            # gamma
        self.target_net_update = target_net_update     # number of steps to update target network
        self.eval_episodes_count = eval_episodes_count # number of episodes to evaluate
        self.eval_every = eval_every # number of steps to run evaluations at
        self.replay_buffer = replay_buffer(replay_buffer_capacity)
        self.env = env
        self.policy_net = net(self.env.unwrapped.observation_space_n, self.env.unwrapped.action_space_n).to(device) # what drives current actions; uses epsilon.
        self.target_net = net(self.env.unwrapped.observation_space_n, self.env.unwrapped.action_space_n).to(device) # copied from policy net periodically; greedy.
        self.logdir = logdir

        # init target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def train(self):
        writer = SummaryWriter(logdir=self.logdir, comment=f"-{self.name}" if self.name else "")
        steps = 1
        recent_loss = []
        recent_eval = []
        avg_rewards = 0
        while True:
            # fill replay buffer with one episode from the current policy (epsilon is used)
            self.load_replay_buffer(policy=self.policy_net, steps=steps)

            # sample transitions
            transitions, idxs, is_weights = self.replay_buffer.sample(self.batch_size)
            if len(transitions) < self.batch_size:
                continue

            # optimize policy_net
            loss = self.optimize(transitions, idxs, is_weights)
            # keep track of recent losses and truncate list to latest `eval_every` losses
            recent_loss.append(loss)
            recent_loss = recent_loss[-self.eval_every:]

            # tensorboard metrics
            epsilon = Agent.eps(self.epsilon_start, self.epsilon_end, self.epsilon_decay, steps)
            writer.add_scalar("env/epsilon", epsilon, steps)
            writer.add_scalar("env/replay_buffer", len(self.replay_buffer), steps)
            writer.add_scalar("train/loss", loss, steps)

            # update the target network, copying all weights and biases in policy_net to target_net
            if steps % self.target_net_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # run evaluation
            if steps % self.eval_every == 0:
                avg_rewards, stddev = self.evaluate_policy(self.policy_net)

                writer.add_scalar("train/avg_rewards", avg_rewards, steps)
                writer.add_scalar("train/ep_rewards_std", stddev, steps)

                recent_eval.append(avg_rewards)
                recent_eval = recent_eval[-10:]

                loss_achieved = sum(recent_loss)/len(recent_loss) <= self.loss_cutoff
                avg_rewards_achieved = sum(recent_eval)/len(recent_eval) >= self.env.spec.reward_threshold
                std_dev_achieved = (self.max_std_dev < 0) or (self.max_std_dev >= 0 and stddev <= self.max_std_dev)

                if loss_achieved and avg_rewards_achieved and std_dev_achieved:
                    break

            steps = steps + 1

        # save model
        policy_name = self.name if self.name else "dqn"
        torch.save(self.policy_net.state_dict(), f"policies/{policy_name}")
        writer.close()

    @staticmethod
    def eps(start, end, decay, steps): 
        # compute epsilon threshold
        return end + (start - end) * math.exp(-1. * steps / decay)

    @staticmethod
    def legal_actions_to_mask(legal_actions, action_space_n):
        mask = [0]*action_space_n
        for n in legal_actions:
            mask[n] = 1
        return mask

    def load_replay_buffer(self, policy=None, episodes_count=1, steps=0):
        """ load replay buffer with episodes_count """
        for eps_idx in range(episodes_count):
            state, info = self.env.reset()
            while True:
                legal_actions = self.env.unwrapped.legal_actions
                action = self.select_action(
                    policy=policy,
                    state=state,
                    epsilon=True,
                    steps=steps,
                    legal_actions=legal_actions).item()

                # perform action
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # insert into replay buffer
                mask = Agent.legal_actions_to_mask(legal_actions, self.env.unwrapped.action_space_n)
                transition = Transition(state, action, next_state if not done else None, reward, mask)
                # set error of new transitions to a very high number so they get sampled
                self.replay_buffer.push(self.replay_buffer.tree.total, transition)

                if done:
                    break
                else:
                    # transition
                    state = next_state

    def evaluate_policy(self, policy):
        ep_rewards = []
        for _ in range(self.eval_episodes_count):
            state, info = self.env.reset()
            ep_reward = 0
            while True:
                legal_actions = self.env.unwrapped.legal_actions
                action = self.select_action(
                    policy=policy,
                    state=state,
                    epsilon=False,
                    legal_actions=legal_actions).item()
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward

                if done:
                    ep_rewards.append(ep_reward)
                    break
                else:
                    state = next_state

        return np.mean(ep_rewards), np.std(ep_rewards)

    def select_action(self, policy, state, epsilon=False, steps=None, legal_actions=[]):
        """
        selects an action with a chance of being random if epsilon is True,
        otherwise selects the action produced by policy.
        """
        if epsilon:
            if steps == None:
                raise ValueError(f"steps must be an integer. Got = {steps}")

            # pick a random number
            sample = random.random()
            # see what the dice rolls
            threshold = Agent.eps(self.epsilon_start, self.epsilon_end, self.epsilon_decay, steps)
            if sample <= threshold:
                # explore
                action = random.choice([i for i in range(self.env.unwrapped.action_space_n+1) if i in legal_actions])
                return torch.tensor([[action]], device=device, dtype=torch.long)

        # greedy action
        with torch.no_grad():
            # index of highest value item returned from policy -> action
            state = torch.Tensor(state).to(device)
            mask = torch.zeros(self.env.unwrapped.action_space_n, device=device).index_fill(0, torch.LongTensor(legal_actions).to(device), 1)
            return policy(state, mask).argmax().view(1, 1)

    def optimize(self, transitions, idxs, is_weights):
        # n transitions -> 1 transition with each attribute containing all the
        # data point values along its axis.
        # e.g. batch.action = list of all actions from each row
        batch = Transition(*zip(*transitions))

        # Compute state action values; the value of each action in batch according
        # to policy_net (feeding it a state and emitting an probability distribution).
        # These are the values that our current network think are right and we want to correct.
        state_action_values = self.state_action_values(batch)
        # compute expected state action values (reward + value of next state according to target_net)

        expected_state_action_values = self.expected_state_action_values(batch)

        # calculate difference between actual and expected action values
        batch_loss = F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction='none')
        loss = (sum(batch_loss * torch.FloatTensor(is_weights).to(device).unsqueeze(1))/self.batch_size).squeeze()

        # update priority
        for i in range(self.batch_size):
            self.replay_buffer.update(idxs[i], batch_loss[i].item())

        # optimizer
        optimizer = self.optimizer(params=self.policy_net.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        # calculate gradients
        loss.backward()
        for param in self.policy_net.parameters():
            # clip gradients
            param.grad.data.clamp_(-1, 1)
        # optimize policy_net
        optimizer.step()

        return loss

    def state_action_values(self, batch):
        """ 
        Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        columns of actions taken. These are the actions which would've been taken
        for each batch state according to policy_net.
        """
        # list -> tensor
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
        mask_batch = torch.tensor(np.array(batch.mask), dtype=torch.float32, device=device)
        action_batch = torch.tensor(np.array(batch.action), dtype=torch.float32, device=device)
        # get action values for each state in batch
        state_action_values = self.policy_net(state_batch, mask_batch)
        # select action from state_action_values according to action_batch value
        return state_action_values.gather(1, action_batch.unsqueeze(1).long())

    def expected_state_action_values(self, batch):
        """
        Compute V(s_{t+1}) for all next states.
        Expected values of actions for non_final_next_states are computed based
        on the "older" target_net; selecting their best reward with max(1)[0].
        This is merged based on the mask, such that we'll have either the expected
        state value or 0 in case the state was final.
        """
        # a bool list indicating if next_state is final (s is not None)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.tensor(np.array([s for s in batch.next_state if s is not None]), dtype=torch.float32, device=device)
        # get legal actions for non final states; (i, v) -> (list of legal actions, non_final_state)
        next_mask = torch.tensor(np.array([i for (i, v) in zip(list(batch.mask), non_final_mask.tolist()) if v]), dtype=torch.float32, device=device)
        # initialize next_state_values to zeros
        next_state_values = torch.zeros(self.batch_size, device=device)

        if len(non_final_next_states) > 0:
            if self.double:
                # double q learning: get actions from policy_net and get their values according to target_net; decoupling
                #                    action selection from evaluation reduces the bias imposed by max in single dqn.
                # next_state_actions: action selection according to policy_net; Q(st+1, a)
                next_state_actions = self.policy_net(non_final_next_states, next_mask).max(1)[1].unsqueeze(-1)
                # next_state_values: action evaluation according to target_net; max Q`(st+1, max Q(st+1, a) )
                next_state_values[non_final_mask] = self.target_net(non_final_next_states, next_mask).gather(1, next_state_actions).squeeze(-1)
            else:
                # max Q`(st+1, a)
                next_state_values[non_final_mask] = self.target_net(non_final_next_states, next_mask).max(1)[0].detach()

        # Compute the expected Q values
        # reward + max Q`(st+1, a) * discount
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
        state_action_values = reward_batch + (next_state_values.unsqueeze(1) * self.discount)

        return state_action_values