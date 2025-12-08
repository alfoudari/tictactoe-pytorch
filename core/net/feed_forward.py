import abc
import torch
import torch.nn as nn 
import math

class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.net = self.model()

    @abc.abstractmethod
    def model(self):
        """
        Return a feed forward network model.
        """

    def forward(self, x, mask=None):
        # forward pass
        actions = self.net(x)
        # mask actions
        if mask is not None:
            actions = self.mask_actions(actions, mask)
        return actions

    def mask_actions(self, actions, mask):
        # turn single actions into a batch with a single action
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        if len(actions) != len(mask):
            raise Exception("actions and mask batches have different sizes.")

        for idx, _ in enumerate(actions):
            for jdx in [i for i, flag in enumerate(mask[idx]) if flag==0]:
                actions[idx][jdx] = -math.inf

        return actions