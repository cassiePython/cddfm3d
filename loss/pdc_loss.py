#!/usr/bin/env python3
# coding: utf-8
import torch.nn as nn

class PDCLoss(nn.Module):
    def __init__(self):
        super(PDCLoss, self).__init__()

    def forward(self, input, target, epoch=None):

        loss = (target - input) ** 2

        return loss.mean()
