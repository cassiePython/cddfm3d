#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
from bfm.bfm import BFM
from utils.util import recon_transform

class VDCLoss(nn.Module):
    def __init__(self):
        super(VDCLoss, self).__init__()
        self._bfm = BFM("bfm/BFM/mSEmTFK68etc.chj")
        self._SML1_loss = torch.nn.SmoothL1Loss(reduction='mean')

    def forward(self, input, target):

        face_shape_t, triangle, face_texture = recon_transform(input, self._bfm)
        gt_vertex, triangle, gt_face_texture = recon_transform(target, self._bfm)

        diff = (gt_vertex - face_shape_t) ** 2
        loss = torch.mean(diff)

        return loss
