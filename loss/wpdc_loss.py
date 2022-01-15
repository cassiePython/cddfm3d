#!/usr/bin/env python3
# coding: utf-8
import torch
import torch.nn as nn
from utils.util import parse_param, recon_transform
from bfm.bfm import BFM

class WPDCLoss(nn.Module):
    """Inspired from https://github.com/cleardusk/3DDFA, but is different from the original design"""
    def __init__(self):
        super(WPDCLoss, self).__init__()

        self.resample_num = 128
        self._bfm = BFM("bfm/BFM/mSEmTFK68etc.chj")
        self.index = None

    def replace_param(self, param, param_gt):
        param_r = param_gt.clone()
        self.index = torch.randperm(param.shape[1]) < self.resample_num
        param_r[:, self.index] = param[:, self.index]
        return param_r

    def _calc_weights_resample(self, param, param_gt):
        param = torch.tensor(param.data.clone(), requires_grad=False)
        param_gt = torch.tensor(param_gt.data.clone(), requires_grad=False)
        param_r = self.replace_param(param, param_gt)

        face_shape_r, _, face_texture_r = recon_transform(param_r, self._bfm)
        face_shape_gt, _, face_texture_gt = recon_transform(param_gt, self._bfm)

        magic_num = 10000  # scale
        weight = ((face_shape_r - face_shape_gt) ** 2 + (face_texture_r - face_texture_gt) ** 2)
        weight_norm = torch.norm(weight)
        weight /= weight_norm
        weight = weight.mean() * magic_num
        weights = torch.ones_like(param_gt)
        weights[:, self.index] = weight

        return weights

    def forward(self, input, target):
        weights = self._calc_weights_resample(input, target)
        loss = weights * (target - input) ** 2
        return loss.mean()
