#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
import math
from torch.nn.modules.loss import _Loss
from bfm.bfm import BFM 
from utils.util import recon_transform
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras


class WingLoss(_Loss):
    def __init__(self, width=10, curvature=2.0, reduction="mean", weight=1.0):
        super(WingLoss, self).__init__(reduction=reduction)
        self.width = width
        self.curvature = curvature
        self.weight = torch.ones(136)
        self.weight[48:68] = self.weight[48:68] * weight 
        self.weight[116:136] = self.weight[116:136] * weight 
        self.weight = self.weight.cuda()

    def forward(self, prediction, target):
        return self.wing_loss(prediction, target, self.width, self.curvature, self.reduction)

    def wing_loss(self, prediction, target, width=10, curvature=2.0, reduction="mean"):
        diff_abs = (target - prediction).abs() * self.weight
        loss = diff_abs.clone()
        idx_smaller = diff_abs < width
        idx_bigger = diff_abs >= width
        # loss[idx_smaller] = width * torch.log(1 + diff_abs[idx_smaller] / curvature)
        loss_smaller = width * torch.log(1 + diff_abs[idx_smaller] / curvature)
        C = width - width * math.log(1 + width / curvature)
        # loss[idx_bigger] = loss[idx_bigger] - C
        loss_biger = loss[idx_bigger] - C
        loss = torch.cat((loss_smaller, loss_biger), 0)
        if reduction == "sum":
            loss = loss.sum()
        if reduction == "mean":
            loss = loss.mean()
        return loss


class LandmarkLoss(nn.Module):
    def __init__(self):
        super(LandmarkLoss, self).__init__()
        self._bfm = BFM("bfm/BFM/mSEmTFK68etc.chj")
        self.wing_loss_fn = WingLoss()
        self.img_size = 224 

        R, T = look_at_view_transform(eye=((0,0,10.0),), at=((0, 0, 0),), up=((0, 1, 0),), device="cuda:0")
        self.cameras = FoVPerspectiveCameras(R=R, T=T, fov=12.5936, degrees=True, device="cuda:0", znear = 0.01, zfar = 50., aspect_ratio = 1.)

    def get_landmark_loss(self, labels, outputs):
        mouth_index = range(48, 68) 
        left_index = range(48) 
        outputs_mouth = outputs[:, mouth_index] 
        outputs_left = outputs[:, left_index] 
        labels_mouth = labels[:, mouth_index] 
        labels_left = labels[:, left_index] 
        results = torch.mean((outputs_mouth - labels_mouth) ** 2) + 20 * torch.mean((outputs_left - labels_left) ** 2) 
        results = results/ 68. 
        return results

    def forward(self, inp_param, gt_param):
        face_shape_t, triangle, texture, face_shape = recon_transform(inp_param, self._bfm, if_edge=True) 
        gt_face_shape_t, gt_triangle, gt_texture, gt_face_shape = recon_transform(gt_param, self._bfm, if_edge=True)  

        transformed_face_shape = self.cameras.transform_points(face_shape_t) 
        landmarks = transformed_face_shape[:, self._bfm.keypoints, :]  
        landmarks = ((landmarks + 1) * self.img_size - 1)/2. 
        landmarks[:, :, :2] = self.img_size - landmarks[:, :, :2]

        gt_transformed_face_shape = self.cameras.transform_points(gt_face_shape_t) 
        gt_landmarks = gt_transformed_face_shape[:, self._bfm.keypoints, :]  
        gt_landmarks = ((gt_landmarks + 1) * self.img_size - 1)/2. 
        gt_landmarks[:, :, :2] = self.img_size - gt_landmarks[:, :, :2]

        loss = self.get_landmark_loss(gt_landmarks[:,:,:2], landmarks[:, :, :2])

        return loss

if __name__ == '__main__':
    pass
