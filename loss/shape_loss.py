#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
from bfm.bfm import BFM 
from utils.util import recon_transform 
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras 


class ShapeLoss(nn.Module):
    def __init__(self):
        super(ShapeLoss, self).__init__()
        self._bfm = BFM("bfm/BFM/mSEmTFK68etc.chj")

    def get_landmark_loss(self, labels, outputs):
        mouth_index = range(48, 68) 
        left_index = range(17, 48) 
        outputs_mouth = outputs[:, mouth_index] 
        outputs_left = outputs[:, left_index] 
        labels_mouth = labels[:, mouth_index] 
        labels_left = labels[:, left_index] 
        results = 100 * torch.mean((outputs_mouth - labels_mouth) ** 2) + 20 * torch.mean((outputs_left - labels_left) ** 2) 
        results = results/ 51. 
        return results 

    def get_edge_loss(self, labels, outputs):
        edge = range(17)
        outputs_edge = outputs[:, edge]
        labels_edge = labels[:, edge]
        results = torch.mean((outputs_edge-labels_edge) **2 ) 
        results = results / 17.
        return results 


    def forward(self, inp_param, gt_param, Pw):


        face_shape_t, triangle, texture, face_shape = recon_transform(inp_param, self._bfm, if_edge=True) 
        gt_face_shape_t, gt_triangle, gt_texture, gt_face_shape = recon_transform(gt_param, self._bfm, if_edge=True)  
        Pw_face_shape_t, Pw_triangle, Pw_texture, Pw_face_shape = recon_transform(Pw, self._bfm, if_edge=True)   

        Pw_landmarks = Pw_face_shape[:, self._bfm.keypoints, :]  
        edge_landmarks = face_shape[:, self._bfm.keypoints, :]
        gt_edge_landmarks = gt_face_shape[:, self._bfm.keypoints, :]


        loss_edge = self.get_edge_loss(gt_edge_landmarks[:,:,:2], edge_landmarks[:, :, :2]) # 原来是直接使用的landmarks和gt_landmarks 
        loss_org = self.get_landmark_loss(Pw_landmarks[:,:,:2], edge_landmarks[:, :, :2])   


        loss = loss_edge + loss_org 
        
        return loss

if __name__ == '__main__':
    pass
