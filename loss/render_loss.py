#!/usr/bin/env python3
# coding: utf-8
import torch 
import torch.nn as nn
from utils.util import recon_transform, get_mask_from_render
from bfm.bfm import BFM
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex 
    )
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.structures import Meshes


class RenderLoss(nn.Module):
    def __init__(self, opt):
        super(RenderLoss, self).__init__()
        self._opt = opt
        self._bfm = BFM("bfm/BFM/mSEmTFK68etc.chj")

        if torch.cuda.is_available():
            self._device = torch.device("cuda:%d" % self._opt.gpu_ids[0])
            torch.cuda.set_device(self._device)

        R, T = look_at_view_transform(eye=((0,0,10.0),), at=((0, 0, 0),), up=((0, 1, 0),), device=self._device)
        cameras = FoVPerspectiveCameras(R=R, T=T, fov=12.5936, degrees=True, device=self._device, znear = 0.01, zfar = 50., aspect_ratio = 1.)  
        bp_new = BlendParams(background_color=(0, 0, 0)) 
        raster_settings = RasterizationSettings(image_size=224, blur_radius=0.0, faces_per_pixel=1,)
        lights = PointLights(location=[[0, 0, 1e5]], ambient_color=[[1.0, 1, 1]],specular_color=[[0,0,0,]],diffuse_color=[[0,0,0]], device=self._device) 
        self._renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                cameras=cameras,
                lights=lights,
                blend_params=bp_new, 
                device=self._device
            )
        )

    def _deform_params(self, params):
        face_shape_t, triangle, face_texture = recon_transform(params, self._bfm)
        
        N = face_texture.shape[0] 
        triangle = torch.from_numpy(triangle).unsqueeze(0).cuda()
        triangle = triangle.repeat(N, 1, 1)

        return face_shape_t, face_texture, triangle 


    def forward(self, inp_param, gt_param, target_img):

        shapes, colors, tris = self._deform_params(inp_param) 
        textures = TexturesVertex(verts_features=colors.to(self._device)) 
        face_mesh = Meshes(
            verts=shapes.to(self._device),   
            faces=tris.to(self._device), 
            textures=textures
        )

        images_render = self._renderer(face_mesh) # (N, 224, 224, 4) RGB
        img = images_render[:, :, :, :3]

        # get mask from gt_params
        gt_shapes, gt_colors, tris = self._deform_params(gt_param)
        gt_textures = TexturesVertex(verts_features=gt_colors.to(self._device)) 
        gt_face_mesh = Meshes(
            verts=gt_shapes.to(self._device),   
            faces=tris.to(self._device), 
            textures=gt_textures 
        )

        gt_images_render = self._renderer(gt_face_mesh)
        gt_img = gt_images_render[:, :, :, :3]
        gt_img = torch.clamp(gt_img, 0, 1, out=None)

        mask = get_mask_from_render(gt_img).cuda()

        diff = (target_img * mask - img * mask) ** 2
        loss = torch.mean(diff)

        return loss 
