from __future__ import division
from models.model_factory import ModelsFactory
from options.test_options import TestOptions
import torch
import numpy as np
from torchvision import utils
from utils.util import recon_transform, parse_styles
from bfm.bfm import BFM
from skimage import io
from stylegan2.model import Generator
from pytorch3d.structures import Meshes
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
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from tqdm import tqdm
import os
import shutil

def write_obj_with_colors(obj_name, vertices, triangles, colors):
    triangles = triangles.copy() # meshlab start with 1
    triangles += 1

    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'

    # write obj
    with open(obj_name, 'w') as f:
        # write vertices & colors
        for i in range(vertices.shape[1]):
            s = 'v {:.4f} {:.4f} {:.4f} {} {} {}\n'.format(vertices[0, i], vertices[1, i], vertices[2, i], colors[0, i],
                                               colors[1, i], colors[2, i])
            f.write(s)

        # write f: ver ind/ uv ind
        for i in range(triangles.shape[1]):
            s = 'f {} {} {}\n'.format(triangles[0, i], triangles[1, i], triangles[2, i])
            f.write(s)

            
class Test:
    def __init__(self):
        self._opt = TestOptions().parse()
        if torch.cuda.is_available():
            self._device = torch.device("cuda:%d" % self._opt.gpu_ids[0])  
            torch.cuda.set_device(self._device) 

        data_loader_test = CustomDatasetDataLoader(self._opt, is_for_train=False)
        self._dataset_test = data_loader_test.load_data()
        self._dataset_test_size = len(data_loader_test)
        print('#test images = %d' % self._dataset_test_size)

        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt, is_train=False)
        self._bfm = BFM("bfm/BFM/mSEmTFK68etc.chj")
        self._epoch = self._opt.load_epoch

        self._img_dir = os.path.join(self._opt.data_dir, self._opt.image_dir)
        self._save_dir = self._opt.save_dir
        self._save_name = self._opt.name
        os.makedirs(self._save_dir, exist_ok=True)

        R, T = look_at_view_transform(eye=((0,0,10.0),), at=((0, 0, 0),), up=((0, 1, 0),), device=self._device)
        cameras = FoVPerspectiveCameras(R=R, T=T, fov=12.5936, degrees=True, device=self._device, znear = 0.01, zfar = 50.,aspect_ratio = 1.)  
        bp_new = BlendParams(background_color=(0, 0, 0)) 
        raster_settings = RasterizationSettings(image_size=1024, blur_radius=0.0, faces_per_pixel=1,)
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

        # Stylegan2
        latent = 512
        n_mlp = 8
        size = 1024#224#1024
        channel_multiplier = 2
        self.g_ema = Generator(size, latent, n_mlp, channel_multiplier=channel_multiplier)
        checkpoint = torch.load("stylegan2/checkpoint/stylegan2-ffhq-config-f.pt")
        self.g_ema.load_state_dict(checkpoint['g_ema'])
        self.g_ema.cuda()

    def save_obj(self, param, obj_name):
        face_shape_t, triangle, face_texture = recon_transform(param, self._bfm)
        vertices = face_shape_t.squeeze(0).cpu().numpy().T
        triangles = triangle.T
        colors = face_texture.squeeze(0).cpu().numpy().T
        colors = np.clip(colors, 0, 1)
        write_obj_with_colors(obj_name, vertices, triangles, colors)


    def save_renderimg(self, param, img_name):
        face_shape_t, triangle, face_texture = recon_transform(param, self._bfm)
        face_shape_t = face_shape_t.squeeze(0)
        tri = torch.from_numpy(triangle).cuda()
        textures = TexturesVertex(verts_features=face_texture)
        face_mesh = Meshes(verts=[face_shape_t], faces=[tri], textures=textures)
        images = self._renderer(face_mesh)
        images = torch.clamp(images, 0, 1, out=None)
        images = images[0, ..., :3].cpu().numpy()
        images = (images * 255).astype(np.uint8)
        img_name = img_name + 'render.png'
        io.imsave(img_name, images)


    def test(self):
        save_dir = os.path.join(self._save_dir, self._save_name)
        os.makedirs(save_dir, exist_ok=True)
        for i_test_batch, test_batch in enumerate(tqdm(self._dataset_test)):
            params = test_batch['param'].cuda()

            latent = test_batch['latent'].cuda()
            params_pred = self._model.forward_test(latent) 
            #print (params_pred) 

            assert len(test_batch['sample_id']) == 1  # assert batch-size=1 in test phase
            sample_id = test_batch['sample_id'][0] # image name

            save_obj_path = os.path.join(save_dir, sample_id)
            save_render_path = os.path.join(save_dir, sample_id)

            src_img_path = os.path.join(self._img_dir, sample_id + ".png")
            tar_img_path = os.path.join(save_dir, sample_id + ".png")
            shutil.copy(src_img_path, tar_img_path)

            #self.save_renderimg(params, save_render_path + "gt_")
            #self.save_obj(params, save_obj_path + "gt_")

            self.save_renderimg(params_pred, save_render_path + "pred_")
            #self.save_obj(params_pred, save_obj_path + "pred_")

    def test_shape(self):
        save_dir = os.path.join(self._save_dir, self._save_name)
        os.makedirs(save_dir, exist_ok=True)

        pca_id = 1
        #pca_val = -10.0
        pca_val = 20.0
        for i_test_batch, test_batch in enumerate(tqdm(self._dataset_test)):
            latent = test_batch['latent'].cuda()
            constant = test_batch['constant'].cuda()
            constant = constant.view(1, 512, 4, 4)

            latent_pred, params_source, params_pred = self._model.forward_test(latent, pca_id, pca_val)

            # img generated by StyleGAN2
            #styles = parse_styles(latent_pred, flatten=True)
            styles = parse_styles(latent_pred, flatten=True)
            img = self.g_ema.forward_test(styles=styles, constant=constant)


            assert len(test_batch['sample_id']) == 1  # assert batch-size=1 in test phase
            sample_id = test_batch['sample_id'][0]  # image name

            save_obj_path = os.path.join(save_dir, sample_id)
            save_render_path = os.path.join(save_dir, sample_id)

            src_img_path = os.path.join(self._img_dir, sample_id + ".png")
            tar_img_path = os.path.join(save_dir, sample_id + ".png")
            shutil.copy(src_img_path, tar_img_path)
            pred_img_path = "%spred_%d_%0.2f.png" % (sample_id, pca_id, pca_val)
            pred_img_path = os.path.join(save_dir, pred_img_path)
            utils.save_image(img, pred_img_path, nrow=1, normalize=True, range=(-1, 1))

            self.save_renderimg(params_source, save_render_path + "source_")
            self.save_renderimg(params_pred, save_render_path + "pred_")

            self.save_obj(params_source, save_obj_path + "source_")
            self.save_obj(params_pred, save_obj_path + "pred_")




if __name__ == "__main__":
    model = Test()
    model.test() # test the APNet
    model.test_shape() # test the editing 
