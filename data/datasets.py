import os.path
import os
import torchvision.transforms as transforms
from data.dataset_factory import DatasetBase
from PIL import Image
import random
import numpy as np
import dill
from bfm.bfm import BFM


class StyleGAN2Dataset(DatasetBase):
    def __init__(self, opt, is_for_train):
        super(StyleGAN2Dataset, self).__init__(opt, is_for_train)
        self._name = 'StyleGAN2Dataset'

        # read dataset
        self._read_dataset_paths()

        # init 3DMM
        self.facemodel = BFM("bfm/BFM/mSEmTFK68etc.chj")
        self.lm3D = self.facemodel.load_lm3d("bfm/BFM/similarity_Lm3D_all.mat") 

    def __getitem__(self, index):
        assert (index < self._dataset_size)

        if not self._opt.serial_batches:
            index = random.randint(0, self._dataset_size - 1)

        # get sample data
        sample_id = self._ids[index]
        latent = self._get_latent_by_id(sample_id)
        param = self._get_param_by_id(sample_id)

        sample = {
                  'latent': latent,
                  'sample_id': sample_id,
                  'param': param,
                }

        return sample

    def __len__(self):
        return self._dataset_size

    def _read_dataset_paths(self):
        self._root = self._opt.data_dir
        self._imgs_dir = os.path.join(self._root, self._opt.image_dir)
        self._params = os.path.join(self._root, self._opt.params_path)
        self._landmark_path = os.path.join(self._root, self._opt.landmarks_path)
        self._latents_path = os.path.join(self._root, self._opt.latents_path)

        # read ids
        use_ids_filename = self._opt.train_list if self._is_for_train else self._opt.test_list
        use_ids_filepath = os.path.join(self._root, use_ids_filename)
        self._ids = self._read_ids(use_ids_filepath)

        # read latents
        self._latents = self._read_latents(self._latents_path)
        self._landmarks = self._read_landmarks(self._landmark_path)
        # read params and rois
        self._params, self._keypoints = self._read_params_keypoints(self._params)

        self._ids = list(set((self._ids)).intersection(set(self._latents.keys())).intersection(set(self._landmarks.keys())).intersection(set(self._params.keys())))

        # dataset size
        self._dataset_size = len(self._ids)

    def _create_transform(self):
        if self._is_for_train:
            transform_list = [
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std=[0.5, 0.5, 0.5]),
                               ]
        else:
            transform_list = [
                               transforms.ToTensor(),
                              ]
        self._transform = transforms.Compose(transform_list)

    def _read_ids(self, file_path):
        ids = np.loadtxt(file_path, delimiter='\t', dtype=np.str)
        return [id[:-4] for id in ids]

    def _read_latents(self, file_path):
        with open(file_path, 'rb') as fin:
            res = dill.load(fin) 
            latents = res['latents']
        return latents

    def _read_params_keypoints(self, file_path):
        with open(file_path, 'rb') as fin:
            res = dill.load(fin)
            outputs, keypoints = res['params'], res['landmarks']
        return outputs, keypoints # 5 landmarks used to align images

    def _read_landmarks(self, file_path):
        with open(file_path, 'rb') as fin:
            res = dill.load(fin)
            outputs = res['outputs'] # 68 landmarks used for landmakrs loss
        return outputs

    def _get_landmark_by_id(self, id):
        if id in self._landmarks:
            return self._landmarks[id]
        else:
            return None 

    def _get_latent_by_id(self, id):
        if id in self._latents:
            return self._latents[id]
        else:
            return None

    def _get_param_by_id(self, id):
        if id in self._params:
            return self._params[id]
        else:
            return None

    def _get_img_by_id(self, id):
        filepath = os.path.join(self._imgs_dir, id+'.png')
        img = Image.open(filepath)  # RGB image
        return img, filepath
