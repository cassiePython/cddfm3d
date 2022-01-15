import numpy as np
from scipy.io import loadmat
from utils.util import load_np_mats
import torch

# define facemodel for reconstruction
class BFM:
    def __init__(self, bfm_fp, is_torch=True, is_cuda=True):
        self.model = load_np_mats(bfm_fp)
        model = self.model[:5]
        if is_torch:
            model = [torch.from_numpy(x) for x in model]
        if is_cuda and is_torch:
            model = [x.cuda() for x in model]
        self.meanshape = model[0]  # mean face shape
        self.idBase = model[1]  # identity basis
        self.exBase = model[2]  # expression basis
        self.meantex = model[3]  # mean face texture
        self.texBase = model[4]  # texture basis
        self.tri = self.model[5]
        self.keypoints = self.model[6]
        self.point_buf = self.model[9]

    # load landmarks for standard face, which is used for image preprocessing
    def load_lm3d(self, fsimilarity_Lm3D_all_mat):

        Lm3D = loadmat(fsimilarity_Lm3D_all_mat)
        Lm3D = Lm3D['lm']

        # calculate 5 facial landmarks
        lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
        Lm3D = np.stack([Lm3D[lm_idx[0],:], np.mean(Lm3D[lm_idx[[1,2]],:],0), np.mean(Lm3D[lm_idx[[3,4]],:],0), Lm3D[lm_idx[5],:], Lm3D[lm_idx[6],:]], axis = 0)
        Lm3D = Lm3D[[1,2,0,3,4],:]
        self.Lm3D = Lm3D
        return Lm3D
