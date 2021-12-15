import numpy as np 
from PIL import Image
from scipy.io import loadmat,savemat
from array import array
import lib_py.chj_file as chj_file
import torch

# define facemodel for reconstruction
class BFM():
    def __init__(self, fmodel):
        self.fmodel = fmodel

    def to_torch(self, is_torch=True, is_cuda=False):
        self.model = chj_file.load_np_mats(self.fmodel)
        model = self.model[:5]
        if is_torch:
            model=[ torch.from_numpy(x) for x in model ]
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

        # 原来是这样
        # calculate 5 facial landmarks using 68 landmarks
        lm_idx = np.array([31,37,40,43,46,49,55]) - 1
        Lm3D = np.stack([Lm3D[lm_idx[0],:],np.mean(Lm3D[lm_idx[[1,2]],:],0),np.mean(Lm3D[lm_idx[[3,4]],:],0),Lm3D[lm_idx[5],:],Lm3D[lm_idx[6],:]], axis = 0)
        Lm3D = Lm3D[[1,2,0,3,4],:]
        self.Lm3D = Lm3D
        return Lm3D

# load input images and corresponding 5 landmarks
def load_img(img_path,lm_path):

    image = Image.open(img_path)
    lm = np.loadtxt(lm_path)

    return image,lm

# save 3D face to obj file
def save_obj(path,v,f,c):
    with open(path,'w') as file:
        for i in range(len(v)):
            file.write('v %f %f %f %f %f %f\n'%(v[i,0],v[i,1],v[i,2],c[i,0],c[i,1],c[i,2]))

        file.write('\n')

        for i in range(len(f)):
            file.write('f %d %d %d\n'%(f[i,0],f[i,1],f[i,2]))

    file.close()