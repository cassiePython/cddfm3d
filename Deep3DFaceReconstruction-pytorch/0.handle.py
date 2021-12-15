# -*- coding:utf-8 -* 
import os ; import sys 
os.chdir( os.path.split( os.path.realpath( sys.argv[0] ) )[0] ) 

from lib_py import chj_file
from lib_py.chj_pic import *

import torch

# CHJ_WARN: you must get this file first, use the original script.
# because the model needs to be applied for
forg_model = "BFM/BFM_model_front.mat"
fmodel = "BFM/mSEmTFK68etc.chj"
fmodel_nms = "BFM/mSEmTFK68etc.nms"

def main():
    f1()
    pass


'''
meanshape (1, 107127) float32 True
meantex (1, 107127) float32 True
idBase (107127, 80) float32 False
exBase (107127, 64) float64 False
1.8620863531659815e-09
texBase (107127, 80) float32 False
tri (70789, 3) float32 False
point_buf (35709, 8) float32 False  # 是每个点周围的三角形
tri_mask2 (54681, 3) float32 False   # 27660 points
keypoints (1, 68) float32 True
frontmask2_idx (27660, 1) float32 True
skinmask (1, 35709) float32 True
'''    
def f1():    
    org_model = scio.loadmat(forg_model)
    
    nms=["meanshape", "idBase", "exBase", 
        "meantex", "texBase", "tri", 
        "keypoints", "frontmask2_idx", "tri_mask2", 
        "point_buf", "skinmask"]
        
    mp={}
    for k, v in org_model.items():
        #p(k, type(v))
        if type(v) == np.ndarray:
            #p(k, v.shape, v.dtype, v.flags.c_contiguous)
            if k=="exBase": 
                _v = v
                v=v.astype(np.float32)
                # p( np.abs(_v -v).max() ) # small
            v = change_value(v) # need
            mp[k]=np.squeeze(v)
    
    # change to np.int32
    for k in nms[5:-1]: mp[k]=mp[k].astype(np.int32)-1
    mp[nms[-1]]=mp[nms[-1]].astype(np.uint8)
    
    # fix T to 0-1
    mp[ nms[3] ] /= 255
    mp[ nms[4] ] /= 255
    
    model=[ mp[x] for x in nms ]
    chj_file.save_np_mats(fmodel, model)
    with open(fmodel_nms, "w") as fp:
        for x in nms: fp.write(x+"\n")
        
       
def change_value(mat):
    if mat.flags.c_contiguous == False:
        return np.ascontiguousarray(mat)
    
    return mat
    

if __name__ == '__main__': 
    main() 
