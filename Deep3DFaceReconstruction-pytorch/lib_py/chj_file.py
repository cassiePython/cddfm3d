# -*- coding:utf-8 -* 

'''
first 2018
'''

import codecs
import os
import numpy as np
import struct
import subprocess
import pickle

__mp_dtype={
    'int32': 0, 'float32':1, 'int64': 2, 'float64':3, 'bool':4, 'uint8':5
}
__mp_dtype_r={
    0:np.int32, 1:np.float32, 2:np.int64, 3:np.float64, 4:np.bool, 5:np.uint8
}

# 18-3-8 多次试验总结出来的    
def save_np_mat(fp, mt):
    dim=len(mt.shape)
    dims_and_type=np.array( [dim]+list(mt.shape)+[ __mp_dtype[str(mt.dtype)] ], np.int32)
    dims_and_type.tofile(fp)
    mt.tofile(fp)
    
    #print("save info: ", [dim]+list(mt.shape))
    
def save_np_mats(fp, mts):
    if type(fp) == str: fp = open(fp, "wb")
    mats_num=np.array( [len(mts)],  np.int32)
    mats_num.tofile(fp)
    for mt in mts: save_np_mat(fp, mt)

def load_np_mat(fp):
    #if type(fp)==str: fp=open(fp,"rb")
    dim = np.fromfile(fp, np.int32, 1)[0]
    dims = np.fromfile(fp, np.int32, dim)
    type_id=np.fromfile(fp, np.int32, 1)[0]
    dtype=__mp_dtype_r[type_id]
    mt = np.fromfile(fp, dtype, dims.prod())
    mt=mt.reshape(dims)
    #print("load info: ", dim, dims, " | ", mt.shape," | ", mt.dtype)
    return mt
    
def load_np_mats(fp):
    if type(fp)==str: fp=open(fp,"rb")
    mats_num = np.fromfile(fp, np.int32, 1)[0]
    mts=[]
    for i in range(mats_num): mts.append( load_np_mat(fp) )
    return mts
     
