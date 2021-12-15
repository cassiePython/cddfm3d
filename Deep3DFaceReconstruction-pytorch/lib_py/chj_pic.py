# -*- coding:utf-8 -* 

import re
import numpy 
import numpy as np
import scipy.io as scio
import cv2 as cv
import cv2
from PIL import Image
from numpy import random
import scipy
import matplotlib.pyplot as plt

def ps(dt): print(dt.shape)
def p(*info): print(*info)
def showimg(img,nm="pic",waite=0):
    cv2.imshow(nm,img)
    return cv2.waitKey(waite)
def showImg(img,nm="pic",waite=0):
    cv2.imshow(nm,img)
    return cv2.waitKey(waite)
def drawCircle(img,x,y,color=(0,255,0),size=2):
    for id in range(len(x)):
        cv2.circle(img,(int(x[id]),int(y[id])),1,color, size)

def drawCirclev2(img,xy,color=(0,255,0),size=2):
    drawCircle(img, xy[:,0],xy[:,1], color, size)

def drawRect(img,rect,color=(255,0,0)):
    r=[ int(x) for x in rect ]
    cv2.rectangle(img,(r[0],r[1]), (r[0]+r[2],r[1]+r[3]), color,1)
    
def drawRectXY(img,rect,color=(255,0,0), size=1):
    cv2.rectangle(img,(int(rect[0]),int(rect[1])), (int(rect[2]),int(rect[3])), color,size)    
    
def drawIds(img,x,y,color=(0,0,255)):
    for id in range(len(x)):
        cv2.putText( img, str(id),(int(x[id]+1),int(y[id]-1)), 1,0.5,color, 1);
        
def drawIds_1base(img,x,y,color=(0,0,255)):
    for id in range(len(x)):
        cv2.putText( img, str(id+1),(int(x[id]+1),int(y[id]-1)), 1,0.5,color, 1);

    
true=True
false=False

def readlines(fname):
    with open(fname) as fp:
        list=fp.readlines()

    for id, item in enumerate(list):
        list[id]=item.strip()
    return list

