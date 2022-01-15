import dlib
import os
from skimage import io
import dill
from tqdm import tqdm
import numpy as np

dlib_landmark_model = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(dlib_landmark_model)
detector = dlib.get_frontal_face_detector()

img_dir = "Images"
imgs = os.listdir(img_dir)
print ("Image Count:", len(imgs))


outputs = {}
for path in tqdm(imgs):
    img_path = os.path.join(img_dir, path)
    img = io.imread(img_path)
    dets = detector(img, 0)
    if list(dets) == []:
        continue
    pts = predictor(img, dets[0]).parts()
    pts = np.array([[pt.x, pt.y] for pt in pts]).T 
    # print (pts.shape)
    key = path.split(".")[0]
    outputs[key] = pts
res = {'outputs':outputs}
save_path = 'landmarks.pkl'
with open(save_path, 'wb') as fout:
    dill.dump(res, fout)
