# -*- coding:utf-8 -* 
import os ; import sys 
os.chdir( os.path.split( os.path.realpath( sys.argv[0] ) )[0] ) 

from network.resnet50_task import *
from lib_py.chj_pic import * 
import glob
from scipy.io import loadmat,savemat

from preprocess_img import Preprocess
from load_data import *
from reconstruct_mesh import Reconstruction

'''
@19-4-9
all pytorch
I do some changes for for visualization, mostly on `z_buffer`. See function `Projection_layer` and this code.
'''

fdata_dir = "./"

def demo():
    # input and output folder
    image_path = fdata_dir + 'input'
    save_path = fdata_dir + 'output'
    img_list = glob.glob(image_path + '/' + '*.png')

    # read face model
    facemodel = BFM( fdata_dir+"BFM/mSEmTFK68etc.chj" )
    is_cuda= True
    facemodel.to_torch(is_torch=True, is_cuda=is_cuda)
    # read standard landmarks for preprocessing images
    lm3D = facemodel.load_lm3d(fdata_dir+"BFM/similarity_Lm3D_all.mat")
    n = 0

    model = resnet50_use()
    model.load_state_dict(torch.load(fdata_dir+"network/th_model_params.pth"))
    model.eval()

    if is_cuda: model.cuda()

    for param in model.parameters():
        param.requires_grad = False
    
    print('reconstructing...')
    for file in img_list:
        n += 1
        print(n)
        # load images and corresponding 5 facial landmarks
        img,lm = load_img(file,file.replace('png','txt'))

        # preprocess input image
        input_img_org,lm_new,transform_params = Preprocess(img,lm,lm3D)

        input_img = input_img_org.astype(np.float32)
        input_img = torch.from_numpy(input_img).permute(0, 3, 1, 2) 
        # the input_img is BGR

        if is_cuda: input_img = input_img.cuda()
        
        arr_coef = model(input_img)
                
        coef = torch.cat(arr_coef, 1)

        # reconstruct 3D face with output coefficients and face model
        face_shape,face_texture,face_color,tri,face_projection,z_buffer,landmarks_2d = Reconstruction(coef,facemodel)

        # see the landmark
        if 1 == 0:
            input_img_org = input_img_org.squeeze()
            landmarks_2d = landmarks_2d.squeeze()
            img = np.array(input_img_org).copy()
            landmarks_2d[:, 1] = 224 - landmarks_2d[:, 1]
            face2ds = landmarks_2d
            drawCirclev2(img, face2ds)

            key = showimg(img)
            if key == 27: break
            continue

        if is_cuda:
            face_shape = face_shape.cpu()
            face_texture = face_texture.cpu()
            face_color = face_color.cpu()
            face_projection  = face_projection.cpu()
            z_buffer = z_buffer.cpu()
            landmarks_2d = landmarks_2d.cpu()


        #exit()
        # reshape outputs
        input_img = np.squeeze(input_img)
        #shape = np.squeeze(face_shape,[0])
        #color = np.squeeze(face_color,[0])
        #landmarks_2d = np.squeeze(landmarks_2d,[0])
        shape = np.squeeze(face_shape)
        color = np.squeeze(face_color)
        #color = np.squeeze(face_texture)
        landmarks_2d = np.squeeze(landmarks_2d)
        
        # for visualization
        z_buffer -= z_buffer.min()
        z_buffer *= 100 
        #face_projection[:, 1] = 224 - face_projection[:, 1]
        #face_projection *= -1
        face3d_project = np.concatenate((face_projection,z_buffer), axis=2)
        
        # CHJ_INFO: change to show what you want
        shape = np.squeeze(face3d_project)
        #p(face_projection.shape, z_buffer.shape)

        # save output files
        # cropped image, which is the direct input to our R-Net
        # 257 dim output coefficients by R-Net
        # 68 face landmarks of cropped image
        #savemat(os.path.join(save_path,file.split('\\')[-1].replace('.png','.mat')),{'cropped_img':input_img[:,:,::-1],'coeff':coef,'landmarks_2d':landmarks_2d,'lm_5p':lm_new})
        save_obj(os.path.join(save_path,file.split('\\')[-1].replace('.png','_mesh-th.obj')),shape,tri+1,np.clip(color,0,1)) # 3D reconstruction face (in canonical view)
        
        # CHJ_INFO: take care !!!!!
        if n>3: break

# ignore this, but it may usefull for you
def chj_recover_ocv(lm2d, transform_params, is_in_ocv=True):
    if is_in_ocv:
        lm2d[:, 1] = 223 - lm2d[:, 1]
    lm2d = (lm2d-112)/transform_params[2] + transform_params[3:].reshape(1, 2)
    lm2d[:, 1] = transform_params[1] - lm2d[:, 1]
    return lm2d


if __name__ == '__main__':
    demo()
    