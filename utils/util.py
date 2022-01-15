import os
import torch
import numpy as np

def parse_styles(styles_tensor, flatten=False): # styles_tensor torch.Size([batch, 9088])
    N = styles_tensor.shape[0]
    styles = []
    splits = [512*15, 512*15+256*3, 512*15+256*3+128*3, 512*15+256*3+128*3+64*3]
    for i in range(15):
        style = styles_tensor[:, i*512: (i+1) * 512]
        if flatten:
            style = style.reshape(N, 1, 512, 1, 1)
        styles.append(style)
    for i in range(3):
        style = styles_tensor[:, i * 256+splits[0]: (i + 1) * 256+splits[0]]
        if flatten:
            style = style.reshape(N, 1, 256, 1, 1)
        styles.append(style)
    for i in range(3):
        style = styles_tensor[:, i * 128+splits[1]: (i + 1) * 128+splits[1]]
        if flatten:
            style = style.reshape(N, 1, 128, 1, 1)
        styles.append(style)
    for i in range(3):
        style = styles_tensor[:, i * 64+splits[2]: (i + 1) * 64+splits[2]]
        if flatten:
            style = style.reshape(N, 1, 64, 1, 1)
        styles.append(style)
    for i in range(2):
        style = styles_tensor[:, i * 32+splits[3]: (i + 1) * 32+splits[3]]
        if flatten:
            style = style.reshape(N, 1, 32, 1, 1)
        styles.append(style)
    return styles

def parse_param(param):
    # parse param
    """Work for both numpy and tensor
    params:
        param: parameters-the output of the DFRNet shape=(Batch, 257)
    returns:
        alpha_shp: identity(shape) coeff (Batch, 80)
        alpha_exp: expression coeff (Batch, 64)
        albedo: texture(albedo) coeff (Batch, 80)
        angles: ruler angles(x,y,z) for rotation (Batch, 3)
        gamma: lighting coeff for 3 channel SH function (Batch, 27)
        translation: translation coeff (Batch, 3)
    """
    alpha_shp = param[:, :80]  # identity(shape) coeff of dim 80
    alpha_exp = param[:, 80:144]  # expression coeff of dim 64
    albedo = param[:, 144:224]  # texture(albedo) coeff of dim 80
    angles = param[:, 224:227]  # ruler angles(x,y,z) for rotation of dim 3
    gamma = param[:, 227:254]  # lighting coeff for 3 channel SH function of dim 27
    translation = param[:, 254:]  # translation coeff of dim 3
    return (alpha_shp, alpha_exp, albedo, angles, gamma, translation)

# compute face shape with identity and expression coeff, based on BFM model
# input: id_coeff with shape [1,80]
#         ex_coeff with shape [1,64]
# output: face_shape with shape [1,N,3], N is number of vertices
def Shape_formation(id_coeff, ex_coeff, facemodel):
    batch_size = id_coeff.size(0)
    face_shape = torch.einsum('ij,aj->ai', facemodel.idBase, id_coeff) + \
            torch.einsum('ij,aj->ai', facemodel.exBase, ex_coeff) + facemodel.meanshape
    face_shape = face_shape.view(batch_size,-1,3)
    # re-center face shape 
    face_shape = face_shape - torch.mean(facemodel.meanshape.view(1,-1,3), dim=1, keepdims = True) 
    return face_shape 

# compute vertex texture(albedo) with tex_coeff
# input: tex_coeff with shape [1,N,3]
# output: face_texture with shape [1,N,3], RGB order, range from 0-255
def Texture_formation(tex_coeff, facemodel):
    n_b = tex_coeff.size(0)
    face_texture = torch.einsum('ij,aj->ai', facemodel.texBase, tex_coeff) + facemodel.meantex

    face_texture = face_texture.view(n_b, -1, 3)
    return face_texture


# compute vertex normal using one-ring neighborhood
# input: face_shape with shape [1,N,3]
# output: v_norm with shape [1,N,3]
def Compute_norm(face_shape, facemodel):
    batch_size = face_shape.size(0)
    face_id = facemodel.tri  # vertex index for each triangle face, with shape [F,3], F is number of faces 
    face_id = torch.tensor(face_id).long().cuda() 
    #iface_id = (face_id - 1).long() 

    point_id = facemodel.point_buf 
    point_id = torch.tensor(point_id).long().cuda() 
    #point_id = (point_id - 1).long() 

    v1 = face_shape[:,face_id[:,0],:]
    v2 = face_shape[:,face_id[:,1],:]
    v3 = face_shape[:,face_id[:,2],:]

    e1 = v1 - v2 
    e2 = v2 - v3 

    face_norm = torch.cross(e1,e2) # compute normal for each face 
    face_norm = torch.cat([face_norm, torch.zeros([batch_size,1,3]).cuda()], dim=1) # concat face_normal with a zero vector at the end 

    v_norm = torch.sum(face_norm[:,point_id,:], dim=2) # compute vertex normal using one-ring neighborhood 
    v_norm = v_norm/torch.norm(v_norm, p=2, dim=2, keepdim=True) 
    return v_norm 

# compute rotation matrix based on 3 ruler angles
# input: angles with shape [1,3]
# output: rotation matrix with shape [1,3,3]
def Compute_rotation_matrix(angles):
    batch_size = angles.size()[0]

    angle_x = angles[:,0]
    angle_y = angles[:,1] 
    angle_z = angles[:,2]

    cosx = torch.cos(angle_x); sinx = torch.sin(angle_x) 
    cosy = torch.cos(angle_y); siny = torch.sin(angle_y) 
    cosz = torch.cos(angle_z); sinz = torch.sin(angle_z) 

    rotation_x = torch.eye(3).repeat(batch_size,1).view(batch_size, 3, 3).cuda() 
    rotation_x[:, 1, 1] = cosx; rotation_x[:, 1, 2] = - sinx 
    rotation_x[:, 2, 1] = sinx; rotation_x[:, 2, 2] = cosx 

    rotation_y = torch.eye(3).repeat(batch_size,1).view(batch_size, 3, 3).cuda() 
    rotation_y[:, 0, 0] = cosy; rotation_y[:, 0, 2] = siny 
    rotation_y[:, 2, 0] = -siny; rotation_y[:, 2, 2] = cosy

    rotation_z = torch.eye(3).repeat(batch_size,1).view(batch_size, 3, 3).cuda() 
    rotation_z[:,0,0] = cosz; rotation_z[:, 0, 1] = - sinz 
    rotation_z[:,1,0] = sinz; rotation_z[:, 1, 1] = cosz 

    # compute rotation matrix for X,Y,Z axis respectively 
    rotation = torch.matmul(torch.matmul(rotation_z,rotation_y),rotation_x) 
    rotation = rotation.permute(0,2,1)#transpose row and column (dimension 1 and 2) 
    return rotation 


# compute vertex color using face_texture and SH function lighting approximation
# input: face_texture with shape [1,N,3]
#          norm with shape [1,N,3]
#         gamma with shape [1,27]
# output: face_color with shape [1,N,3], RGB order, range from 0-255
#          lighting with shape [1,N,3], color under uniform texture
def Illumination_layer(face_texture,norm,gamma):
    batch_size = gamma.size()[0]
    num_vertex = face_texture.size()[1] 
    init_lit = torch.tensor([0.8,0,0,0,0,0,0,0,0]).cuda() 
    gamma = gamma.view(-1,3,9) 
    gamma = gamma + init_lit.view(1,1,9) 

    # parameter of 9 SH function 
    pi = 3.1415927410125732 
    a0 = pi 
    a1 = 2 * pi/torch.sqrt(torch.tensor(3.0)).cuda() 
    a2 = 2 * pi/torch.sqrt(torch.tensor(8.0)).cuda() 
    c0 = 1 / torch.sqrt(torch.tensor(4 * pi)).cuda()
    c1 = torch.sqrt(torch.tensor(3.0)).cuda() / torch.sqrt(torch.tensor(4*pi))
    c2 = 3 * torch.sqrt(torch.tensor(5.0)).cuda() / torch.sqrt(torch.tensor(12*pi))

    Y0 = (a0*c0).view(1,1,1).repeat(batch_size, num_vertex,1)
    Y1 = (-a1 * c1 * norm[:,:,1]).view(batch_size, num_vertex,1) 
    Y2 = (a1 * c1 * norm[:,:,2]).view(batch_size,num_vertex,1)
    Y3 = (-a1 * c1 * norm[:,:,0]).view(batch_size,num_vertex,1)
    Y4 = (a2 * c2 * norm[:,:,0] * norm[:,:,1]).view(batch_size,num_vertex,1) 
    Y5 = (-a2 * c2 * norm[:,:,1] * norm[:,:,2]).view(batch_size,num_vertex,1)
    Y6 = (a2 * c2 * 0.5 / torch.sqrt(torch.tensor(3.0)) * (3* norm[:,:,2] ** 2 - 1)).view(batch_size,num_vertex,1)
    Y7 = (-a2 * c2 * norm[:,:,0] * norm[:,:,2]).view(batch_size,num_vertex,1)
    Y8 = (a2  * c2 * 0.5 * (norm[:,:,0] ** 2 - norm[:,:,1] ** 2)).view(batch_size,num_vertex,1)

    Y = torch.cat([Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8],dim=2)

    # Y shape:[batch,N,9].
    lit_r = torch.squeeze(torch.matmul(Y, torch.unsqueeze(gamma[:,0,:],2)),2) #[batch,N,9] * [batch,9,1] = [batch,N]
    lit_g = torch.squeeze(torch.matmul(Y, torch.unsqueeze(gamma[:,1,:],2)),2)
    lit_b = torch.squeeze(torch.matmul(Y, torch.unsqueeze(gamma[:,2,:],2)),2)

    # shape:[batch,N,3]
    face_color = torch.stack([lit_r*face_texture[:,:,0],lit_g*face_texture[:,:,1],lit_b*face_texture[:,:,2]],axis = 2)
    lighting = torch.stack([lit_r,lit_g,lit_b],axis = 2)*128
    return face_color, lighting 


# project 3D face onto image plane
# input: face_shape with shape [1,N,3]
#          rotation with shape [1,3,3]
#         translation with shape [1,3]
# output: face_projection with shape [1,N,2]
#           z_buffer with shape [1,N,1]
def Projection_layer(face_shape, rotation, translation):  # we choose the focal length and camera position empirically
    face_shape_r = torch.matmul(face_shape, rotation) 
    face_shape_t = face_shape_r + translation.view(-1, 1, 3).repeat(1, face_shape_r.size()[1], 1)

    return face_shape_t 

def recon_transform(param, bfm, if_light=True, if_edge=False): 
    """compute projection on image plane
    params:
        param: parameters-the output of the DFRNet shape=(Batch, 257)
        bfm: BFM model
    returns:
        face3d_project: reconstructed vertex Tensor(Batch, N, 3)
        face_texture: albedo texture  Tensor(Batch, N, 3)
        triangle: triangles Tensor(Batch, N2, 3)
        landmarks_2d: 68 landmark on image plane (Batch, 68, 2)
    """
    id_coeff, ex_coeff, tex_coeff, angles, gamma, translation = parse_param(param)

    ############################  used to test render loss 2020-10-18 #######################
    #id_coeff = id_coeff.detach()
    #ex_coeff = ex_coeff.detach()
    #angles = angles.detach()
    #gamma = gamma.detach()
    #translation = translation.detach()
    ########################################################################################
    # compute face shape
    face_shape = Shape_formation(id_coeff, ex_coeff, bfm)
    # compute vertex texture(albedo)
    face_texture = Texture_formation(tex_coeff, bfm)

    # vertex normal
    face_norm = Compute_norm(face_shape, bfm) 

    # rotation matrix
    rotation = Compute_rotation_matrix(angles)
    face_norm_r = torch.matmul(face_norm, rotation) 

    face_shape_t = Projection_layer(face_shape, rotation, translation)

    # compute vertex color using SH function lighting approximation 
    if if_light:
        face_texture, lighting = Illumination_layer(face_texture, face_norm_r, gamma) 

    # vertex index for each face of BFM model
    triangle = bfm.tri

    if if_edge:
        return face_shape_t, triangle, face_texture, face_shape 
        

    return face_shape_t, triangle, face_texture 


def get_mask_from_render(render_img):  # (Batch, img_size, img_size, 1) (0,1)
    image = render_img.detach().cpu().numpy()
    image = (image * 255).astype(np.int32)
    idx = image > 0
    mask = np.zeros(shape=image.shape)
    mask[idx] = 1.0
    return torch.from_numpy(mask).float()

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


__mp_dtype_r={
    0: np.int32, 1: np.float32, 2: np.int64, 3: np.float64, 4: np.bool, 5: np.uint8
}


def load_np_mat(fp):
    #if type(fp)==str: fp=open(fp,"rb")
    dim = np.fromfile(fp, np.int32, 1)[0]
    dims = np.fromfile(fp, np.int32, dim)
    type_id=np.fromfile(fp, np.int32, 1)[0]
    dtype=__mp_dtype_r[type_id]
    mt = np.fromfile(fp, dtype, dims.prod())
    mt = mt.reshape(dims)
    #print("load info: ", dim, dims, " | ", mt.shape," | ", mt.dtype)
    return mt

def load_np_mats(fp):
    if type(fp) == str:
        with open(fp, "rb") as fp:
            mats_num = np.fromfile(fp, np.int32, 1)[0]
            mts=[]
            for i in range(mats_num):
                mts.append( load_np_mat(fp) )
    return mts

