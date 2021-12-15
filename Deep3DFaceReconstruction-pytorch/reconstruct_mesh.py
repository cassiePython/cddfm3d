import torch
# 19-4-12  pytorch

# input: coeff with shape [1,257]
def Split_coeff(coeff):
    id_coeff = coeff[:,:80] # identity(shape) coeff of dim 80
    ex_coeff = coeff[:,80:144] # expression coeff of dim 64
    tex_coeff = coeff[:,144:224] # texture(albedo) coeff of dim 80
    angles = coeff[:,224:227] # ruler angles(x,y,z) for rotation of dim 3
    gamma = coeff[:,227:254] # lighting coeff for 3 channel SH function of dim 27
    translation = coeff[:,254:] # translation coeff of dim 3

    return id_coeff,ex_coeff,tex_coeff,angles,gamma,translation

# CHJ_ADD
class _need_const:
    import numpy as np
    a0 = np.pi
    a1 = 2 * np.pi / np.sqrt(3.0)
    a2 = 2 * np.pi / np.sqrt(8.0)
    c0 = 1 / np.sqrt(4 * np.pi)
    c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
    c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)
    d0 = 0.5/ np.sqrt(3.0)

    illu_consts=[a0, a1, a2, c0, c1, c2, d0]

    focal = 1015.0
    center = 112.0
    cam_pos = 10
    p_matrix = np.concatenate([[focal], [0.0], [center], [0.0], [focal], [center], [0.0], [0.0], [1.0]],
                              axis=0).astype(np.float32)  # projection matrix
    p_matrix = np.reshape(p_matrix, [1, 3, 3])
    p_matrix = torch.from_numpy(p_matrix)
    gpu_p_matrix = None


# compute face shape with identity and expression coeff, based on BFM model
# input: id_coeff with shape [1,80]
#         ex_coeff with shape [1,64]
# output: face_shape with shape [1,N,3], N is number of vertices
def Shape_formation(id_coeff,ex_coeff,facemodel):
    n_b = id_coeff.size(0)
    face_shape = torch.einsum('ij,aj->ai',facemodel.idBase,id_coeff) + \
                torch.einsum('ij,aj->ai',facemodel.exBase,ex_coeff) + \
                facemodel.meanshape

    face_shape = face_shape.view(n_b,-1,3)
    # re-center face shape
    face_shape = face_shape - facemodel.meanshape.view(1, -1, 3).mean(dim=1, keepdim=True)

    return face_shape

# compute vertex texture(albedo) with tex_coeff
# input: tex_coeff with shape [1,N,3]
# output: face_texture with shape [1,N,3], RGB order, range from 0-255
def Texture_formation(tex_coeff,facemodel):
    n_b = tex_coeff.size(0)
    face_texture = torch.einsum('ij,aj->ai',facemodel.texBase,tex_coeff) + facemodel.meantex

    face_texture = face_texture.view(n_b,-1,3)
    return face_texture

# compute vertex normal using one-ring neighborhood
# input: face_shape with shape [1,N,3]
# output: v_norm with shape [1,N,3]
def Compute_norm(face_shape,facemodel):

    face_id = facemodel.tri # vertex index for each triangle face, with shape [F,3], F is number of faces
    point_id = facemodel.point_buf # adjacent face index for each vertex, with shape [N,8], N is number of vertex
    shape = face_shape
    v1 = shape[:,face_id[:,0],:]
    v2 = shape[:,face_id[:,1],:]
    v3 = shape[:,face_id[:,2],:]
    e1 = v1 - v2
    e2 = v2 - v3
    face_norm = e1.cross(e2) # compute normal for each face
    empty = torch.zeros((face_norm.size(0), 1, 3), dtype=face_norm.dtype, device=face_norm.device)

    face_norm = torch.cat((face_norm, empty), 1) # concat face_normal with a zero vector at the end

    v_norm = face_norm[:,point_id,:].sum(2) # compute vertex normal using one-ring neighborhood 
    # CHJ: not average, directly normalize
    v_norm = v_norm/v_norm.norm(dim=2).unsqueeze(2) # normalize normal vectors

    return v_norm


# compute rotation matrix based on 3 ruler angles
# input: angles with shape [1,3]
# output: rotation matrix with shape [1,3,3]
def Compute_rotation_matrix(angles):

    n_b = angles.size(0)
    sinx = torch.sin(angles[:, 0])
    siny = torch.sin(angles[:, 1])
    sinz = torch.sin(angles[:, 2])
    cosx = torch.cos(angles[:, 0])
    cosy = torch.cos(angles[:, 1])
    cosz = torch.cos(angles[:, 2])

    rotXYZ = torch.eye(3).view(1, 3, 3).repeat(n_b * 3, 1, 1).view(3, n_b, 3, 3)

    if angles.is_cuda: rotXYZ = rotXYZ.cuda()

    rotXYZ[0, :, 1, 1] = cosx
    rotXYZ[0, :, 1, 2] = -sinx
    rotXYZ[0, :, 2, 1] = sinx
    rotXYZ[0, :, 2, 2] = cosx
    rotXYZ[1, :, 0, 0] = cosy
    rotXYZ[1, :, 0, 2] = siny
    rotXYZ[1, :, 2, 0] = -siny
    rotXYZ[1, :, 2, 2] = cosy
    rotXYZ[2, :, 0, 0] = cosz
    rotXYZ[2, :, 0, 1] = -sinz
    rotXYZ[2, :, 1, 0] = sinz
    rotXYZ[2, :, 1, 1] = cosz

    rotation = rotXYZ[2].bmm(rotXYZ[1]).bmm(rotXYZ[0])

    return rotation.permute(0, 2, 1)

# project 3D face onto image plane
# input: face_shape with shape [1,N,3]
#          rotation with shape [1,3,3]
#         translation with shape [1,3]
# output: face_projection with shape [1,N,2]
#           z_buffer with shape [1,N,1]
def Projection_layer(face_shape,rotation,translation,focal=1015.0,center=112.0): # we choose the focal length and camera position empirically

    n_b, nV, _ = face_shape.size()
    if face_shape.is_cuda:
        if _need_const.gpu_p_matrix is None:
            _need_const.gpu_p_matrix = _need_const.p_matrix.cuda()

        p_matrix = _need_const.gpu_p_matrix.expand(n_b, 3, 3)
    else:
        p_matrix = _need_const.p_matrix.expand(n_b, 3, 3)

    face_shape_r = face_shape.bmm(rotation)  # CHJ: R has been transposed
    face_shape_t = face_shape_r + translation.view(n_b, 1, 3)

    face_shape_t[:, :, 2] = _need_const.cam_pos - face_shape_t[:, :, 2]

    aug_projection = face_shape_t.bmm(p_matrix.permute(0, 2, 1))

    #print(aug_projection)
    #exit()

    face_projection = aug_projection[:, :, 0:2] / aug_projection[:, :, 2:]
    
    # CHJ_WARN: I do this for visualization
    z_buffer = _need_const.cam_pos - aug_projection[:, :, 2:] # CHJ: same as the z of  face_shape_t

    return face_projection, z_buffer

# CHJ: It's different from what I knew.
# compute vertex color using face_texture and SH function lighting approximation
# input: face_texture with shape [1,N,3]
#          norm with shape [1,N,3]
#         gamma with shape [1,27]
# output: face_color with shape [1,N,3], RGB order, range from 0-255
#          lighting with shape [1,N,3], color under uniform texture

def Illumination_layer(face_texture, norm, gamma):

    n_b, num_vertex, _ = face_texture.size()
    n_v_full = n_b * num_vertex
    gamma = gamma.view(-1, 3, 9).clone() 
    gamma[:, :, 0] += 0.8

    gamma = gamma.permute(0, 2, 1)

    a0, a1, a2, c0, c1, c2, d0 = _need_const.illu_consts

    Y0 = torch.ones(n_v_full).float() * a0*c0
    if gamma.is_cuda: Y0=Y0.cuda()
    norm = norm.view(-1, 3)
    nx, ny, nz = norm[:,0], norm[:,1], norm[:,2]
    arrH = []

    arrH.append( Y0 )
    arrH.append(-a1*c1*ny)
    arrH.append(a1*c1*nz)
    arrH.append(-a1*c1*nx)
    arrH.append(a2*c2*nx*ny)
    arrH.append(-a2*c2*ny*nz)
    arrH.append(a2*c2*d0*(3*nz.pow(2)-1))
    arrH.append(-a2*c2*nx*nz)
    arrH.append(a2*c2*0.5*(nx.pow(2)-ny.pow(2)))

    H = torch.stack(arrH, 1)
    Y = H.view(n_b, num_vertex, 9)

    # Y shape:[batch,N,9].

    # shape:[batch,N,3]
    lighting = Y.bmm(gamma)

    face_color = face_texture * lighting
    #lighting *= 128

    #print( face_color[0, 5] )
    return face_color,lighting

# face reconstruction with coeff and BFM model
def Reconstruction(coeff,facemodel):
    id_coeff,ex_coeff,tex_coeff,angles,gamma,translation = Split_coeff(coeff)

    # compute face shape
    face_shape = Shape_formation(id_coeff, ex_coeff, facemodel)
    # compute vertex texture(albedo)
    face_texture = Texture_formation(tex_coeff, facemodel)
    #print(face_texture[0, 2])
    # vertex normal
    face_norm = Compute_norm(face_shape,facemodel)
    # rotation matrix
    rotation = Compute_rotation_matrix(angles)
    face_norm_r = face_norm.bmm(rotation)

    # compute vertex projection on image plane (with image sized 224*224)
    face_projection,z_buffer = Projection_layer(face_shape,rotation,translation)
    #face_projection = torch.stack((face_projection[:,:,0], 224 - face_projection[:,:,1]), 2)

    # compute 68 landmark on image plane
    landmarks_2d = face_projection[:,facemodel.keypoints,:]

    #print(face_projection)
    #print(z_buffer)
    #print(landmarks_2d)
    #exit()

    # compute vertex color using SH function lighting approximation
    face_color,lighting = Illumination_layer(face_texture, face_norm_r, gamma)

    # vertex index for each face of BFM model
    tri = facemodel.tri

    return face_shape,face_texture,face_color,tri,face_projection,z_buffer,landmarks_2d



# def Reconstruction_for_render(coeff,facemodel):
#     id_coeff,ex_coeff,tex_coeff,angles,gamma,translation = Split_coeff(coeff)
#     face_shape = Shape_formation(id_coeff, ex_coeff, facemodel)
#     face_texture = Texture_formation(tex_coeff, facemodel)
#     face_norm = Compute_norm(face_shape,facemodel)
#     rotation = Compute_rotation_matrix(angles)
#     face_shape_r = np.matmul(face_shape,rotation)
#     face_shape_r = face_shape_r + np.reshape(translation,[1,1,3])
#     face_norm_r = np.matmul(face_norm,rotation)
#     face_color,lighting = Illumination_layer(face_texture, face_norm_r, gamma)
#     tri = facemodel.face_buf

#     return face_shape_r,face_norm_r,face_color,tri