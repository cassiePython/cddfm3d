from network.resnet50_task import *
import glob
from preprocess_img import Preprocess
from load_data import *
from align.detector import detect_faces
from tqdm import tqdm
import dill
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 3DMM parameters and landmarks")
    parser.add_argument("path", type=str, help="path to the image dataset")
    args = parser.parse_args()
    fdata_dir = "./"

    image_dir = args.path #"/home/wangcan/work/cddfm3d/stylegan2-pytorch/Images_40000"
    img_list = glob.glob(image_dir + '/' + '*.png')

    print ("Total Imgs:", len(img_list))
    print ("Test sample:", img_list[0])

    # read face model
    facemodel = BFM( fdata_dir+"BFM/mSEmTFK68etc.chj" )
    is_cuda= True
    facemodel.to_torch(is_torch=True, is_cuda=is_cuda)
    # read standard landmarks for preprocessing images
    lm3D = facemodel.load_lm3d(fdata_dir+"BFM/similarity_Lm3D_all.mat")

    model = resnet50_use()
    model.load_state_dict(torch.load(fdata_dir+"network/th_model_params.pth"))
    model.eval()
    if is_cuda: model.cuda()
    for param in model.parameters():
        param.requires_grad = False

    count = 0
    params = {}
    landmarks = {}
    boxes = {}

    print ("Start to extract 3DMM parameters and landmarks ...")
    for file in tqdm(img_list):
        key = file.split("/")[-1][:-4]
        img = Image.open(file)
        try:
            bounding_boxes, lm = detect_faces(img)
            if lm.shape[0] >= 2:
                bounding_boxes = bounding_boxes[0]
                lm = lm[0]
            lm = lm.reshape((2,5)).T
        except:
            print ("fail to detect face:", key)
            count += 1
            continue
        input_img_org, lm_new, transform_params, _, _ = Preprocess(img,lm,lm3D)
        input_img = input_img_org.astype(np.float32)
        input_img = torch.from_numpy(input_img).permute(0, 3, 1, 2)
        if is_cuda: input_img = input_img.cuda()
        arr_coef = model(input_img)
        coef = torch.cat(arr_coef, 1).squeeze(0) # (257)
        coef = coef.cpu().numpy()
        params[key] = coef
        landmarks[key] = lm
        boxes[key] = bounding_boxes

    print ("%d faces cannot be detected for this dataset" % count)

    save_path = "params.pkl"
    print ("save outputs as 'params.pkl' ...")
    res = {'params': params, 'landmarks': landmarks, 'boxes': boxes}
    with open(save_path, 'wb') as fout:
        dill.dump(res, fout)
    print ("finish all!")