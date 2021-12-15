from PIL import Image
from .detector import detect_faces
from .align_trans import get_reference_facial_points, warp_and_crop_face
import numpy as np
import os
from tqdm import tqdm


def align(image_or_folder, crop_size=112, save_path=None, vis=True):  # 传入已读的图像
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square=True) * scale

    if save_path is not None and not os.path.isdir(save_path) :
        os.makedirs(save_path)

    if os.path.isdir(image_or_folder):
        for subfolder in tqdm(os.listdir(image_or_folder)):
            if not os.path.isdir(os.path.join(save_path, subfolder)):
                os.makedirs(os.path.join(save_path, subfolder))
            for image_name in os.listdir(os.path.join(image_or_folder, subfolder)):
                print("Processing\t{}".format(os.path.join(image_or_folder, subfolder, image_name)))
                img = Image.open(os.path.join(image_or_folder, subfolder, image_name))
                try: # Handle exception
                    _, landmarks = detect_faces(img)
                except Exception:
                    print("{} is discarded due to exception!".format(os.path.join(image_or_folder, subfolder, image_name)))
                    continue
                if len(landmarks) == 0:  # If the landmarks cannot be detected, the img will be discarded
                    print("{} is discarded due to non-detected landmarks!".format(os.path.join(image_or_folder, subfolder, image_name)))
                    continue
                facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
                warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
                img_warped = Image.fromarray(warped_face)
                if image_name.split('.')[-1].lower() not in ['jpg', 'jpeg']:  # not from jpg
                    image_name = '.'.join(image_name.split('.')[:-1]) + '.jpg'
                img_warped.save(os.path.join(save_path, subfolder, image_name))
    else:
        img = Image.open(image_or_folder)
        try:  # Handle exception
            _, landmarks = detect_faces(img)
        except Exception:
            print("{} is discarded due to exception!".format(image_or_folder))
        if len(landmarks) == 0:  # If the landmarks cannot be detected, the img will be discarded
            print("{} is discarded due to non-detected landmarks!".format(image_or_folder))
        facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
        img_warped = Image.fromarray(warped_face)
        if vis:
            img_warped.show()
        if save_path is not None:
            if image_or_folder.split('.')[-1].lower() not in ['jpg', 'jpeg']:  # not from jpg
                image_or_folder = '.'.join(os.path.basename(image_or_folder).split('.')[:-1]) + '.jpg'
            else:
                image_or_folder = os.path.basename(image_or_folder)

            img_warped.save(os.path.join(save_path, image_or_folder))
        return img_warped
