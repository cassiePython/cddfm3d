import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import os

"""
Generate training pairs: (w, I_w):
Each training sample is generated by combining up
to 5 separately sampled latent vectors, similar to the mixing
regularizer in StyleGAN. We use the latent code in StyleSpace.
"""

def generate(args, g_ema, device, mean_latent):
    with torch.no_grad():
        g_ema.eval()

        for i in tqdm(range(args.pics)):
            sample_zs = []
            for _ in range(args.counts):
                sample_z = torch.randn(args.sample, args.latent, device=device)
                sample_zs.append(sample_z)

            sample, latent, constant, styles = g_ema(sample_zs, truncation=args.truncation, truncation_latent=mean_latent, return_all=True)
            #print ("styles size in StyleSpace: ", styles.shape)

            os.makedirs("Images", exist_ok=True)
            os.makedirs("Latents", exist_ok=True)
            os.makedirs("Constants", exist_ok=True)


            utils.save_image(
                sample,
                f'Images/{str(i).zfill(8)}.png',
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

            np.save(f'Latents/{str(i).zfill(8)}.npy', styles.cpu().numpy())
            np.save(f'Constants/{str(i).zfill(8)}.npy', constant.cpu().numpy())
            #np.save(f'noise/noise_{str(i).zfill(6)}.npy', np.array(noise)) #Use None

def merge(read_dir, save_name, save_key):
    import dill

    latents = os.listdir(read_dir)
    results = {}

    for latent in latents:
        key = latent.split(".")[0]
        path = os.path.join(read_dir, latent)
        data = np.load(path)
        data = data.flatten()
        results[key] = data

    with open(save_name, 'wb') as fw:
        res = {save_key: results}
        dill.dump(res, fw)


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--pics', type=int, default=20)
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt', type=str, default="stylegan2-ffhq-config-f.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--counts', type=int, default=5)


    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt, map_location='cuda:0')

    g_ema.load_state_dict(checkpoint['g_ema'])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    print ("start generation ...")

    generate(args, g_ema, device, mean_latent)

    # merge all latents/constants into a single file
    print ("merge all latents/constants into a single file ...")
    merge("Latents", "latents.pkl", 'latents')
    merge("Constants", "constants.pkl", 'constants')
    print ("finish all!")
