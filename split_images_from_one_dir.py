import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str) # the root folder, whose images folder will be recursively liked to output
parser.add_argument('output', type=str) # the output folder
parser.add_argument('--mask_dir', default='mask')
args = parser.parse_args()

from termcolor import cprint, colored
from os.path import join

def run(cmd: str):
    print(colored('Run: ', 'blue') + cmd)
    os.system(cmd)


# path: should contain multiple humans, images in humans, cameras in images, and actual image in cameras
for human in os.listdir(args.path):
    human_dir = join(args.path, human)
    img_dir = join(human_dir, 'images') # copy target (split destination)
    msk_dir = join(human_dir, args.mask_dir) # eval results (to be split)
    for cam in os.listdir(img_dir):
        cam_dir = join(img_dir, cam) # reference: original image dir
        for img in os.listdir(cam_dir):
            msk = img.replace('.jpg', '.png')
            new_msk = f"{human}.{cam}.{msk}" # like F1_06_000000.02.000000.jpg
            msk_cam_dir = join(msk_dir, cam)
            msk_path = join(msk_cam_dir, msk) # copy result like (...)F1_06_000000/images/02/000000.png
            new_msk_path = join(args.output, new_msk) # copy from (to be split)
            run(f'mkdir -p {msk_cam_dir}')
            run(f'mv {new_msk_path} {msk_path}')

