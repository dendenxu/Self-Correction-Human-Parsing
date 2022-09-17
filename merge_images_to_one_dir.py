import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str) # the root folder, whose images folder will be recursively liked to output
parser.add_argument('output', type=str) # the output folder
args = parser.parse_args()

from termcolor import cprint, colored
from os.path import join

def run(cmd: str):
    print(colored('Run: ', 'blue') + cmd)
    os.system(cmd)

run(f'mkdir -p {args.output}')

# path: should contain multiple humans, images in humans, cameras in images, and actual image in cameras
for human in os.listdir(args.path):
    human_dir = join(args.path, human)
    img_dir = join(human_dir, 'images')
    for cam in os.listdir(img_dir):
        cam_dir = join(img_dir, cam)
        if cam_dir == args.output:
            continue
        for img in os.listdir(cam_dir):
            img_path = join(cam_dir, img)
            new_img = f"{human}.{cam}.{img}" # like F1_06_000000.02.000000.jpg
            new_img_path = join(args.output, new_img)
            run(f'ln -s {img_path} {new_img_path}')

