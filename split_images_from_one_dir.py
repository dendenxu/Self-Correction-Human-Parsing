import os
import sys
import argparse
from os.path import join
from termcolor import cprint, colored


def run(cmd):
    func = sys._getframe(1).f_code.co_name
    print(colored(func, 'yellow')+": "+colored(cmd, 'blue'))
    code = os.system(cmd)
    if code != 0:
        raise RuntimeError(colored(str(code), 'red')+" <- "+colored(func, 'yellow')+": "+colored(cmd, 'red'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)  # the root folder, whose images folder will be recursively liked to output
    parser.add_argument('output', type=str)  # the output folder
    parser.add_argument('--mask_dir', default='mask')
    parser.add_argument('--one_human', action='store_true')
    args = parser.parse_args()

    humans = os.listdir(args.path)
    if args.one_human:  # replace human
        humans = [os.path.split(args.path)[-1]]
        args.path = args.path.replace(humans[0], '')
    # path: should contain multiple humans, images in humans, cameras in images, and actual image in cameras
    for human in humans:
        human_dir = join(args.path, human)
        img_dir = join(human_dir, 'images')  # copy target (split destination)
        msk_dir = join(human_dir, args.mask_dir)  # eval results (to be split)
        for cam in sorted(os.listdir(img_dir)):
            cam_dir = join(img_dir, cam)  # reference: original image dir
            if cam_dir == args.output.replace(args.mask_dir, 'images'):
                continue
            for img in sorted(os.listdir(cam_dir)):
                msk = img.replace('.jpg', '.png')
                new_msk = f"{human}.{cam}.{msk}"  # like F1_06_000000.02.000000.jpg
                msk_cam_dir = join(msk_dir, cam)
                msk_path = join(msk_cam_dir, msk)  # copy result like (...)F1_06_000000/images/02/000000.png
                new_msk_path = join(args.output, new_msk)  # copy from (to be split)
                run(f'mkdir -p {msk_cam_dir}')
                run(f'mv {new_msk_path} {msk_path}')


if __name__ == '__main__':
    main()
