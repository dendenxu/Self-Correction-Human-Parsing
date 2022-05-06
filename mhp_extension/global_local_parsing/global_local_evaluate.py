#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   evaluate.py
@Time    :   8/4/19 3:36 PM
@Desc    :
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import argparse
import numpy as np
import torch

from torch.utils import data
from tqdm import tqdm
from PIL import Image as PILImage
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import networks
from utils.miou import compute_mean_ioU
from utils.transforms import BGR2RGB_transform
from utils.transforms import transform_parsing, transform_logits
from mhp_extension.global_local_parsing.global_local_datasets import CropDataValSet


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    # Network Structure
    parser.add_argument("--arch", type=str, default='resnet101')
    # Data Preference
    parser.add_argument("--data-dir", type=str, default='./data/LIP')
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--split-name", type=str, default='crop_pic')
    parser.add_argument("--input-size", type=str, default='473,473')
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--random-mirror", action="store_true")
    parser.add_argument("--random-scale", action="store_true")
    # Evaluation Preference
    parser.add_argument("--log-dir", type=str, default='./log')
    parser.add_argument("--model-restore", type=str, default='./log/checkpoint.pth.tar')
    parser.add_argument("--gpu", type=str, default='0', help="choose gpu device.")
    parser.add_argument("--save-results", action="store_true", help="whether to save the results.")
    parser.add_argument("--flip", action="store_true", help="random flip during the test.")
    parser.add_argument("--multi-scales", type=str, default='1', help="multiple scales during the test")
    return parser.parse_args()


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def multi_scale_testing(model, batch_input_im, crop_size=[473, 473], flip=True, multi_scales=[1]):
    flipped_idx = (15, 14, 17, 16, 19, 18)
    if len(batch_input_im.shape) > 4:
        batch_input_im = batch_input_im.squeeze()
    if len(batch_input_im.shape) == 3:
        batch_input_im = batch_input_im.unsqueeze(0)

    interp = torch.nn.Upsample(size=crop_size, mode='bilinear', align_corners=True)
    ms_outputs = []
    for s in multi_scales:
        interp_im = torch.nn.Upsample(scale_factor=s, mode='bilinear', align_corners=True)
        scaled_im = interp_im(batch_input_im)
        parsing_output = model(scaled_im)
        parsing_output = parsing_output[0][-1]
        output = parsing_output
        if flip:
            flipped_output = parsing_output[1]
            flipped_output[14:20, :, :] = flipped_output[flipped_idx, :, :]
            output += flipped_output.flip(dims=[-1])
            output *= 0.5
        output = interp(output)
        ms_outputs.append(output)
    ms_fused_parsing_output = torch.stack(ms_outputs)
    ms_fused_parsing_output = ms_fused_parsing_output.mean(0)  # supporting batch dimension
    ms_fused_parsing_output = ms_fused_parsing_output.permute(0, 2, 3, 1)  # HWC
    parsing = torch.argmax(ms_fused_parsing_output, dim=3)
    parsing = parsing.data.cpu().numpy()
    ms_fused_parsing_output = ms_fused_parsing_output.data.cpu().numpy()
    return parsing, ms_fused_parsing_output


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    multi_scales = [float(i) for i in args.multi_scales.split(',')]
    gpus = [int(i) for i in args.gpu.split(',')]
    assert len(gpus) == 1
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.benchmark = True
    cudnn.enabled = True

    h, w = map(int, args.input_size.split(','))
    input_size = [h, w]

    model = networks.init_model(args.arch, num_classes=args.num_classes, pretrained=None)

    IMAGE_MEAN = model.mean
    IMAGE_STD = model.std
    INPUT_SPACE = model.input_space
    print('image mean: {}'.format(IMAGE_MEAN))
    print('image std: {}'.format(IMAGE_STD))
    print('input space:{}'.format(INPUT_SPACE))
    if INPUT_SPACE == 'BGR':
        print('BGR Transformation')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),

        ])
    if INPUT_SPACE == 'RGB':
        print('RGB Transformation')
        transform = transforms.Compose([
            transforms.ToTensor(),
            BGR2RGB_transform(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])

    # Data loader
    lip_test_dataset = CropDataValSet(args.data_dir, args.split_name, crop_size=input_size, transform=transform,
                                      flip=args.flip)
    num_samples = len(lip_test_dataset)
    print('Totoal testing sample numbers: {}'.format(num_samples))
    testloader = data.DataLoader(lip_test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    # Load model weight
    state_dict = torch.load(args.model_restore)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    sp_results_dir = os.path.join(args.log_dir, args.split_name + '_parsing')
    if not os.path.exists(sp_results_dir):
        os.makedirs(sp_results_dir)

    palette = get_palette(20)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(testloader)):
            image, meta = batch
            if (len(image.shape) > 4):
                image = image.squeeze()
            parsing, logits = multi_scale_testing(model, image.cuda(), crop_size=input_size, flip=args.flip,
                                                  multi_scales=multi_scales)
            if args.save_results:
                for b in range(len(parsing)):
                    im_name = meta['name'][b]
                    c = meta['center'].numpy()[b]
                    s = meta['scale'].numpy()[b]
                    w = meta['width'].numpy()[b]
                    h = meta['height'].numpy()[b]
                    parsing_result = transform_parsing(parsing[b], c, s, w, h, input_size)
                    parsing_result_path = os.path.join(sp_results_dir, im_name + '.png')
                    output_im = PILImage.fromarray(np.asarray(parsing_result, dtype=np.uint8))
                    output_im.putpalette(palette)
                    output_im.save(parsing_result_path)
                    # # save logits
                    # logits_result = transform_logits(logits, c, s, w, h, input_size)
                    # logits_result_path = os.path.join(sp_results_dir, im_name + '.npy')
                    # np.save(logits_result_path, logits_result)
        return


if __name__ == '__main__':
    main()
