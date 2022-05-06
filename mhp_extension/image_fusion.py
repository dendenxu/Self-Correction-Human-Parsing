import argparse
import cv2
import os
import json
import numpy as np
from PIL import Image as PILImage
import joblib


def patch2img_output(patch_dir, img_name, img_height, img_width, bbox, bbox_type, num_dim):
    """transform bbox patch outputs to image output"""
    assert bbox_type == 'gt' or 'msrcnn'
    output = np.zeros((img_height, img_width, num_dim), dtype=np.uint8)
    for i in range(len(bbox)):  # person index starts from 1
        file_path = os.path.join(patch_dir, os.path.splitext(img_name)[0] + '_' + str(i + 1) + '_' + bbox_type + '.png')
        instance_output = cv2.imread(file_path)
        output[bbox[i][1]:bbox[i][3] + 1, bbox[i][0]:bbox[i][2] + 1] += instance_output[:, :, :num_dim]

    return output


def multi_process(a, args):
    img_name = a['im_name']
    img_height = a['img_height']
    img_width = a['img_width']
    msrcnn_bbox = a['person_bbox']
    # bbox_score = a['person_bbox_score']

    parsing = patch2img_output(args.gt_output_dir, img_name, img_height, img_width, msrcnn_bbox, bbox_type='msrcnn', num_dim=3)
    mask = (parsing != 0).any(axis=-1).astype(np.uint8) * 255

    parsing_dir = os.path.join(args.save_dir, 'global_parsing')
    schp_dir = os.path.join(args.save_dir, 'schp')

    os.makedirs(parsing_dir, exist_ok=True)
    os.makedirs(schp_dir, exist_ok=True)

    parsing_path = os.path.join(parsing_dir, os.path.splitext(img_name)[0] + '.png')
    mask_path = os.path.join(schp_dir, os.path.splitext(img_name)[0] + '.png')

    cv2.imwrite(parsing_path, parsing)
    cv2.imwrite(mask_path, mask)

    return


def main(args):
    json_file = open(args.test_json_path)
    anno = json.load(json_file)['root']

    results = joblib.Parallel(n_jobs=64, verbose=10, pre_dispatch="all")(
        [joblib.delayed(multi_process)(a, args) for i, a in enumerate(anno)]
    )


def get_arguments():
    parser = argparse.ArgumentParser(description="obtain final prediction by logits fusion")
    parser.add_argument("--test_json_path", type=str, default='./data/CIHP/cascade_152_finetune/test.json')
    # parser.add_argument("--global_output_dir", type=str,
    # default='./data/CIHP/global/global_result-cihp-resnet101/global_output')
    # parser.add_argument("--msrcnn_output_dir", type=str,
    # default='./data/CIHP/cascade_152__finetune/msrcnn_result-cihp-resnet101/msrcnn_output')
    parser.add_argument("--gt_output_dir", type=str, default='./data/CIHP/cascade_152__finetune/gt_result-cihp-resnet101/gt_output')
    # parser.add_argument("--mask_output_dir", type=str, default='./data/CIHP/cascade_152_finetune/mask')
    parser.add_argument("--save_dir", type=str, default='./data/CIHP/fusion_results/cihp-msrcnn_finetune')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    main(args)
