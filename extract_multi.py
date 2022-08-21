import os
import time
import torch
import shutil
from os import chdir
from tqdm import tqdm
from glob import glob
from os.path import join
from termcolor import colored

current_dir = os.path.abspath(os.path.dirname(__file__))


def move_root(): return chdir(current_dir)
def move_mhp(): return chdir(join(current_dir, 'mhp_extension'))
def move_tool(): return chdir(join(current_dir, 'mhp_extension', 'detectron2', 'tools'))


# # declare which gpu device to use
# global_cuda_device = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').strip().split(',')[0])


# def check_mem(global_cuda_device):
#     devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
#     total, used = devices_info[int(global_cuda_device)].split(',')
#     return total, used


# def occumpy_mem(global_cuda_device):
#     total, used = check_mem(global_cuda_device)
#     total = int(total) # in MB
#     used = int(used) # in MB
#     max_mem = int(total * 0.5)
#     block_mem = max_mem - used
#     x = torch.empty(256, 1024, block_mem, dtype=torch.float32, device='cuda') # will respect cuda_visible_devices
#     del x

# occumpy_mem(global_cuda_device)

def myprint(cmd, level):
    color = {'run': 'blue', 'info': 'green', 'warn': 'yellow', 'error': 'red'}[level]
    print(colored(cmd, color))


def log(text):
    myprint(text, 'info')


def mywarn(text):
    myprint(text, 'warn')


def myerror(text):
    myprint(text, 'error')


def run_cmd(cmd, verbo=True):
    if verbo:
        myprint('[run] ' + cmd, 'run')
    os.system(cmd)
    return []


def check_and_run(outname, cmd):
    if (os.path.exists(outname) and os.path.isfile(outname)) or (os.path.isdir(outname) and len(os.listdir(outname)) >= 3):
        mywarn('Skip {}'.format(cmd))
    else:
        run_cmd(cmd)


def schp_pipeline(img_dir, ckpt_dir):
    tmp_dir = os.path.abspath(join(args.tmp, 'tmp_' + '_'.join(img_dir.split(os.sep)[-3:])))
    resdir = join(args.tmp, img_dir.split(os.sep)[-3], img_dir.split(os.sep)[-1])
    if os.path.exists(resdir):
        return 0
    move_mhp()
    annotations = join(tmp_dir, 'Demo.json')
    cmd = f"python3 ./coco_style_annotation_creator/test_human2coco_format.py --dataset 'Demo' --json_save_dir {tmp_dir} --test_img_dir {img_dir}"
    check_and_run(annotations, cmd)

    move_tool()
    # 通过设置环境变量来控制
    os.environ['annotations'] = annotations
    os.environ['img_dir'] = img_dir
    cmd = f"python3 ./finetune_net.py --num-gpus 1 --config-file ../configs/Misc/demo.yaml --eval-only MODEL.WEIGHTS {join(ckpt_dir, 'detectron2_maskrcnn_cihp_finetune.pth')} TEST.AUG.ENABLED False DATALOADER.NUM_WORKERS 16 OUTPUT_DIR {join(tmp_dir, 'detectron2_prediction')} SOLVER.IMS_PER_BATCH 32"
    check_and_run(join(tmp_dir, 'detectron2_prediction'), cmd)

    move_mhp()
    cmd = f"python3 make_crop_and_mask_w_mask_nms.py --img_dir {img_dir} --save_dir {tmp_dir} --img_list {annotations} --det_res {tmp_dir}/detectron2_prediction/inference/instances_predictions.pth"
    check_and_run(join(tmp_dir, 'crop_pic'), cmd)

    move_root()
    os.environ['PYTHONPATH'] = '{}:{}'.format(current_dir, os.environ.get('PYTHONPATH', ''))
    cmd = f"python3 mhp_extension/global_local_parsing/global_local_evaluate.py --data-dir {tmp_dir} --split-name crop_pic --model-restore {ckpt_dir}/exp_schp_multi_cihp_local.pth --log-dir {tmp_dir} --save-results"
    check_and_run(join(tmp_dir, 'crop_pic_parsing'), cmd)

    # if not os.path.exists(join(tmp_dir, 'global_pic')):
    #     os.system('ln -s {} {}'.format(img_dir, join(tmp_dir, 'global_pic')))
    # cmd = f"python mhp_extension/global_local_parsing/global_local_evaluate.py --data-dir {tmp_dir} --split-name global_pic --model-restore {ckpt_dir}/exp_schp_multi_cihp_global.pth --log-dir {tmp_dir} --save-results"
    # check_and_run(join(tmp_dir, 'global_pic_parsing'), cmd)

    # cmd = f"python mhp_extension/logits_fusion.py --test_json_path {tmp_dir}/crop.json --global_output_dir {tmp_dir}/global_pic_parsing --gt_output_dir {tmp_dir}/crop_pic_parsing --mask_output_dir {tmp_dir}/crop_mask --save_dir {tmp_dir}/mhp_fusion_parsing"
    cmd = f"python mhp_extension/image_fusion.py --test_json_path {tmp_dir}/crop.json --gt_output_dir {tmp_dir}/crop_pic_parsing --save_dir {tmp_dir}/mhp_fusion_parsing"
    run_cmd(cmd)
    # check the output
    out_dir = join(tmp_dir, 'mhp_fusion_parsing', 'global_tag')
    visnames = sorted(glob(join(out_dir, '*_vis.png')))
    imgnames = sorted(glob(join(img_dir, '*.jpg'))+glob(join(img_dir, '*.png')))
    if len(visnames) == len(imgnames):
        log('[log] Finish extracting')
        log('[log] Copy results')
        os.makedirs(os.path.dirname(resdir), exist_ok=True)
        # shutil.copytree(join(tmp_dir, 'mhp_fusion_parsing', 'schp'), resdir)
        os.system('cp -r {} {}'.format(join(tmp_dir, 'mhp_fusion_parsing', 'schp'), resdir))
        # for name in ['global_pic_parsing', 'crop_pic_parsing']:
        for name in ['crop_pic_parsing']:
            dirname = join(tmp_dir, name)
            if os.path.exists(dirname):
                log('[log] remove {}'.format(dirname))
                shutil.rmtree(dirname)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--ckpt_dir', type=str, default='/nas/share/schp')
    parser.add_argument('--tmp', type=str, default='data')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--only', type=str, nargs='*', default=[])
    args = parser.parse_args()

    for sub in sorted(os.listdir(join(args.path, 'images'))):
        if args.only and sub not in args.only:
            continue
        schp_pipeline(join(args.path, 'images', sub), args.ckpt_dir)


# def mhp_fusion_parsing(img_dir, tmp):
#     tmp_dir = os.path.abspath(join(tmp, 'tmp_' + '_'.join(img_dir.split(os.sep)[-3:])))
#     resdir = join(tmp, img_dir.split(os.sep)[-3], img_dir.split(os.sep)[-1])
#     cmd = f"python mhp_extension/logits_fusion.py --test_json_path {tmp_dir}/crop.json --global_output_dir {tmp_dir}/global_pic_parsing --gt_output_dir {tmp_dir}/crop_pic_parsing --mask_output_dir {tmp_dir}/crop_mask --save_dir {tmp_dir}/mhp_fusion_parsing"
#     run_cmd(cmd)
#     # check the output
#     out_dir = join(tmp_dir, 'mhp_fusion_parsing', 'global_tag')
#     visnames = sorted(glob(join(out_dir, '*_vis.png')))
#     imgnames = sorted(glob(join(img_dir, '*.jpg'))+glob(join(img_dir, '*.png')))
#     if len(visnames) == len(imgnames):
#         log('[log] Finish extracting')
#         log('[log] Copy results')
#         os.makedirs(os.path.dirname(resdir), exist_ok=True)
#         # shutil.copytree(join(tmp_dir, 'mhp_fusion_parsing', 'schp'), resdir)
#         os.system('cp -r {} {}'.format(join(tmp_dir, 'mhp_fusion_parsing', 'schp'), resdir))
#         for name in ['global_pic_parsing', 'crop_pic_parsing']:
#             dirname = join(tmp_dir, name)
#             if os.path.exists(dirname):
#                 log('[log] remove {}'.format(dirname))
#                 shutil.rmtree(dirname)
