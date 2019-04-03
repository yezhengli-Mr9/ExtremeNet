#!/usr/bin/env python
import os
import json
import torch
import pprint
import argparse
import importlib
import numpy as np
import cv2, tqdm

import matplotlib
matplotlib.use("Agg")

from config import system_configs
from nnet.py_factory import NetworkFactory

from config import system_configs
from utils import crop_image, normalize_
from external.nms import soft_nms_with_points as soft_nms
from utils.color_map import colormap
from utils.visualize import vis_mask, vis_octagon, vis_ex, vis_class, vis_bbox
from dextr import Dextr
from db.datasets import datasets

torch.backends.cudnn.benchmark = False

#
# class_name = ['__background__', "probe_right", "spin_cord"]
class_name = ['__background__', "probe", "scissor","probe"]
image_ext = ['jpg', 'jpeg', 'png', 'webp']

def parse_args():
    parser = argparse.ArgumentParser(description="Demo CornerNet")
    parser.add_argument("--cfg_file", help="config file", 
                        default='medical_ExtremeNet', type=str)
    parser.add_argument("--demo", help="demo image path or folders",
                        default="data/medical_img/test2017", type=str)
    parser.add_argument("--model_path",
                        default='cache/nnet/medical_ExtremeNet/medical_ExtremeNet_26000.pkl')
    parser.add_argument("--show_mask", action='store_true',
                        help="Run Deep extreme cut to obtain accurate mask")

    args = parser.parse_args()
    return args

def _rescale_dets(detections, ratios, borders, sizes):
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
    xs    /= ratios[:, 1][:, None, None]
    ys    /= ratios[:, 0][:, None, None]
    xs    -= borders[:, 2][:, None, None]
    ys    -= borders[:, 0][:, None, None]
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)

def _rescale_ex_pts(detections, ratios, borders, sizes):
    xs, ys = detections[..., 5:13:2], detections[..., 6:13:2]
    xs    /= ratios[:, 1][:, None, None]
    ys    /= ratios[:, 0][:, None, None]
    xs    -= borders[:, 2][:, None, None]
    ys    -= borders[:, 0][:, None, None]
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)

def _box_inside(box2, box1):
    inside = (box2[0] >= box1[0] and box2[1] >= box1[1] and \
       box2[2] <= box1[2] and box2[3] <= box1[3])
    return inside 

def kp_decode(nnet, images, K, kernel=3, aggr_weight=0.1, 
              scores_thresh=0.1, center_thresh=0.1, debug=False):
    detections = nnet.test(
        [images], kernel=kernel, aggr_weight=aggr_weight, 
        scores_thresh=scores_thresh, center_thresh=center_thresh, debug=debug)
    detections = detections.data.cpu().numpy()
    return detections

if __name__ == "__main__":
    args = parse_args()
    cfg_file = os.path.join(
        system_configs.config_dir, args.cfg_file + ".json")
    print("[demo] cfg_file: {}".format(cfg_file))

    with open(cfg_file, "r") as f:
        configs = json.load(f)
            
    configs["system"]["snapshot_name"] = args.cfg_file
    system_configs.update_config(configs["system"])
    print("system config...")
    pprint.pprint(system_configs.full)
    
    print("loading parameters: {}".format(args.model_path))
    print("building neural network...")
    train_split = system_configs.train_split
    dataset = system_configs.dataset
    training_db = datasets[dataset](configs["db"], train_split)
    nnet = NetworkFactory(training_db, configs["cuda_flag"])
    print("loading parameters...")
    nnet.load_pretrained_params(args.model_path)
    if torch.cuda.is_available() and  configs["cuda_flag"]:
        nnet.cuda()
    nnet.eval_mode()

    K             = configs["db"]["top_k"]
    aggr_weight   = configs["db"]["aggr_weight"]
    scores_thresh = configs["db"]["scores_thresh"]
    center_thresh = configs["db"]["center_thresh"]
    suppres_ghost = True
    nms_kernel    = 3
    
    scales        = configs["db"]["test_scales"]
    weight_exp    = 8
    categories    = configs["db"]["categories"]
    print('''[demo] configs["db"]''', configs["db"])
    nms_threshold = configs["db"]["nms_threshold"]
    max_per_image = configs["db"]["max_per_image"]
    nms_algorithm = {
        "nms": 0,
        "linear_soft_nms": 1, 
        "exp_soft_nms": 2
    }["exp_soft_nms"]
    if args.show_mask:
        dextr = Dextr()


    mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
    std  = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
    top_bboxes = {}
    # print("[demo] args.demo", args.demo, "os.path.isdir(args.demo)", os.path.isdir(args.demo))
    if os.path.isdir(args.demo):
        image_names = []
        ls = os.listdir(args.demo)
        # print("os.listdir(args.demo)", ls)
        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in image_ext:
                image_names.append(os.path.join(args.demo, file_name))
    else:
        image_names = [args.demo]
    # print("[demo] image_names", image_names, "args.demo", args.demo,
    #     "os.path.isdir(args.demo)", os.path.isdir(args.demo),"args", args)
    for image_id in tqdm.tqdm(range(len(image_names))):
        image_name  = image_names[image_id]
        # print("image_name.split('.')[-2][-6:]", image_name.split('.')[-2][-6:])
        # if 1310 >= int(image_name.split('.')[-2][-6:]):
        #     continue
        print('Running ', image_name)
        image      = cv2.imread(image_name)

        height, width = image.shape[0:2]

        detections = []

        for scale in scales:
            new_height = int(height * scale)
            new_width  = int(width * scale)
            new_center = np.array([new_height // 2, new_width // 2])

            inp_height = new_height | 127
            inp_width  = new_width  | 127

            images  = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
            ratios  = np.zeros((1, 2), dtype=np.float32)
            borders = np.zeros((1, 4), dtype=np.float32)
            sizes   = np.zeros((1, 2), dtype=np.float32)

            out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
            height_ratio = out_height / inp_height
            width_ratio  = out_width  / inp_width

            resized_image = cv2.resize(image, (new_width, new_height))
            resized_image, border, offset = crop_image(
                resized_image, new_center, [inp_height, inp_width])

            resized_image = resized_image / 255.
            normalize_(resized_image, mean, std)

            images[0]  = resized_image.transpose((2, 0, 1))
            borders[0] = border
            sizes[0]   = [int(height * scale), int(width * scale)]
            ratios[0]  = [height_ratio, width_ratio]

            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
            images = torch.from_numpy(images)
            # print("[demo] scales", scales)
            dets   = kp_decode(
                nnet, images, K, aggr_weight=aggr_weight, 
                scores_thresh=scores_thresh, center_thresh=center_thresh,
                kernel=nms_kernel, debug=True)
            
            dets   = dets.reshape(2, -1, 14)
            dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
            dets[1, :, [5, 7, 9, 11]] = out_width - dets[1, :, [5, 7, 9, 11]]
            dets[1, :, [7, 8, 11, 12]] = dets[1, :, [11, 12, 7, 8]].copy()
            dets   = dets.reshape(1, -1, 14)

            _rescale_dets(dets, ratios, borders, sizes)
            _rescale_ex_pts(dets, ratios, borders, sizes)
            dets[:, :, 0:4] /= scale
            dets[:, :, 5:13] /= scale
            detections.append(dets)

        detections = np.concatenate(detections, axis=1)

        classes    = detections[..., -1]
        classes    = classes[0]
        detections = detections[0]

        # reject detections with negative scores
        keep_inds  = (detections[:, 4] > 0)
        detections = detections[keep_inds]
        classes    = classes[keep_inds]

        top_bboxes[image_id] = {}
        for j in range(categories):
            keep_inds = (classes == j)
            top_bboxes[image_id][j + 1] = \
                detections[keep_inds].astype(np.float32)
            soft_nms(top_bboxes[image_id][j + 1], 
                     Nt=nms_threshold, method=nms_algorithm)

        scores = np.hstack([
            top_bboxes[image_id][j][:, 4] 
            for j in range(1, categories + 1)
        ])
        if len(scores) > max_per_image:
            kth    = len(scores) - max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, categories + 1):
                keep_inds = (top_bboxes[image_id][j][:, 4] >= thresh)
                top_bboxes[image_id][j] = top_bboxes[image_id][j][keep_inds]

        if suppres_ghost:
            for j in range(1, categories + 1):
                n = len(top_bboxes[image_id][j])
                for k in range(n):
                    inside_score = 0
                    if top_bboxes[image_id][j][k, 4] > 0.2:
                        for t in range(n):
                            if _box_inside(top_bboxes[image_id][j][t], 
                                           top_bboxes[image_id][j][k]):
                                inside_score += top_bboxes[image_id][j][t, 4]
                        if inside_score > top_bboxes[image_id][j][k, 4] * 3:
                            top_bboxes[image_id][j][k, 4] /= 2


        if 1: # visualize
            color_list    = colormap(rgb=True)
            mask_color_id = 0
            image         = cv2.imread(image_name)
            input_image   = image.copy()
            mask_image    = image.copy()
            bboxes = {}
            # print("[demo] categories", categories)
            Threshold= {1:0.3, 2:0.0, 3:0.3}
            for j in range(1, categories +1):
                keep_inds = (top_bboxes[image_id][j][:, 4] > 0.3) #yezheng: this threshold is important
                cat_name  = class_name[j]
                for bbox in top_bboxes[image_id][j][keep_inds]:
                    sc    = bbox[4]
                    ex    = bbox[5:13].astype(np.int32).reshape(4, 2)
                    bbox  = bbox[0:4].astype(np.int32)
                    txt   = '{}{:.2f}'.format(cat_name, sc)
                    color_mask = color_list[mask_color_id % len(color_list), :3]
                    mask_color_id += 1
                    # image = vis_bbox(image, 
                    #                  (bbox[0], bbox[1], 
                    #                   bbox[2] - bbox[0], bbox[3] - bbox[1]))
                    image = vis_class(image, 
                                      (bbox[0], bbox[1] - 2), txt)
                    # image = vis_octagon( image, ex, color_mask)
                    image = vis_ex(image, ex, color_mask)

                    # if args.show_mask:
                    mask = dextr.segment(input_image[:, :, ::-1], ex) # BGR to RGB
                    mask = np.asfortranarray(mask.astype(np.uint8))
                    mask_image = vis_bbox(mask_image, 
                                         (bbox[0], bbox[1], 
                                          bbox[2] - bbox[0], 
                                          bbox[3] - bbox[1]))
                    mask_image = vis_class(mask_image, 
                                           (bbox[0], bbox[1] - 2), txt)
                    mask_image = vis_mask(mask_image, mask, color_mask)
            #yezheng: comment out
            if args.show_mask:
                cv2.imshow('mask', mask_image)
            # cv2.imshow('out', image)
            # cv2.waitKey()
            # cv2.imwrite("out_images/"+ image_name.split('/')[-1].split('.')[0]+"_out.png", image)
            cv2.imwrite("out_images/"+ image_name.split('/')[-1].split('.')[0]+"_out.png", mask_image)


