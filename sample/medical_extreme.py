import cv2, sys
import math
import numpy as np
import torch
import random
import string

from config import system_configs
from utils import crop_image, normalize_, color_jittering_, lighting_
from .utils import random_crop_pts, draw_gaussian, gaussian_radius
from utils.debugger import Debugger

def _resize_image_pts(image, detections, extreme_pts, size):
    detections    = detections.copy()
    height, width = image.shape[0:2]
    new_height, new_width = size

    image = cv2.resize(image, (new_width, new_height))
    
    height_ratio = new_height / height
    width_ratio  = new_width  / width
    detections[:, 0:4:2] *= width_ratio
    detections[:, 1:4:2] *= height_ratio
    extreme_pts[:, :, 0] *= width_ratio
    extreme_pts[:, :, 1] *= height_ratio
    return image, detections, extreme_pts

def _clip_detections_pts(image, detections, extreme_pts):
    detections    = detections.copy()
    height, width = image.shape[0:2]

    detections[:, 0:4:2] = np.clip(detections[:, 0:4:2], 0, width - 1)
    detections[:, 1:4:2] = np.clip(detections[:, 1:4:2], 0, height - 1)
    extreme_pts[:, :, 0] = np.clip(extreme_pts[:, :, 0], 0, width - 1)
    extreme_pts[:, :, 1] = np.clip(extreme_pts[:, :, 1], 0, height - 1)
    keep_inds  = ((detections[:, 2] - detections[:, 0]) > 0) & \
                 ((detections[:, 3] - detections[:, 1]) > 0)
    detections = detections[keep_inds]
    extreme_pts = extreme_pts[keep_inds]
    return detections, extreme_pts

def kp_detection(db, k_ind, data_aug, debug):
    data_rng   = system_configs.data_rng
    batch_size = system_configs.batch_size

    categories   = db.configs["categories"]
    input_size   = db.configs["input_size"]
    output_size  = db.configs["output_sizes"][0]

    border        = db.configs["border"]
    lighting      = db.configs["lighting"]
    rand_crop     = db.configs["rand_crop"]
    rand_color    = db.configs["rand_color"]
    rand_scales   = db.configs["rand_scales"]
    gaussian_bump = db.configs["gaussian_bump"]
    gaussian_iou  = db.configs["gaussian_iou"]
    gaussian_rad  = db.configs["gaussian_radius"]

    max_tag_len = 128

    # allocating memory
    images     = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)
    t_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    l_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    b_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    r_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    ct_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    t_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    l_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    b_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    r_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    t_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    l_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    b_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    r_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    ct_tags    = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    tag_masks  = np.zeros((batch_size, max_tag_len), dtype=np.uint8)
    tag_lens   = np.zeros((batch_size, ), dtype=np.int32)

    db_size = db.db_inds.size
    # print("[kp_detection] db.db_inds", db.db_inds, "db_size", db_size)
    for b_ind in range(batch_size):
        if not debug and 0 == k_ind:
            db.shuffle_inds()

        db_ind = db.db_inds[k_ind]
        k_ind  = (k_ind + 1) % db_size

        # reading image
        # print("[kp_detection] db_ind", db_ind, "db.image_file", db.image_file)
        if 'linux' == sys.platform:
            image_file = db.image_file(db_ind)
        else:
            image_file = db.image_file(db_ind)
        # print("[kp_detection] image_file", image_file)
        image  = cv2.imread(image_file)

        # reading detections
        detections, extreme_pts = db.detections(db_ind)
        # print("[kp_detection] detections", detections)
        # cropping an image randomly
        if rand_crop:
            image, detections, extreme_pts = random_crop_pts(
                image, detections, extreme_pts, 
                rand_scales, input_size, border=border)
        else:
            assert 0
            # image, detections = _full_image_crop(image, detections)

        image, detections, extreme_pts = _resize_image_pts(
            image, detections, extreme_pts, input_size)
        detections, extreme_pts = _clip_detections_pts(
            image, detections, extreme_pts)

        width_ratio  = output_size[1] / input_size[1]
        height_ratio = output_size[0] / input_size[0]

        # flipping an image randomly
        if np.random.uniform() > 0.5:
            image[:] = image[:, ::-1, :]
            width    = image.shape[1]
            detections[:, [0, 2]] = width - detections[:, [2, 0]] - 1
            extreme_pts[:, :, 0] = width - extreme_pts[:, :, 0] - 1
            extreme_pts[:, 1, :], extreme_pts[:, 3, :] = \
                extreme_pts[:, 3, :].copy(), extreme_pts[:, 1, :].copy()
        
        image = image.astype(np.float32) / 255.
        if not debug:
            if rand_color:
                color_jittering_(data_rng, image)
                if lighting:
                    lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)
        normalize_(image, db.mean, db.std)
        images[b_ind] = image.transpose((2, 0, 1))

        for ind, detection in enumerate(detections):
            category = int(detection[-1]) - 1
            extreme_pt = extreme_pts[ind]

            xt, yt = extreme_pt[0, 0], extreme_pt[0, 1]
            xl, yl = extreme_pt[1, 0], extreme_pt[1, 1]
            xb, yb = extreme_pt[2, 0], extreme_pt[2, 1]
            xr, yr = extreme_pt[3, 0], extreme_pt[3, 1]
            xct    = (xl + xr) / 2
            yct    = (yt + yb) / 2

            fxt = (xt * width_ratio)
            fyt = (yt * height_ratio)
            fxl = (xl * width_ratio)
            fyl = (yl * height_ratio)
            fxb = (xb * width_ratio)
            fyb = (yb * height_ratio)
            fxr = (xr * width_ratio)
            fyr = (yr * height_ratio)
            fxct = (xct * width_ratio)
            fyct = (yct * height_ratio)

            xt = int(fxt)
            yt = int(fyt)
            xl = int(fxl)
            yl = int(fyl)
            xb = int(fxb)
            yb = int(fyb)
            xr = int(fxr)
            yr = int(fyr)
            xct = int(fxct)
            yct = int(fyct)

            if gaussian_bump:
                width  = detection[2] - detection[0]
                height = detection[3] - detection[1]

                width  = math.ceil(width * width_ratio)
                height = math.ceil(height * height_ratio)

                if gaussian_rad == -1:
                    radius = gaussian_radius((height, width), gaussian_iou)
                    radius = max(0, int(radius))
                else:
                    radius = gaussian_rad
                # print("[kp_detection], t_heatmaps",t_heatmaps.shape,"b_ind",b_ind,
                #  "category", category,"[xt, yt]", [xt, yt], "radius", radius)

# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 5 category 2 [xt, yt] [79, 63] radius 2
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 5 category 1 [xt, yt] [79, 65] radius 7
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 3 category 0 [xt, yt] [1, 63] radius 13
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 7 category 0 [xt, yt] [49, 53] radius 6
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 7 category 1 [xt, yt] [0, 75] radius 10
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 4 category 0 [xt, yt] [53, 59] radius 4
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 6 category 0 [xt, yt] [0, 36] radius 5
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 6 category 2 [xt, yt] [81, 46] radius 4
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 5 category 0 [xt, yt] [127, 57] radius 5
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 5 category 2 [xt, yt] [50, 59] radius 3
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 7 category 0 [xt, yt] [65, 43] radius 6
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 7 category 2 [xt, yt] [0, 66] radius 2
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 7 category 1 [xt, yt] [6, 63] radius 13
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 8 category 0 [xt, yt] [68, 73] radius 5
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 6 category 0 [xt, yt] [44, 47] radius 7
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 8 category 0 [xt, yt] [127, 45] radius 9
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 8 category 2 [xt, yt] [57, 82] radius 6
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 7 category 0 [xt, yt] [100, 50] radius 8
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 7 category 2 [xt, yt] [7, 84] radius 3
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 7 category 1 [xt, yt] [12, 79] radius 15
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 9 category 0 [xt, yt] [46, 40] radius 6
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 9 category 2 [xt, yt] [110, 66] radius 2
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 9 category 1 [xt, yt] [108, 62] radius 12
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 10 category 0 [xt, yt] [127, 45] radius 5
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 10 category 2 [xt, yt] [59, 66] radius 3
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 10 category 1 [xt, yt] [98, 63] radius 8
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 9 category 0 [xt, yt] [0, 41] radius 4
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 9 category 2 [xt, yt] [40, 64] radius 2
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 9 category 1 [xt, yt] [39, 60] radius 7
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 10 category 0 [xt, yt] [0, 51] radius 12
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 10 category 2 [xt, yt] [127, 55] radius 2
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 8 category 0 [xt, yt] [0, 49] radius 5
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 8 category 2 [xt, yt] [89, 69] radius 2
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 8 category 1 [xt, yt] [55, 65] radius 9
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 9 category 0 [xt, yt] [0, 22] radius 3
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 9 category 2 [xt, yt] [88, 35] radius 6
# [kp_detection], t_heatmaps (11, 3, 128, 128) b_ind 10 category 0 [xt, yt] [83, 51] radius 4

                #yezheng: draw_gaussian(does not return anything)
                draw_gaussian(t_heatmaps[b_ind, category], [xt, yt], radius)
                draw_gaussian(l_heatmaps[b_ind, category], [xl, yl], radius)
                draw_gaussian(b_heatmaps[b_ind, category], [xb, yb], radius)
                draw_gaussian(r_heatmaps[b_ind, category], [xr, yr], radius)
                draw_gaussian(ct_heatmaps[b_ind, category], [xct, yct], radius)
            else:
                t_heatmaps[b_ind, category, yt, xt] = 1
                l_heatmaps[b_ind, category, yl, xl] = 1
                b_heatmaps[b_ind, category, yb, xb] = 1
                r_heatmaps[b_ind, category, yr, xr] = 1
#             print("[kp_detection], t_heatmaps",np.sum(t_heatmaps),
#                 "l_heatmaps", np.sum(l_heatmaps), "b_heatmaps", np.sum(b_heatmaps) )

#  [kp_detection], t_heatmaps 89.74134 l_heatmaps 73.33087 b_heatmaps 81.54751
# [kp_detection], t_heatmaps 91.35594 l_heatmaps 74.773506 b_heatmaps 82.99014
# [kp_detection], t_heatmaps 120.70633 l_heatmaps 104.12389 b_heatmaps 112.34053
# [kp_detection], t_heatmaps 138.09032 l_heatmaps 133.47427 b_heatmaps 141.69092
# [kp_detection], t_heatmaps 152.16406 l_heatmaps 142.3869 b_heatmaps 155.76468
# [kp_detection], t_heatmaps 228.73175 l_heatmaps 218.95456 b_heatmaps 232.33237
# [kp_detection], t_heatmaps 249.74953 l_heatmaps 239.97234 b_heatmaps 253.35014
# [kp_detection], t_heatmaps 252.96764 l_heatmaps 244.32278 b_heatmaps 257.7006
# [kp_detection], t_heatmaps 344.81143 l_heatmaps 336.1666 b_heatmaps 349.54437
# [kp_detection], t_heatmaps 365.82922 l_heatmaps 357.1844 b_heatmaps 370.56216
# [kp_detection], t_heatmaps 404.90088 l_heatmaps 396.25604 b_heatmaps 409.63382
# [kp_detection], t_heatmaps 408.119 l_heatmaps 399.47415 b_heatmaps 413.98428
# [kp_detection], t_heatmaps 554.1235 l_heatmaps 545.2966 b_heatmaps 559.9888
# [kp_detection], t_heatmaps 575.14124 l_heatmaps 566.31433 b_heatmaps 581.00653
# [kp_detection], t_heatmaps 578.3594 l_heatmaps 569.5325 b_heatmaps 585.357
# [kp_detection], t_heatmaps 601.02057 l_heatmaps 592.19366 b_heatmaps 624.4286
# [kp_detection], t_heatmaps 609.5387 l_heatmaps 600.71185 b_heatmaps 632.9467
# [kp_detection], t_heatmaps 626.92267 l_heatmaps 618.0958 b_heatmaps 662.2971
# [kp_detection], t_heatmaps 635.4408 l_heatmaps 626.61395 b_heatmaps 670.81525
# [kp_detection], t_heatmaps 743.9494 l_heatmaps 686.07666 b_heatmaps 730.27795
# [kp_detection], t_heatmaps 773.2998 l_heatmaps 715.42706 b_heatmaps 759.62836
# [kp_detection], t_heatmaps 881.32434 l_heatmaps 823.93567 b_heatmaps 868.136

            tag_ind = tag_lens[b_ind]
            t_regrs[b_ind, tag_ind, :] = [fxt - xt, fyt - yt]
            l_regrs[b_ind, tag_ind, :] = [fxl - xl, fyl - yl]
            b_regrs[b_ind, tag_ind, :] = [fxb - xb, fyb - yb]
            r_regrs[b_ind, tag_ind, :] = [fxr - xr, fyr - yr]
            #xt, yt are current/ rough ones
            t_tags[b_ind, tag_ind] = yt * output_size[1] + xt #yezheng: why they need to do this? -- from CornerNet
            l_tags[b_ind, tag_ind] = yl * output_size[1] + xl 
            b_tags[b_ind, tag_ind] = yb * output_size[1] + xb
            r_tags[b_ind, tag_ind] = yr * output_size[1] + xr
            ct_tags[b_ind, tag_ind] = yct * output_size[1] + xct
            tag_lens[b_ind] += 1

    for b_ind in range(batch_size):
        tag_len = tag_lens[b_ind]
        tag_masks[b_ind, :tag_len] = 1

    if debug:
        debugger = Debugger(num_classes=80)
        t_hm = debugger.gen_colormap(t_heatmaps[0])
        l_hm = debugger.gen_colormap(l_heatmaps[0])
        b_hm = debugger.gen_colormap(b_heatmaps[0])
        r_hm = debugger.gen_colormap(r_heatmaps[0])
        ct_hm = debugger.gen_colormap(ct_heatmaps[0])
        img = images[0] * db.std.reshape(3, 1, 1) + db.mean.reshape(3, 1, 1)
        img =  (img * 255).astype(np.uint8).transpose(1, 2, 0)
        debugger.add_blend_img(img, t_hm, 't_hm')
        debugger.add_blend_img(img, l_hm, 'l_hm')
        debugger.add_blend_img(img, b_hm, 'b_hm')
        debugger.add_blend_img(img, r_hm, 'r_hm')
        debugger.add_blend_img(
            img, np.maximum(np.maximum(t_hm, l_hm), 
                            np.maximum(b_hm, r_hm)), 'extreme')
        debugger.add_blend_img(img, ct_hm, 'center')
        debugger.show_all_imgs(pause=True)

    images     = torch.from_numpy(images)
    t_heatmaps = torch.from_numpy(t_heatmaps)
    l_heatmaps = torch.from_numpy(l_heatmaps)
    b_heatmaps = torch.from_numpy(b_heatmaps)
    r_heatmaps = torch.from_numpy(r_heatmaps)
    ct_heatmaps = torch.from_numpy(ct_heatmaps)
    t_regrs    = torch.from_numpy(t_regrs)
    l_regrs    = torch.from_numpy(l_regrs)
    b_regrs    = torch.from_numpy(b_regrs)
    r_regrs    = torch.from_numpy(r_regrs)
    t_tags     = torch.from_numpy(t_tags)
    l_tags     = torch.from_numpy(l_tags)
    b_tags     = torch.from_numpy(b_tags)
    r_tags     = torch.from_numpy(r_tags)
    ct_tags    = torch.from_numpy(ct_tags)
    tag_masks  = torch.from_numpy(tag_masks)

    return {
        "xs": [images, t_tags, l_tags, b_tags, r_tags, ct_tags],
        "ys": [t_heatmaps, l_heatmaps, b_heatmaps, r_heatmaps, ct_heatmaps,
               tag_masks, t_regrs, l_regrs, b_regrs, r_regrs]
    }, k_ind

def sample_data(db, k_ind, data_aug=True, debug=False):
    return globals()[system_configs.sampling_function](db, k_ind, data_aug, debug)
