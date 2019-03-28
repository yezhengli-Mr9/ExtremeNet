import pycocotools.coco as cocoapi
import sys
import cv2
import numpy as np
import pickle
import json
SPLITS = [ 'train']# , 'val'
ANN_PATH = '../data/coco/annotations/instances_{}2017.json'
OUT_PATH = '../data/coco/annotations/instances_small_{}2017.json'
IMG_DIR = '../data/coco/{}2017/'
DEBUG = False
from scipy.spatial import ConvexHull

def _coco_box_to_bbox(box):
  bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                  dtype=np.int32)
  return bbox

def _get_extreme_points(pts):
  l, t = min(pts[:, 0]), min(pts[:, 1])
  r, b = max(pts[:, 0]), max(pts[:, 1])
  # 3 degrees
  thresh = 0.02
  w = r - l + 1
  h = b - t + 1
  
  pts = np.concatenate([pts[-1:], pts, pts[:1]], axis=0)
  t_idx = np.argmin(pts[:, 1])
  t_idxs = [t_idx]
  tmp = t_idx + 1
  while tmp < pts.shape[0] and pts[tmp, 1] - pts[t_idx, 1] <= thresh * h:
    t_idxs.append(tmp)
    tmp += 1
  tmp = t_idx - 1
  while tmp >= 0 and pts[tmp, 1] - pts[t_idx, 1] <= thresh * h:
    t_idxs.append(tmp)
    tmp -= 1
  tt = [(max(pts[t_idxs, 0]) + min(pts[t_idxs, 0])) // 2, t]

  b_idx = np.argmax(pts[:, 1])
  b_idxs = [b_idx]
  tmp = b_idx + 1
  while tmp < pts.shape[0] and pts[b_idx, 1] - pts[tmp, 1] <= thresh * h:
    b_idxs.append(tmp)
    tmp += 1
  tmp = b_idx - 1
  while tmp >= 0 and pts[b_idx, 1] - pts[tmp, 1] <= thresh * h:
    b_idxs.append(tmp)
    tmp -= 1
  bb = [(max(pts[b_idxs, 0]) + min(pts[b_idxs, 0])) // 2, b]

  l_idx = np.argmin(pts[:, 0])
  l_idxs = [l_idx]
  tmp = l_idx + 1
  while tmp < pts.shape[0] and pts[tmp, 0] - pts[l_idx, 0] <= thresh * w:
    l_idxs.append(tmp)
    tmp += 1
  tmp = l_idx - 1
  while tmp >= 0 and pts[tmp, 0] - pts[l_idx, 0] <= thresh * w:
    l_idxs.append(tmp)
    tmp -= 1
  ll = [l, (max(pts[l_idxs, 1]) + min(pts[l_idxs, 1])) // 2]

  r_idx = np.argmax(pts[:, 0])
  r_idxs = [r_idx]
  tmp = r_idx + 1
  while tmp < pts.shape[0] and pts[r_idx, 0] - pts[tmp, 0] <= thresh * w:
    r_idxs.append(tmp)
    tmp += 1
  tmp = r_idx - 1
  while tmp >= 0 and pts[r_idx, 0] - pts[tmp, 0] <= thresh * w:
    r_idxs.append(tmp)
    tmp -= 1
  rr = [r, (max(pts[r_idxs, 1]) + min(pts[r_idxs, 1])) // 2]

  return np.array([tt, ll, bb, rr])

if __name__ == '__main__':
  for split in SPLITS:
    data = json.load(open(ANN_PATH.format(split), 'r'))
    small_data = {'categories':data['categories'], 'images':[], 'annotations':[]}
    #----
    print("[gen_small_coco_extreme_points] data.keys()", data.keys())

    # [gen_small_coco_extreme_points] data.keys() dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
    # print("[gen_small_coco_extreme_points] data['categories']", data['categories'])
    #----
    coco = cocoapi.COCO(ANN_PATH.format(split))
    img_ids = coco.getImgIds()
    num_images = len(img_ids)
    num_classes = 80
    tot_box = 0
    print('num_images', num_images)
    anns_all = data['annotations']
    # print("[gen_small_coco_extreme_points] anns_all",type(anns_all), type([]) )#,anns_all)

    for i, ann in enumerate(anns_all):
        
      img_id = ann['image_id']
      img_info = coco.loadImgs(ids=[img_id])[0]
      img_path = IMG_DIR.format(split) + img_info['file_name']
      img = cv2.imread(img_path)
      # print("img_path", img_path)
      if None is not img:
        small_data['annotations'].append(ann)
        small_data['images'].append(img_info)
        # print("img_info['file_name']", img_info['file_name'], "img_path", img_path)
        tot_box += 1
    del data
    print('tot_box', tot_box)   
    # data['annotations'] = anns_all
    json.dump(small_data, open(OUT_PATH.format(split), 'w'))#yezheng: this does nothing
  

