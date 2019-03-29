import glob, os, cv2
import numpy as np
import json,tqdm
import copy
import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)


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



#http://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#create-custom-coco-dataset
def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        # print("[create_sub_mask_annotation] poly", poly, "contour", contour)
        poly = poly.simplify(1.0, preserve_topology=True) #False
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation

if '__main__' == __name__:
	output_file  = "../data/medical_img/annotations"
	if not os.path.exists(output_file):
		print("NOT EXIST, to create one")
		os.makedirs(output_file)
	output_file_extreme = os.path.join(output_file,"instances_extreme_train2017.json")
	output_file_init = os.path.join(output_file,"instances_train2017.json")
	del output_file

	labels_list_to_check = [4,5,6]#[3,4]
	data_extreme = {"annotations":[],"images":[],"categories": []}
	min_base_index = min(labels_list_to_check) - 1
	# for label in labels_list_to_check:
  #========
	# data_extreme["categories"].append({"id":3-min_base_index,"name":"spin_cord", "supercategory":"spin_cord"})
	# data_extreme["categories"].append({"id":4-min_base_index,"name":"probe_right", "supercategory":"probe"})
	#========  
	data_extreme["categories"].append({"id":4-min_base_index,"name":"probe_right", "supercategory":"probe"})
	data_extreme["categories"].append({"id":6-min_base_index,"name":"probe_left", "supercategory":"probe"})
	data_extreme["categories"].append({"id":5-min_base_index,"name":"scissor", "supercategory":"scissor"})
  #========
	mask_list = glob.glob('../../../medical_img/data/test/mask_label/*.png')
	# print("mask_list", mask_list)
	mask_list = [m for m in mask_list if "corrected" not in m]
	mask_list.sort()
	annotation_id = 1 
	data_init = copy.deepcopy(data_extreme)
	for i in tqdm.tqdm(range(len(mask_list))):
		mask_name = mask_list[i]
		splits = mask_name.split('/')[-1].split('_')[0]#[:-4].split('_')[0].split('e')
		index = int(''.join([s for s in splits if s.isdigit()]))
		# mask = cv2.imread(mask_name,0)#.astype(np.float64)
		
		img_origin_directory = "/home/yezheng/medical_img/data/test/img"
		img_origin = cv2.imread(os.path.join(img_origin_directory, 
			"{:06d}.jpg".format(index) ) )
		if None is img_origin:
			mask_directory = "/Users/yezheng/medical_img/data/test/img"
			img_origin = cv2.imread(os.path.join(img_origin_directory, 
			"{:06d}.jpg".format(index) ) )
		data_extreme["images"].append({"file_name":os.path.join(img_origin_directory, 
			"{:06d}.jpg".format(index) ),"id":index})
		data_init["images"].append({"file_name":os.path.join(img_origin_directory, 
			"{:06d}.jpg".format(index) ),"id":index})

		# mask_corrected = np.zeros(mask.shape)
		for label in labels_list_to_check:

			mask_corrected_name = '../../../medical_img/data/test/mask_label/_corrected{}/correctedFrame{:04d}_ordered_{}.png'.format(label, index, label)
			
			mask_corrected = cv2.imread(mask_corrected_name,0)
			# print(mask_corrected.shape)
			if None is mask_corrected:  
				print("mask_corrected_name NOT EXIST", mask_corrected_name)
				continue
			mask_corrected = (mask_corrected >0).astype(np.int32)
			ann = create_sub_mask_annotation(sub_mask = mask_corrected, image_id = index, category_id = label-min_base_index, annotation_id = annotation_id , is_crowd = 0)
			# print("[gen_medical_extreme_points] ann", ann)
			annotation_id += 1 # automatically increase
			tmp = np.where(mask_corrected > 0)
			pts = np.asarray(tmp).transpose()[:, ::-1].astype(np.int32)
			extreme_points = _get_extreme_points(pts).astype(np.int32)
			data_init['annotations'].append(ann)
			ann['extreme_points'] = extreme_points.copy().tolist()
			data_extreme['annotations'].append(ann)
			del ann
			
	json.dump(data_extreme, open(output_file_extreme, 'w'))
	json.dump(data_init, open(output_file_init, 'w'))
	