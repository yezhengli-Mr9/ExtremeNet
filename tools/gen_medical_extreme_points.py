import glob, os, cv2
import numpy as np

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
	label2name = {}
	# label2name[1] = "inst"
	# label2name[2] = "artery"
	# label2name[3] = "spin"

				
	mask_directory = '/home/zhang7/medical_img/data/test/mask_label'
	img_origin_directory = '/home/zhang7/medical_img/data/test/img'
	glob_path = os.path.join(mask_directory, "*.png")
	mask_list = glob.glob(glob_path)
	if 0  ==  len(mask_list):
		mask_directory = '/Users/yezheng/medical_img/data/test/mask_label/_corrected4'#'/Users/yezheng/medical_img/data/test/mask_label'
		img_origin_directory = '/Users/yezheng/medical_img/data/test/img'
		glob_path = os.path.join(mask_directory, "*.png")
		mask_list = glob.glob(glob_path)
	mask_list.sort()
	# print("[gen_medical_extreme_points] mask_list", mask_list , "glob_path", glob_path)
	max__ = 0
	cl = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
	# print("mask_list", mask_list)
	for i, mask_fname in enumerate(mask_list):
		
		tmp = "".join([n for n in mask_fname.split('/')[-1].split('_')[0] if n.isdigit()])
		# print("mask_fname.split('/')[-1]",  mask_fname.split('/')[-1], "tmp", tmp)
		mask_number = int(tmp )

		# print(mask_number)	

		#-----
		mask = cv2.imread(mask_fname, 0)
		# #=========
		# tmp = np.where(5 == mask) # instrument 1
		# if len(tmp[0])>0:
		# 	pts = np.asarray(tmp).transpose()[:, ::-1].astype(np.int32)
		# 	# print("[gen_medical_extreme_points] pts", len(pts))
		# 	extreme_points = _get_extreme_points(pts).astype(np.int32)
		# 	print("(before resize) extreme_points", extreme_points)
		# #=========
		# mask = mask.astype(np.float64)

		# mask = cv2.resize(mask, dsize = (img__.shape[1], img__.shape[0]), interpolation = cv2.INTER_AREA)#interpolation = cv2.INTER_NEAREST)
		mask= mask.astype(np.int32)
		# print("mask",  mask.shape)
		# img__ = cv2.resize(img_origin, dsize = (256,256))
		# #=========
		# tmp = np.where(5 == mask) # instrument 1
		# if len(tmp[0])>0:
		# 	pts = np.asarray(tmp).transpose()[:, ::-1].astype(np.int32)
		# 	# print("[gen_medical_extreme_points] pts", len(pts))
		# 	extreme_points = _get_extreme_points(pts).astype(np.int32)
		# 	print("(after resize) extreme_points", extreme_points)
		# #=========
		img_origin = cv2.imread(os.path.join(img_origin_directory, 
			"{:06d}.jpg".format(mask_number) ) )
		# cv2.imshow("img",img_origin)
		# cv2.waitKey()

		img_origin = cv2.resize(img_origin.astype(np.float64), dsize = (mask.shape[1], mask.shape[0]))
		
		img__ = img_origin
		
		# if i >10:
		# 	exit()
		
		if int(np.max(mask)) > max__:
			max__ = int(np.max(mask))
			print("max__", max__)
		for label in np.unique(mask):
			if label>0:
				tmp = np.where(label == mask) # instrument 1
				if label >100:
					label = 4
					mask = (mask>0)*label
				pts = np.asarray(tmp).transpose()[:, ::-1].astype(np.int32)
				extreme_points = _get_extreme_points(pts).astype(np.int32)
				# if 5 == label: 
					# print("[after resize] extreme_points", extreme_points)
				# 	print(" np.sum(mask, axis = 1).T", np.sum(mask, axis = 1).T)
				# 	print(" np.sum(mask, axis = 0).T", np.sum(mask, axis = 0).T)

				#mask = mask.reshape(img__.shape[0], img__.shape[1], 1)
				# print("mask", mask.shape)
				out_img = (0.4 * img__ + 0.6 * 255*(label == mask)[...,None]).astype(np.uint8)
				for j in range(extreme_points.shape[0]):
					cv2.circle(out_img, (extreme_points[j, 0], extreme_points[j, 1]), 5, cl[j], -1)
				# try:
				# 	
				# except:
				# try:
				out_name = '/Users/yezheng/github/ExtremeNet/out_extreme_medical_img_{}/{:07d}_extreme{}.png'.format(label,mask_number,label)
				# print("[gen_medical_extreme_points] out_name", out_name)
				cv2.imwrite(out_name, out_img)
			# except:
				out_name = '/home/yezheng/github/ExtremeNet/out_extreme_medical_img_{}/{:07d}_extreme{}.png'.format(label,mask_number,label)
				# print("[gen_medical_extreme_points] out_name", out_name)
				cv2.imwrite(out_name, out_img)
				# if 5 == label:
				# 	print("[gen_medical_extreme_points] out_name", out_name)
				# 	print("[gen_medical_extreme_points] extreme_points", extreme_points)
				
		    # cv2.waitKey()
		    	# if 2 == label:


