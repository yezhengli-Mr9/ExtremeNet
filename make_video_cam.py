import cv2, glob, scipy 
import numpy as np
import sys
import os
from scipy import signal
from PIL import Image
from tqdm import tqdm
import copy


if '__main__' == __name__:
	# L = os.listdir('temp/')
	# for n in L:
	# 	os.remove(os.path.join('temp', n))
	print("sys.argv", 	sys.argv)
	if 1 < len(sys.argv):
		# __path = os.path.join(os.getcwd(), str(sys.argv[1]))
		__path = str(sys.argv[1])
		new_filename = str(sys.argv[1])
		type_output_num = len(sys.argv) -1
		
		output_path = "output_test_comparison"+str(type_output_num)+".avi"
	else: 
		print("ERROR: yezheng: input path")
		exit(1)
	
	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	font                   = cv2.FONT_HERSHEY_SIMPLEX
	bottomLeftCornerOfText = (10,30)
	fontScale              = 1
	fontColor              = (128,128,128)
	lineType               = 2

	print("os.path.join(__path,'/*')", __path)
	img_list = glob.glob(__path+'/Cam_On_Image_*.png')
	# print("img_list", img_list)

	# img_list = [path_name for path_name in img_list if '.png' ==path_name[-4:] and not 'z' == path_name.split('/')[-1][0]]

	img_list = [path_name for path_name in img_list if '.png' ==path_name[-4:] and not 'z' == path_name.split('/')[-1][0]]

	# print("img_list", img_list)
	img_list.sort()
	# img_list = img_list[]
	# print("img_list", img_list)
	# if 1 == type_output_num:
	# 	output_path = os.path.join(__path,'output_zhang7.avi')
	# elif 2 == type_output_num:
	# 	output_path = os.path.join(__path,output_name) #'output_zhang7.avi'
	output_path = os.path.join(new_filename,output_path)
	print("output_path", output_path)

	# out = cv2.VideoWriter(output_path, fourcc, 20.0, (256 * int(type_output_num/int(type_output_num/3)), 256*int(type_output_num/3)))#(256*, 256)
	out = cv2.VideoWriter(output_path, fourcc, 20.0, (256 * type_output_num, 256))#(256*, 256)
	count = 0

	for img_path_idx in tqdm(range(len(img_list))):
		img_path = img_list[img_path_idx]
		# print("img_path.split('/')[-1][:-4]", img_path.split('/')[-1][:-4])
		splits = img_path.split('/')[-1].split('_')[3]#[:-4].split('_')[0].split('e')
		# print("[make_video] splits", splits, splits.split())
		
		index = int(''.join([s for s in splits if s.isdigit()]))

		# print("[make_video] index", index)
		# if "Frame"== splits[0]:
		# 	1 == img 
		img = cv2.imread(img_path)
		# print("index",index)
		# cv2.imshow("img", img)
		# cv2.waitKey()
		# if index >10:
		# 	exit()
		if None is img:
			print("img_path", img_path)
		else:
			# print("img_path", img_path)
			pass
		if np.max(img) <10:
			
			img = (1 == img) *1.0
			# print("img.shape", img.shape, "np.unique(img)", np.unique(img), np.sum(img[...,0] == img[...,1]) )
			img = 250*img.astype(np.float64)
		# print("img",np.sum(img)/120)
		img = cv2.resize(img, #dsize = img_comparison.shape
			(256,256))
		# if 1 == type_output_num:
		# 	text = '{:05d}'.format(index)
		# else:
		# 	text = '{:05d}(new)'.format(index)
		text = __path.split('_')[2] +"({})".format(index)
		
		cv2.putText(img,text, 
			bottomLeftCornerOfText, 
			font, 
			fontScale,
			fontColor,
			lineType)


		# print("img_comparison", np.max(img_comparison), np.min(img_comparison),img_comparison.shape)
		# print("img",img.shape, "img_comparison",img_comparison.shape)
		# try:
		# 	del new_img
		# except:
		# 	pass
		new_img = copy.deepcopy(img)
		new_img = img
		# del img
		for i in range(2,type_output_num+1):
				# compare_filename = "/Users/yezheng/medical_img/maskrcnn_single_img/clip_num_4/iter0/"+"{:05d}.png".format(index)
			# compare_prefix = '/Users/yezheng/medical_img/data/test/mask_label/iter0'
			compare_prefix = sys.argv[i]#'/Users/yezheng/medical_img/box_result/test/clip_num_4'
			img_com_list = glob.glob(os.path.join(compare_prefix,"Cam_On_Image_*.png"))
			img_com_list.sort()
			img_com_list = [n for n in img_com_list if "{:07d}".format(index) in n]
			if 0 == len(img_com_list):
				img_com_list = glob.glob(os.path.join(compare_prefix,"Cam_On_Image_*.png"))
				img_com_list = [n for n in img_com_list if "{:04d}".format(index) in n]
			img_com_list.sort()
			# print('[make_video] img_com_list', img_com_list)
			if 0 == len(img_com_list):
				compare_filename  = None
				img_comparison = np.zeros((256,256,3))+255
			else:
				compare_filename = img_com_list[0]#compare_prefix+'/{:05d}.png'.format(index)#"/Frame{:04d}_ordered.png".format(index)
				# print("compare_filename", compare_filename)
				img_comparison = cv2.imread(compare_filename)
				if (256,256) is not img_comparison.shape:
					img_comparison = cv2.resize(img_comparison, (256,256))
				if np.max(img_comparison) <10:
					if i == 3:
						img_comparison = (i == img_comparison) *1.0
					if i == 2:
						img_comparison = (blood_idx == img_comparison) *1.0
					if i == 4:
						img_comparison = (2 == img_comparison) *1.0
					img_comparison = 120*img_comparison#.astype(np.int64)
			if img_comparison is None:
				count += 1
				print("yezheng's Exception", img_path, img.shape, img_comparison,"index", index, "compare_filename", compare_filename)
				# if count >10:
				# 	break
				continue
			
			text = compare_prefix.split('_')[2]
			cv2.putText(img_comparison,text,  bottomLeftCornerOfText, 
				font, 
				fontScale,
				fontColor,
				lineType)
			try:
				new_img = np.concatenate((img_comparison, new_img),axis=1)
			except Exception as e: 
				print("Exception", e, img.shape, img_comparison.shape,compare_filename)
				continue
			del img_comparison
		# print("new_img", new_img.shape)	
		# new_img= np.uint8(new_img)
		out.write(np.uint8(new_img))
		# cv2.imshow("new_img", new_img)
		# cv2.waitKey()
	out.release()
