import os
# 解决 Windows 上 OpenMP 库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time

def lowlight(image_path, DCE_net):
	scale_factor = 12
	data_lowlight = Image.open(image_path)

	data_lowlight = (np.asarray(data_lowlight)/255.0)

	data_lowlight = torch.from_numpy(data_lowlight).float()

	h=(data_lowlight.shape[0]//scale_factor)*scale_factor
	w=(data_lowlight.shape[1]//scale_factor)*scale_factor
	data_lowlight = data_lowlight[0:h,0:w,:]
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	start = time.time()
	enhanced_image,params_maps = DCE_net(data_lowlight)

	end_time = (time.time() - start)

	print("Processing time for", os.path.basename(image_path), ":", end_time)
	# 创建结果保存路径
	result_dir = '../../selected_700_lowlight/result_Zero_DCE++/'
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
	
	# 获取文件名并保存到结果目录
	image_name = os.path.basename(image_path)
	result_path = os.path.join(result_dir, image_name)
	torchvision.utils.save_image(enhanced_image, result_path)
	return end_time

if __name__ == '__main__':

	os.environ['CUDA_VISIBLE_DEVICES']='0'
	scale_factor = 12
	
	# 加载模型（只加载一次）
	DCE_net = model.enhance_net_nopool(scale_factor).cuda()
	DCE_net.load_state_dict(torch.load('snapshots_Zero_DCE++/Epoch4.pth'))
	DCE_net.eval()

	with torch.no_grad():

		filePath = '../../selected_700_lowlight/original/'
		test_list = glob.glob(filePath + "*.jpg")
		print("Found", len(test_list), "images to process")
		sum_time = 0
		for image in test_list:
			print("Processing:", image)
			sum_time = sum_time + lowlight(image, DCE_net)

		print("Total processing time:", sum_time, "seconds")
		print("Average time per image:", sum_time/len(test_list) if len(test_list) > 0 else 0, "seconds")
		

