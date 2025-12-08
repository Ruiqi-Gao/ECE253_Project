import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random


random.seed(1143)


def populate_train_list(lowlight_images_path):




	image_list_lowlight = glob.glob(lowlight_images_path + "*.jpg")

	train_list = image_list_lowlight

	random.shuffle(train_list)

	return train_list

	

class lowlight_loader(data.Dataset):

	def __init__(self, lowlight_images_path, image_size=512):

		self.train_list = populate_train_list(lowlight_images_path) 
		self.size = image_size  # 允许自定义图片尺寸

		self.data_list = self.train_list
		print("Total training examples:", len(self.train_list))


		

	def __getitem__(self, index):

		data_lowlight_path = self.data_list[index]
		
		data_lowlight = Image.open(data_lowlight_path)
		
		# 兼容新旧版本的 Pillow：使用 LANCZOS 替代已弃用的 ANTIALIAS
		try:
			# Pillow 10.0.0+ 使用 Resampling
			data_lowlight = data_lowlight.resize((self.size,self.size), Image.Resampling.LANCZOS)
		except AttributeError:
			# 旧版本使用 LANCZOS（与 ANTIALIAS 效果相同）
			data_lowlight = data_lowlight.resize((self.size,self.size), Image.LANCZOS)
		data_lowlight = (np.asarray(data_lowlight)/255.0) 
		data_lowlight = torch.from_numpy(data_lowlight).float()

		return data_lowlight.permute(2,0,1)

	def __len__(self):
		return len(self.data_list)

