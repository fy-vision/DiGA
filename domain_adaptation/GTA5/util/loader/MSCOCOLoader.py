import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image, ImageFile
from .augmentations import *
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MSCOCOLoader(data.Dataset):
	def __init__(self, root, img_list_path, max_iters=None, crop_size=None, mean=(128, 128, 128), transform=None):
		self.n_classes = 19
		self.root = root
		self.crop_size = crop_size
		self.mean = mean
		self.transform = transform
		# self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
		self.img_ids = [i_id.strip() for i_id in open(img_list_path)]

		if not max_iters==None:
		   self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

		self.files = []
		# for split in ["train", "trainval", "val"]:
		for img_name in zip(self.img_ids):
			#img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, img_name[0]))
			img_file = osp.join(self.root,  "train/%s" % (img_name))
			self.files.append({
				"img": img_file,
				"name": img_name
			})

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		datafiles = self.files[index]

		image = Image.open(datafiles["img"]).convert('RGB')
		#name = datafiles["name"]

		# resize
		if self.crop_size != None:
			image = image.resize((self.crop_size[1], self.crop_size[0]), Image.BICUBIC)
		# transform
		if self.transform != None:
			image = self.transform(image)

		image = np.asarray(image, np.float32)

		size = image.shape
		image = image[:, :, ::-1]  # change to BGR
		image -= self.mean
		image = image.transpose((2, 0, 1)) / 128.0
		#image = image.transpose((2, 0, 1)) / 255.0

		return image.copy()

			

if __name__ == '__main__':
	dst = GTA5DataSet("./data", is_transform=True)
	trainloader = data.DataLoader(dst, batch_size=4)
	for i, data in enumerate(trainloader):
		imgs, labels = data
		if i == 0:
			img = torchvision.utils.make_grid(imgs).numpy()
			img = np.transpose(img, (1, 2, 0))
			img = img[:, :, ::-1]
			plt.imshow(img)
			plt.show()
