import os
import random
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt


class ImageFolder(data.Dataset):
	def __init__(self, root, image_size=512, mode='train', augmentation_prob=0.4):
		"""Initializes image paths and preprocessing module."""
		self.root = root

		# GT : Ground Truth
		self.GT_paths = root[:-1]+'_GT/'
		# print("GT_paths: ",self.GT_paths)
		imgs_path = os.listdir(root)
		imgs_path.sort()
		# if mode == 'test':
		# 	print(imgs_path)
		self.image_paths = list(map(lambda x: os.path.join(root, x), imgs_path))

		self.image_size = image_size
		self.mode = mode
		# if self.mode == 'test':
		# 	print('image_paths: ',self.image_paths)
		self.RotationDegree = [0, 90, 180, 270]
		self.augmentation_prob = augmentation_prob
		print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		image_path = self.image_paths[index]

		# filename = image_path.split('_')[-1][:-len(".jpg")]
		filename = image_path.split('/')[-1][:-len(".png")]

		# if self.mode == 'test':
		# 	print("index: ", index)
		# 	print("image_path: ", self.image_paths[index])
		# 	print("filename: ", filename)

		GT_path = self.GT_paths + filename + ".png"
		# print("image_path: ", image_path)
		# print("filename: ", filename)
		# print("GT_path: ",GT_path)

		image = Image.open(image_path)
		GT = Image.open(GT_path)
		# arr = np.array(GT)
		# print('arr',arr[0])

		aspect_ratio = image.size[1]/image.size[0]  # 高宽比

		Transform = []

		ResizeRange = random.randint(300, 320)
		Transform.append(T.Resize((int(ResizeRange*aspect_ratio), ResizeRange)))
		p_transform = random.random()

		if (self.mode == 'train') and p_transform <= self.augmentation_prob:
			RotationDegree = random.randint(0, 3)
			RotationDegree = self.RotationDegree[RotationDegree]
			if (RotationDegree == 90) or (RotationDegree == 270):
				aspect_ratio = 1/aspect_ratio

			Transform.append(T.RandomRotation((RotationDegree, RotationDegree)))

			RotationRange = random.randint(-10, 10)
			Transform.append(T.RandomRotation((RotationRange, RotationRange)))

			CropRange = random.randint(250, 270)
			Transform.append(T.CenterCrop((int(CropRange*aspect_ratio), CropRange)))

			# print("Transform: ", Transform)
			Transform = T.Compose(Transform)
			# print("T.Compose(Transform): ", T.Compose(Transform))


			image = Transform(image)
			GT = Transform(GT)

			ShiftRange_left = random.randint(0, 20)
			ShiftRange_upper = random.randint(0, 20)
			ShiftRange_right = image.size[0] - random.randint(0, 20)
			ShiftRange_lower = image.size[1] - random.randint(0, 20)
			image = image.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
			GT = GT.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))

			if random.random() < 0.5:
				image = F.hflip(image)
				GT = F.hflip(GT)

			if random.random() < 0.5:
				image = F.vflip(image)
				GT = F.vflip(GT)

			Transform = T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02)

			image = Transform(image)
			# print('image: ',image)
			Transform =[]

		Transform.append(T.Resize((int(512*aspect_ratio)-int(512*aspect_ratio) % 16, 512)))
		Transform.append(T.ToTensor())
		Transform = T.Compose(Transform)

		image = Transform(image)
		GT = Transform(GT)

		Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		image = Norm_(image)
		# print("1, ",GT)
		return image, GT

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.image_paths)


def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.4):
	"""Builds and returns Dataloader."""

	dataset = ImageFolder(root=image_path, image_size=image_size, mode=mode, augmentation_prob=augmentation_prob)
	if mode == 'test':
		data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	else:
		data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


	return data_loader
