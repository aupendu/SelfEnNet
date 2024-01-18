import os
import sys
import glob
import torch
import torchvision
import time
import random
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms, utils
import imageio
import pickle
from PIL import Image

class CustomDataset(Dataset):
	def __init__(self, args, is_train):

		self.args = args
		self.is_train = is_train

		if self.is_train:
			self.gtdata = args.data_train_high.split('+')
			self.gt_files = []
			for f in range(len(self.gtdata)):
				self.imgpath = os.path.join(args.path, 'train', self.gtdata[f], 'high')
				self.path_bin = os.path.join(args.path, 'bin', 'train', self.gtdata[f], 'high')
				os.makedirs(self.path_bin, exist_ok=True)
				gtfiles = self._save()
				self.gt_files.extend(gtfiles)

			self.lowdata = args.data_train_low.split('+')
			self.low_files = []
			for f in range(len(self.lowdata)):
				self.imgpath = os.path.join(args.path, 'train', self.lowdata[f], 'low')
				self.path_bin = os.path.join(args.path, 'bin', 'train', self.lowdata[f], 'low')
				os.makedirs(self.path_bin, exist_ok=True)
				lowfiles = self._save()
				self.low_files.extend(lowfiles)

			self.repeat = (self.args.batch_size*self.args.update_peritr)//len(self.low_files)

		else:
			self.lowdata = args.data_test.split('+')
			self.low_files = []
			for f in range(len(self.lowdata)):
				self.imgpath = os.path.join(args.path, 'test', self.lowdata[f], 'low')
				lowfiles = glob.glob(self.imgpath+'/*.png')
				self.low_files.extend(lowfiles)

				lowfiles = glob.glob(self.imgpath+'/*.jpg')
				self.low_files.extend(lowfiles)

				lowfiles = glob.glob(self.imgpath+'/*.JPG')
				self.low_files.extend(lowfiles)

				lowfiles = glob.glob(self.imgpath+'/*.bmp')
				self.low_files.extend(lowfiles)

	def _save(self):
		files = os.listdir(self.imgpath)

		for i in range(len(files)):
			if os.path.exists(self.path_bin+'/'+files[i][:-4]+'.pt')==False:
				_img = imageio.imread(self.imgpath+'/'+files[i])
				with open(self.path_bin+'/'+files[i][:-4]+'.pt', 'wb') as _f:
					pickle.dump(_img, _f)

		return glob.glob(self.path_bin+'/*.pt')
	
	def _crop(self, img, psize):
		ih, iw = img.shape[:2]
		x0 = random.randrange(0, ih-psize+1)
		x1 = random.randrange(0, iw-psize+1)
		img = img[x0:x0+psize, x1:x1+psize, :]
		img = self._augment(img)
		return self._np2Tensor(img)

	def _augment(self, img, is_aug=True):
		hflip = is_aug and random.random() < 0.5
		vflip = is_aug and random.random() < 0.5
		rot90 = is_aug and random.random() < 0.5
		if hflip: img = img[:, ::-1, :]
		if vflip: img = img[::-1, :, :]
		if rot90: img = img.transpose(1, 0, 2)
		return img

	def _np2Tensor(self, img):
		np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
		tensor = torch.from_numpy(np_transpose).float()
		return tensor

	def __getitem__(self, index):
		if self.is_train: 

			index = self._get_index(index, 'low')
			with open(self.low_files[index], 'rb') as _f:
				LW = pickle.load(_f)
			Lbatch = self._crop(LW, self.args.patch_size)

			index = self._get_index(index, 'high')
			with open(self.gt_files[index], 'rb') as _f:
				GT = pickle.load(_f)
			GTbatch = self._crop(GT, self.args.patch_size)

			return GTbatch, Lbatch

		else:
			LW = imageio.imread(self.low_files[index])
			Lbatch = self._np2Tensor(LW)

			s_img = self.low_files[index].split('/')
			savepath = os.path.join('Results', self.args.model+ '_' +self.args.save_folder, s_img[len(s_img)-3], s_img[len(s_img)-1][:-4])
			if self.args.test_only:
				os.makedirs(os.path.join('Results', self.args.model+ '_' +self.args.save_folder, s_img[len(s_img)-3]), exist_ok=True)

				return Lbatch, 0, savepath+'.png'

			else:

				istring = self.low_files[index]
				istring = istring.replace("low", "high")

				HW = imageio.imread(istring)
				Hbatch = self._np2Tensor(HW)
				
				return Lbatch, Hbatch, savepath+'.png'

	def __len__(self): 
		if self.is_train:
			return len(self.low_files)*self.repeat
		else:
			return len(self.low_files)

	def _get_index(self, index, itype):
		if itype=='low': return index % len(self.low_files)
		if itype=='high': return index % len(self.gt_files)


class Data:
	def __init__(self, args):
		
		if not args.test_only:
			train_dataset = CustomDataset(args, is_train=True)
			self.loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)
			
		test_dataset = CustomDataset(args, is_train=False)
		self.loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.n_threads)
			
		

