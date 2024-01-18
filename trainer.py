import os
import sys
from math import exp
import torch
import torchvision
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import pickle
from PIL import Image
import math
import random
import numpy as np
from common import Image_Quality_Metric
import time
from decimal import Decimal
import imageio
import copy


import torchvision.models as models




class Noise_Smooth_4N(nn.Module):
	def __init__(self):
		super(Noise_Smooth_4N,self).__init__()
		self.alpha = 1.2 #1.2

	def forward(self, x):

		batch_size = x.size()[0]
		h_x = x.size()[2]
		w_x = x.size()[3]
		count_h =  (x.size()[2]-2) * x.size()[3]
		count_w = x.size()[2] * (x.size()[3] - 2)

		V, _ = torch.max(x, dim=1, keepdim=True)
		V = torch.log(V+1e-12)
		
		Vh_tv1 = torch.abs(V[:,:,2:,:]-V[:,:,1:h_x-1,:])
		Vw_tv1 = torch.abs(V[:,:,:,2:]-V[:,:,:,1:w_x-1])

		Vh_tv2 = torch.abs(V[:,:,:h_x-2,:]-V[:,:,1:h_x-1,:])
		Vw_tv2 = torch.abs(V[:,:,:,:w_x-2]-V[:,:,:,1:w_x-1])

		Vh_tv1 = torch.pow(Vh_tv1, self.alpha) + 0.0001
		Vw_tv1 = torch.pow(Vw_tv1, self.alpha) + 0.0001

		Vh_tv2 = torch.pow(Vh_tv2, self.alpha) + 0.0001
		Vw_tv2 = torch.pow(Vw_tv2, self.alpha) + 0.0001

		Vh_tv1 = torch.reciprocal(Vh_tv1)
		Vw_tv1 = torch.reciprocal(Vw_tv1)

		Vh_tv2 = torch.reciprocal(Vh_tv2)
		Vw_tv2 = torch.reciprocal(Vw_tv2)

		h_tv1 = torch.pow((x[:,:,2:,:]-x[:,:,1:h_x-1,:]),2)
		w_tv1 = torch.pow((x[:,:,:,2:]-x[:,:,:,1:w_x-1]),2)

		h_tv2 = torch.pow((x[:,:,:h_x-2,:]-x[:,:,1:h_x-1,:]),2)
		w_tv2 = torch.pow((x[:,:,:,:w_x-2]-x[:,:,:,1:w_x-1]),2)

		h_tv1 = torch.mul(h_tv1, Vh_tv1).sum()
		w_tv1 = torch.mul(w_tv1, Vw_tv1).sum()

		h_tv2 = torch.mul(h_tv2, Vh_tv2).sum()
		w_tv2 = torch.mul(w_tv2, Vw_tv2).sum()
		return (h_tv1/count_h+w_tv1/count_w+h_tv2/count_h+w_tv2/count_w)/batch_size


class Trainer():
	def __init__(self, args, my_loader, my_model):
		self.args = args
		if not args.test_only:
			self.loader_train = my_loader.loader_train
			self.optimizer = torch.optim.Adam(my_model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
			self.decay = args.decay.split('+')
			self.decay = [int(i) for i in self.decay] 
			self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.decay, gamma=args.gamma)

		self.loader_test = my_loader.loader_test
		self.model = my_model

		self.metric = Image_Quality_Metric()

		self.noise_smooth = Noise_Smooth_4N()
		self.MSE = nn.MSELoss()

		os.makedirs('model_dir/'+self.args.exp, exist_ok=True)

		self.valpsnr = 0
		self.bestepoch = 0

		
		if args.resume:
			checkpoint = torch.load(args.trained_model)
			self.model.load_state_dict(checkpoint['model'])
			if args.resume:
				self.optimizer.load_state_dict(checkpoint['optimizer'])
				self.scheduler.load_state_dict(checkpoint['scheduler'])
				self.valpsnr = checkpoint['bestpsnr']
				self.bestepoch = checkpoint['bestepoch']
				print(self.scheduler.last_epoch+1)

		if args.pre_train or args.test_only:
			if args.pre_train: checkpoint = torch.load(args.pretrained_model)
			if args.test_only: checkpoint = torch.load(args.trained_model)
			pretrained_dict = checkpoint['model']
			new_state_dict = self.model.state_dict()
			for name, param in pretrained_dict.items():
				if name in new_state_dict:
					input_param = new_state_dict[name]
					if input_param.shape == param.shape:
						input_param.copy_(param)


	def Enhance(self, img, r):
		enhance_image = torch.pow(img, r)
		return enhance_image

	def train_Enhance(self):
		self.model.train()
		Indentity_R, Continual_R, Gamma_R = 0, 0, 0
		lr = self.optimizer.param_groups[0]['lr']
		print('===========================================\n')
		print('[Epoch: %d] [Learning Rate: %.2e]'%(self.scheduler.last_epoch+1, Decimal(lr)))
		startepoch = time.time()
		loss_name = ['R-Idty', 'R-Cont', 'R-GCons']
		for i_batch, (GTbatch, Lowbatch) in enumerate(self.loader_train):

			Lowbatch, GTbatch = Lowbatch/(255.0/self.args.rgb_range), GTbatch/(255.0/self.args.rgb_range)
			Lowbatch, GTbatch = Lowbatch.cuda(), GTbatch.cuda()

			kx = 0.75 
			_Lowbatch = self.Enhance(Lowbatch, kx)

			self.optimizer.zero_grad()

			# ------- Forward Pass -------

			high_enhance, r_enhance, _ = self.model((Lowbatch, 'enhance_'+str(self.args.level)))
			_, r_continual, _ = self.model((high_enhance, 'enhance_'+str(self.args.level)))

			high_GT, r_identity, _ = self.model((GTbatch, 'enhance_'+str(self.args.level)))
			_high_enhance, _r_enhance, _ = self.model((_Lowbatch, 'enhance_'+str(self.args.level)))


			# ------- Loss Calculation -------

			loss_continual_r = torch.mean(torch.pow(r_continual - 1, 2))
			loss_identity_r = torch.mean(torch.pow(r_identity - 1, 2))
			loss_gamma_r = torch.mean(torch.pow(r_enhance - kx*_r_enhance, 2))

			loss_gt_mse = torch.mean(torch.pow(GTbatch - high_GT, 2))
			loss_mse = torch.mean(torch.pow(high_enhance - _high_enhance, 2))
			loss =   1e-2*loss_identity_r  + 1e-2*loss_continual_r + 1*loss_gamma_r               


			loss.backward()
			self.optimizer.step()

			Indentity_R += loss_identity_r.data.cpu()
			Continual_R += loss_continual_r.data.cpu()
			Gamma_R += loss_gamma_r.data.cpu()

			if (i_batch+1)%100==0:
				all_losses = [Indentity_R/100, Continual_R/100, Gamma_R/100]
				total_time = time.time()-startepoch
				startepoch = time.time()
				istring = ''
				for ks in range(len(all_losses)):
					istring = istring+'['+loss_name[ks]+' %0.4f] '%(all_losses[ks])
				print('[Batch: %d] %s[Time: %.1f s]'%(i_batch+1, istring, total_time))
				Indentity_R, Continual_R, Gamma_R = 0, 0, 0
		self.scheduler.step()


	def train_Denoise(self):
		self.model.train()
		smooth_IE, Indentity_I, LowIdentity_I = 0, 0, 0
		lr = self.optimizer.param_groups[0]['lr']
		print('===========================================\n')
		print('[Epoch: %d] [Learning Rate: %.2e]'%(self.scheduler.last_epoch+1, Decimal(lr)))
		startepoch = time.time()
		loss_name = ['ILow_TV', 'High-Idty', 'Low-Idty']
		for i_batch, (GTbatch, Lowbatch) in enumerate(self.loader_train):

			Lowbatch, GTbatch = Lowbatch/(255.0/self.args.rgb_range), GTbatch/(255.0/self.args.rgb_range)
			Lowbatch, GTbatch = Lowbatch.cuda(), GTbatch.cuda()

			self.optimizer.zero_grad()

			# ------- Forward Pass -------

			high_enhance, r, low_denoised = self.model((Lowbatch, 'denoise_'+str(self.args.level)))
			_, _, denoised_identity = self.model((GTbatch, 'denoise_'+str(self.args.level)))

			# ------- Loss Calculation -------

			loss_Ismooth_TV = self.noise_smooth(low_denoised) 
			loss_identity_i = self.MSE(denoised_identity, GTbatch)
			loss_identity_low = self.MSE(low_denoised, Lowbatch)

			loss =   1e-1*loss_Ismooth_TV + 1e-1*loss_identity_low + 1*loss_identity_i 

			loss.backward()
			self.optimizer.step()

			smooth_IE += loss_Ismooth_TV.data.cpu()
			Indentity_I += loss_identity_i.data.cpu()
			LowIdentity_I += loss_identity_low.data.cpu()
			

			if (i_batch+1)%100==0:
				all_losses = [smooth_IE/100, Indentity_I/100, LowIdentity_I/100]
				total_time = time.time()-startepoch
				startepoch = time.time()
				istring = ''
				for ks in range(len(all_losses)):
					istring = istring+'['+loss_name[ks]+' %0.4f] '%(all_losses[ks])
				print('[Batch: %d] %s[Time: %.1f s]'%(i_batch+1, istring, total_time))
				smooth_IE, Indentity_I, LowIdentity_I = 0, 0, 0
		self.scheduler.step()



	def test_Enhance(self):
		torch.set_grad_enabled(False)
		self.model.eval()
		ipsnr = 0
		for i_batch, (LowImage, HighImage, _) in enumerate(self.loader_test):

			LowImage = LowImage.cuda()
			LowImage = LowImage/(255.0/self.args.rgb_range)

			torch.cuda.empty_cache()
			with torch.no_grad():
				high_out, r, _ = self.model((LowImage, 'enhance_'+str(self.args.level)))

			HighImage = HighImage/(255.0/self.args.rgb_range)
			ipsnr += self.metric.psnr(high_out.cpu(), HighImage)


		torch.set_grad_enabled(True)
		ipsnr = ipsnr/len(self.loader_test)

		if  not self.args.test_only:
			if ipsnr>self.valpsnr:
				self.valpsnr = ipsnr
				self.bestepoch = self.scheduler.last_epoch
				torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 
							'scheduler': self.scheduler.state_dict(), 'bestepoch': self.bestepoch, 'bestpsnr': self.valpsnr,
							}, 'model_dir/'+self.args.exp+'/'+self.args.model+'_best_Enhance_'+str(self.args.level)+'.pth.tar')

			torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 
						'scheduler': self.scheduler.state_dict(), 'bestepoch': self.bestepoch, 'bestpsnr': self.valpsnr,
						}, 'model_dir/'+self.args.exp+'/'+self.args.model+'_checkpoint_Enhance_'+str(self.args.level)+'.pth.tar')

		print('Test PSNR: %f [Best PSNR: %f at %d]'%(ipsnr, self.valpsnr, self.bestepoch))


	def test_Denoise(self):
		torch.set_grad_enabled(False)
		self.model.eval()
		ipsnr = 0
		for i_batch, (LowImage, HighImage, _) in enumerate(self.loader_test):

			LowImage = LowImage.cuda()
			LowImage = LowImage/(255.0/self.args.rgb_range)

			torch.cuda.empty_cache()
			with torch.no_grad():
				high_out, r, _ = self.model((LowImage, 'denoise_'+str(self.args.level)))

			HighImage = HighImage/(255.0/self.args.rgb_range)
			ipsnr += self.metric.psnr(high_out.cpu(), HighImage)

		torch.set_grad_enabled(True)
		ipsnr = ipsnr/len(self.loader_test)

		if  not self.args.test_only:
			if ipsnr>self.valpsnr:
				self.valpsnr = ipsnr
				self.bestepoch = self.scheduler.last_epoch
				torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 
							'scheduler': self.scheduler.state_dict(), 'bestepoch': self.bestepoch, 'bestpsnr': self.valpsnr,
							}, 'model_dir/'+self.args.exp+'/'+self.args.model+'_best_Denoise_1.pth.tar')

			torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 
						'scheduler': self.scheduler.state_dict(), 'bestepoch': self.bestepoch, 'bestpsnr': self.valpsnr,
						}, 'model_dir/'+self.args.exp+'/'+self.args.model+'_checkpoint_Denoise_1.pth.tar')

		print('Test PSNR: %f [Best PSNR: %f at %d]'%(ipsnr, self.valpsnr, self.bestepoch))


	def test(self):
		torch.set_grad_enabled(False)
		self.model.eval()
		for i_batch, (LowImage, _, savepath) in enumerate(self.loader_test):

			LowImage = LowImage.cuda()
			LowImage = LowImage/(255.0/self.args.rgb_range)

			torch.cuda.empty_cache()
			with torch.no_grad():
				if self.args.level==0: high_out, r, _ = self.model((LowImage, 'enhance_0'))
				if self.args.level==1: high_out, r, _ = self.model((LowImage, 'denoise_1'))

			if self.args.test_only:
				output = torch.round((255.0/self.args.rgb_range)*torch.clamp(high_out, 0, self.args.rgb_range))
				output = torch.squeeze(output).cpu()
				output = np.uint8(output.numpy().transpose((1, 2, 0)))
				imageio.imwrite(savepath[0], output)

		torch.set_grad_enabled(True)


	def terminate(self):
		if self.args.test_only:
			self.test()
			return True
		else:
			epoch = self.scheduler.last_epoch
			return epoch >= self.args.epochs


