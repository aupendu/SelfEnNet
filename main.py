import torch
import model
from option import args
from trainer import Trainer
from dataloader import Data
import numpy as np
import random
import time

#torch.backends.cudnn.deterministic=True
#torch.backends.cudnn.benchmark = True

torch.autograd.set_detect_anomaly(True)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

def main():
	global model
	loader = Data(args)
	_model = model.Model(args)
	t = Trainer(args, loader, _model)
	if args.test_only:
		t.test()
	else:
		while not t.terminate():
			if args.level==0:
				t.train_Enhance()
				t.test_Enhance()

			if args.level==1:
				t.train_Denoise()
				t.test_Denoise()

if __name__ == '__main__':
	main()
