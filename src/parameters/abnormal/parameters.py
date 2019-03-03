
import numpy as np

### Setting hyperparameters
class hyperparameters():
	def __init__(self):
		self.batch_size                  = 64
		self.iterations                  = 1000
		self.lr							 = 1e-4
		self.sigma 						 = 0.01
		self.sigma1						 = 0.00000000000000000000000000000001
		self.D                           = 4096
		self.N 							 = 0.5
		self.gamma 						 = float(1/4096.0)

		self.stats_freq 				 = 1
		self.img_chnl 					 = 3
		self.img_size 					 = 224

		self.gpu_flag 					 = True
		self.verbose 					 = False
		self.pre_trained_flag			 = True
		self.intensity_normalization	 = False

		self.model_type 				 = 'alexnet'
		self.method 					 = 'OC-CNN'
		self.classifier_type			 = 'OC-CNN'



### Setting print colors
class bcolors:
	HEADER	  = '\033[95m'
	BLUE	  = '\033[94m'
	GREEN	  = '\033[92m'
	YELLOW 	  = '\033[93m'
	FAIL	  = '\033[91m'
	ENDC	  = '\033[0m'
	BOLD      = '\033[1m'
	UNDERLINE = '\033[4m'



hyper_para = hyperparameters()
colors     = bcolors()