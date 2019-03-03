
import sys
import argparse

from utils import *

import numpy as np

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1','True'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0','False'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def str2float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac

parser = argparse.ArgumentParser()

# Optional arguments
parser.add_argument(
	"--dataset",
	default='founder',
	help="select one of the dataset founder abnormal umdface02"
)

parser.add_argument(
	"--lr",
	default=1e-4,
	type=str2float,
	help="set learning rate (default recommended)"
)

parser.add_argument(
	"--gpu_flag",
	default=True,
	type=str2bool,
	help="change to False if you want to train on CPU (Seriously??)"
)

parser.add_argument(
	"--verbose",
	default=True,
	type=str2bool,
	help="if want to display inbetween stats"
)

parser.add_argument(
	"--iterations",
	default=220,
	type=int,
	help="number of iterations (its not epochs)"
)

parser.add_argument(
	"--sigma",
	default=1.00,
	type=str2float,
	help="gaussian standard deviation parameter"
)

parser.add_argument(
	"--model_type",
	default='alexnet',
	help="choose alexnet or vggnet"
)

parser.add_argument(
	"--method",
	default='OC_CNN',
	help="choose one of the following \n-OC_CNN \n-OC_SVM_linear \n-Bi_SVM_linear"
)

parser.add_argument(
	"--default",
	default=True,
	type=str2bool,
	help="run with default parameters (when used other arguments are useless)"
)

parser.add_argument(
	"--class_number",
	default=1,
	type=int,
	help="choose class of interest for one class"
)

parser.add_argument(
	"--batch_size",
	default=64,
	type=int,
	help="choose class of interest for one class"
)

parser.add_argument(
	"--pre_trained_flag",
	default=True,
	type=str2bool,
	help="choose btween random pretrained weights"
)

parser.add_argument(
	"--N",
	default=0.5,
	type=str2float,
	help="value between 0-1"
)

parser.add_argument(
	"--model_mode",
	default='train',
	help="train or eval"
)

parser.add_argument(
	"--classifier_type",
	default='OC_CNN',
	help="choose one of the following \n-OC_CNN \n-OC_SVM_linear"
)

args = parser.parse_args()

### Parse the argument
dataset = args.dataset
model_type = args.model_type
class_number = args.class_number

### import parameters for respective datasets
sys.path.append('../parameters/'+dataset+'/')
from parameters import *

if(args.default==False):
	hyper_para.lr 				= float(args.lr)
	hyper_para.gpu_flag 		= args.gpu_flag
	hyper_para.iterations 		= int(args.iterations)
	hyper_para.method 			= args.method
	hyper_para.verbose 			= args.verbose
	hyper_para.sigma 			= float(args.sigma)
	hyper_para.pre_trained_flag = args.pre_trained_flag
	hyper_para.N 				= args.N
	hyper_para.gamma 			= float(args.gamma)
	hyper_para.model_type		= model_type
	hyper_para.classifier_type 	= args.classifier_type
	hyper_para.batch_size    	= args.batch_size

auc=0.0

if(args.model_mode=='train'):
	auc = choose_method(dataset, model_type, class_number, hyper_para)
elif(args.model_mode=='eval'):
	print('work under progress')
	# auc = choose_models(dataset, model_type, class_number, hyper_para)
else:
	raise argparse.ArgumentTypeError('model type can be either train or eval')


print auc