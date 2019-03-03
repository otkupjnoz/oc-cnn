

### importing all the important libraries
import torch
import torchvision

import VGG_FACE_torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
import torchvision.models as models
import torchvision.transforms as transforms


import cv2
import scipy.io
from scipy import misc
from sklearn import svm
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score


import os
import sys
import copy
import h5py
import time
import pickle
import random
import argparse

import numpy as np
import cPickle as cp
# import matplotlib.pyplot as plt

from subprocess import call
from termcolor import cprint


import visdom


from classifier import *
from light_cnn import *







### frequently used variables
def AddNoise(inputs, sigma):

	noise_shape = np.shape(inputs)
	
	noise = np.random.normal(0, sigma, noise_shape)
	noise = torch.from_numpy(noise)
	noise = torch.autograd.Variable(noise).float()

	if(inputs.is_cuda):
		outputs = inputs + noise.cuda()
	else:
		outputs = inputs + noise

	return outputs

def get_fuv(hyper_para, model_type):

	## defining frequnetly used global variables
	running_loss=0.0

	inm  = nn.InstanceNorm1d(1, affine=False)
	relu = nn.ReLU()

	mean = 0.0*np.ones( (hyper_para.D,) )
	cov  = hyper_para.sigma*np.identity(hyper_para.D)

	if(model_type=='vggface'):
		imagenet_mean = np.asarray([0.485, 0.456, 0.406])
		imagenet_std  = np.asarray([0.229, 0.224, 0.225])
	else:
		imagenet_mean = np.asarray([0.36703529411, 0.410832941, 0.506612941])
		imagenet_std  = np.asarray([1.0, 1.0, 1.0])

	classifier = classifier_nn(hyper_para.D)

	return running_loss, inm, relu, mean, cov, imagenet_mean, imagenet_std, classifier

def load_dataset(dataset, class_number, mean, std, hyper_para):

	if(dataset=='abnormal'):
		# change path files according to path on your device
		# normal_data_path = '/home/labuser/Desktop/research/datasets/anomaly/normal/mat/raw/anomaly_normal_data_'+str(class_number)+'.mat'
		# abnormal_data_path = '/home/labuser/Desktop/research/datasets/anomaly/abnormal/mat/raw/anomaly_abnormal_data_'+str(class_number)+'.mat'

		with h5py.File(normal_data_path, 'r') as f:
			normal_data = f['trainData'][()]

		if(hyper_para.verbose==True):
			print('train data loaded.')
		
		with h5py.File(abnormal_data_path, 'r') as f:
			abnormal_data = f['trainData'][()]

		if(hyper_para.verbose==True):
			print('test data loaded.')

		normal_data = np.swapaxes(np.swapaxes(np.swapaxes(normal_data,2,3),1,2),0,1)
		abnormal_data = np.swapaxes(np.swapaxes(np.swapaxes(abnormal_data,2,3),1,2),0,1)
		no_abnormal_data = np.shape(abnormal_data)[0]
		no_normal_data = np.shape(normal_data)[0]

		### split the data into train and test
		rand_id = np.random.permutation(no_normal_data)
		normal_data = normal_data[rand_id,:,:,:]
		train_data = normal_data[no_abnormal_data:no_normal_data,:,:,:]
		test_data = np.concatenate( (normal_data[0:no_abnormal_data,:,:,:],abnormal_data), axis=0 )
		test_label = np.concatenate( (np.ones(no_abnormal_data,), np.zeros(no_abnormal_data,)), axis=0)

		no_train_data = np.shape(train_data)[0]
		no_test_data = np.shape(test_data)[0]
		
		for i in range(3):
			# print m[i]
			train_data[:,i,:,:] = (train_data[:,i,:,:]-mean[i])/std[i]
			test_data[:,i,:,:] = (test_data[:,i,:,:]-mean[i])/std[i]

		if(hyper_para.verbose==True):
			print('data pre-processed.')

		train_data = torch.from_numpy(train_data)
		test_data = torch.from_numpy(test_data)
	
	elif(dataset=='founder'):	
		# change path files according to path on your device
		# folder_path = '/home/labuser/Desktop/research/datasets/FounderType-200/og/'
		
		test_data_path = folder_path+'test_data_char_'+str(class_number)+'.mat'
		train_data_path = folder_path+'train_data_char_'+str(class_number)+'.mat'

		with h5py.File(train_data_path, 'r') as f:
			train_data = f['trainData'][()]

		if(hyper_para.verbose==True):
			print('train data loaded.')

		with h5py.File(test_data_path, 'r') as f:
			test_data = f['testData'][()]

		if(hyper_para.verbose==True):
			print('test data loaded.')

		train_data = np.swapaxes(np.swapaxes(np.swapaxes(train_data,2,3),1,2),0,1)
		test_data = np.swapaxes(np.swapaxes(np.swapaxes(test_data,2,3),1,2),0,1)
		

		for i in range(3):
			train_data[:,i,:,:] = (train_data[:,i,:,:]-mean[i])/std[i]
			test_data[:,i,:,:] = (test_data[:,i,:,:]-mean[i])/std[i]


		unk_full_data = np.zeros( (5000, hyper_para.img_chnl, hyper_para.img_size, hyper_para.img_size) )
		k=0
		for i in range(100, 200):
			unk_data_path = folder_path+'test_data_char_'+str(i+1)+'_trimmed.mat'
			with h5py.File(unk_data_path, 'r') as f:
				unk_data = f['testData'][()]
			unk_data = np.swapaxes(np.swapaxes(np.swapaxes(unk_data,2,3),1,2),0,1)
			no_unk_data = np.shape(unk_data)[0]
			
			for j in range(no_unk_data):
				unk_full_data[k,:,:,:] = unk_data[j,:,:,:]
				k+=1
			unk_data = None
		for j in range(3):
			unk_full_data[:,j,:,:] = (unk_full_data[:,j,:,:]-mean[j])/std[j]
		no_unk_full_data = np.shape(unk_full_data)[0]
		
		if(hyper_para.verbose==True):
			print('Unknown data loaded.')

		no_test_data = np.shape(test_data)[0]
		test_data  = np.concatenate( (test_data, unk_full_data), axis=0)
		test_label = np.concatenate( (np.ones(no_test_data,), np.zeros(no_unk_full_data,)), axis=0)
		
		if(hyper_para.verbose==True):
			print('data pre-processed.')

		no_test_data = np.shape(test_data)[0]
		no_train_data = np.shape(train_data)[0]

		train_data = torch.from_numpy(train_data)
		test_data = torch.from_numpy(test_data)

	elif(dataset=='umdface02'):
		# change path files according to path on your device
		# folder_path = '/home/labuser/Desktop/research/datasets/UMDAA-02/'
		
		test_data_path = folder_path+'test_data_user_'+str(class_number)+'.mat'
		train_data_path = folder_path+'train_data_user_'+str(class_number)+'.mat'

		with h5py.File(train_data_path, 'r') as f:
			train_data = f['trainData'][()]

		if(hyper_para.verbose==True):
			print('train data loaded.')

		with h5py.File(test_data_path, 'r') as f:
			test_data = f['testData'][()]

		if(hyper_para.verbose==True):
			print('test data loaded.')

		train_data = np.swapaxes(np.swapaxes(np.swapaxes(train_data,2,3),1,2),0,1)
		test_data = np.swapaxes(np.swapaxes(np.swapaxes(test_data,2,3),1,2),0,1)

		train_data = train_data/255.0
		test_data = test_data/255.0
		
		for i in range(3):
			train_data[:,i,:,:] = (train_data[:,i,:,:]-mean[i])/std[i]
			test_data[:,i,:,:] = (test_data[:,i,:,:]-mean[i])/std[i]


		unk_full_data = np.zeros( (2150, hyper_para.img_chnl, hyper_para.img_size, hyper_para.img_size) )
		k=0
		for i in range(0, 48):
			if(int(i+1)!=int(class_number)):
				unk_data_path = folder_path+'test_data_user_'+str(i+1)+'_trimmed.mat'
				with h5py.File(unk_data_path, 'r') as f:
					unk_data = f['testData'][()]
				unk_data = np.swapaxes(np.swapaxes(np.swapaxes(unk_data,2,3),1,2),0,1)
				unk_data = unk_data/255.0
				no_unk_data = np.shape(unk_data)[0]
				
				for j in range(no_unk_data):
					unk_full_data[k,:,:,:] = unk_data[j,:,:,:]
					k+=1
				unk_data=None

		if(hyper_para.verbose==True):
			print('Unknown data loaded.')

		if(hyper_para.verbose==True):
			print('Data pre-processing...')

		for j in range(3):
			unk_full_data[:,j,:,:] = (unk_full_data[:,j,:,:]-mean[j])/std[j]
		no_unk_full_data = np.shape(unk_full_data)[0]

		no_test_data = np.shape(test_data)[0]
		test_data  = np.concatenate( (test_data, unk_full_data), axis=0)
		test_label = np.concatenate( (np.ones(no_test_data,), np.zeros(no_unk_full_data,)), axis=0)
		
		no_test_data = np.shape(test_data)[0]
		no_train_data = np.shape(train_data)[0]

		if(hyper_para.verbose==True):
			print('data pre-processed.')

		train_data = torch.from_numpy(train_data)
		test_data = torch.from_numpy(test_data)
	
	else:
		raise argparse.ArgumentTypeError('Invalid Dataset Name')

	return train_data, test_data, test_label

def choose_network(model_type, pre_trained_flag):


	if(model_type=='alexnet'):
		model = torchvision.models.alexnet(pretrained=pre_trained_flag)
		new_classifier = nn.Sequential(*list(model.classifier.children())[:-2])
		model.classifier = new_classifier
	elif(model_type=='vgg16'):
		model = torchvision.models.vgg16(pretrained=pre_trained_flag)
		new_classifier = nn.Sequential(*list(model.classifier.children())[:-3])
		model.classifier = new_classifier
	elif(model_type=='vgg19'):
		model = torchvision.models.vgg19(pretrained=pre_trained_flag)
		new_classifier = nn.Sequential(*list(model.classifier.children())[:-3])
		model.classifier = new_classifier
	elif(model_type=='vgg16bn'):
		model = torchvision.models.vgg16_bn(pretrained=pre_trained_flag)
		new_classifier = nn.Sequential(*list(model.classifier.children())[:-3])
		model.classifier = new_classifier
	elif(model_type=='vgg19bn'):
		model = torchvision.models.vgg19_bn(pretrained=pre_trained_flag)
		new_classifier = nn.Sequential(*list(model.classifier.children())[:-3])
		model.classifier = new_classifier
	elif(model_type=='vggface'):
		model = VGG_FACE_torch.VGG_FACE_torch
		model.load_state_dict(torch.load('VGG_FACE.pth'))
		model = model[:-3]
	elif(model_type=='lightcnn'):
		model = LightCNN_29Layers_v2(num_classes=80013)
		model = torch.nn.DataParallel(model)
		model.load_state_dict(torch.load('LightCNN_29Layers_V2_checkpoint.pth')['state_dict'])
		new_model=nn.Sequential(*list(model.module.children())[:-1])
	else:
		raise argparse.ArgumentTypeError('models supported in this version of code are alexnet, vgg16, vgg19, vgg16bn, vgg19bn. \n Enter model_type as one fo this argument')

	return model

def choose_classifier(dataset, class_number, model_type, model, classifier, D, hyper_para, train_data, test_data, test_label, no_train_data, no_test_data, inm, relu, m, s):

	if(hyper_para.verbose==True):
		print('Extracting features.....')

	train_features = np.memmap('../../temp_files/train_features_temp.bin', dtype='float32', mode='w+', shape=(no_train_data,hyper_para.D))
	train_features = torch.from_numpy(train_features)

	for i in range(no_train_data):
		temp = model(torch.autograd.Variable(train_data[i:(i+1)].cuda().contiguous().float())).float()
		temp = temp.view(1,1,hyper_para.D)
		temp = inm(temp)
		temp = relu(temp.view(hyper_para.D))
		train_features[i:(i+1)] = temp.data.cpu()
	train_data = None

	if(hyper_para.verbose==True):
		print('Features extracted.')

	## test on the test set
	test_features = np.memmap('../../temp_files/test_features_temp.bin', dtype='float32', mode='w+', shape=(no_test_data,hyper_para.D))
	test_scores   = np.memmap('../../temp_files/test_scores_temp.bin', dtype='float32', mode='w+', shape=(no_test_data,1))
	test_features = torch.from_numpy(test_features)

	if(hyper_para.verbose==True):
		print('Computing test scores and AUC....')

	area_under_curve=0.0
	if(hyper_para.classifier_type=='OC_CNN'):
		test_scores   = torch.from_numpy(test_scores)
		k=0
		print np.shape(test_features)
		start = time.time()
		for j in range(no_test_data):
			temp = model(AddNoise(torch.autograd.Variable(test_data[j:(j+1)].cuda().contiguous().float()), hyper_para.sigma1)).float()
			temp = temp.view(1,1,hyper_para.D)
			temp = inm(temp)
			temp = temp.view(hyper_para.D)
			
			test_features[k:(k+1)] = temp.data.cpu()
			test_scores[k:(k+1)]   = classifier(relu(temp)).data.cpu()[1]
			# print(classifier(relu(temp)).data.cpu())
			
			k = k+1
		end = time.time()
		print(end-start)
		test_scores    = test_scores.numpy()
		test_features  = test_features.numpy()
		train_features = train_features.numpy()

		test_scores = (test_scores-np.min(test_scores))/(np.max(test_scores)-np.min(test_scores))

	elif(hyper_para.classifier_type=='OC_SVM_linear'):
		# train one-class svm
		oc_svm_clf = svm.OneClassSVM(kernel='linear', nu=float(hyper_para.N))
		oc_svm_clf.fit(train_features.numpy())
		k=0
		mean_kwn = np.zeros( (no_test_data,1) )
		for j in range(no_test_data):
			temp = model(torch.autograd.Variable(test_data[j:(j+1)].cuda().contiguous().float())).float()
			temp = temp.view(1,1,hyper_para.D)
			temp = inm(temp)
			temp = temp.view(hyper_para.D)			
			test_features[k:(k+1)] = temp.data.cpu()
			temp 				   = np.reshape(relu(temp).data.cpu().numpy(), (1, hyper_para.D))
			test_scores[k:(k+1)]   = oc_svm_clf.decision_function(temp)[0]

			k = k+1

		test_features  = test_features.numpy()
		train_features = train_features.numpy()

		joblib.dump(oc_svm_clf,'../../save_folder/saved_models/'+dataset+'/classifier/'+str(class_number) +'/'+
																				model_type+'_OCCNNlin'    +'_'+
																				str(hyper_para.iterations)+'_'+
																				str(hyper_para.lr)		  +'_'+
																				str(hyper_para.sigma)	  +'_'+
																				str(hyper_para.N)         +'.pkl')

	fpr, tpr, thresholds = metrics.roc_curve(test_label, test_scores)

	if(hyper_para.verbose==True):
		print('Test scores and AUC computed.')

	return area_under_curve, train_features, test_scores, test_features

def choose_method(dataset, model_type, class_number, hyper_para):

	auc=0.0
	if(hyper_para.method=='OC_CNN'):
		auc = OC_CNN(dataset, model_type, class_number, hyper_para)
	elif(hyper_para.method=='OC_SVM_linear'):
		auc = OC_SVM_linear(dataset, model_type, class_number, hyper_para)
	elif(hyper_para.method=='Bi_SVM_linear'):
		auc = Bi_SVM_linear(dataset, model_type, class_number, hyper_para)
	elif(hyper_para.method=='SVDD'):
		print('look at matlab code')
	elif(hyper_para.method=='SMPM'):
		print('look at matlab code')
	else:
		raise argparse.ArgumentTypeError('model_type argument can be only one of these OC_CNN, OC_SVM_linear, Bi_SVM_linear')

	return auc

def OC_CNN(dataset, model_type, class_number, hyper_para):

	running_loss, inm, relu, mean, cov, imagenet_mean, imagenet_std, classifier = get_fuv(hyper_para, model_type)

	if(hyper_para.verbose==True):
		print('Loading dataset '+dataset+'...')

	train_data, test_data, test_label = load_dataset(dataset, class_number, imagenet_mean, imagenet_std, hyper_para)

	if(hyper_para.verbose==True):
		print(dataset+' dataset loaded.')

	no_train_data = np.shape(train_data.numpy())[0]
	no_test_data  = np.shape(test_data.numpy())[0]

	### choose one network which produces D dimensional features
	if(hyper_para.verbose==True):
		print('Loading network '+hyper_para.model_type+'...')
	
	model = choose_network(model_type, hyper_para.pre_trained_flag)

	if(hyper_para.verbose==True):
		print('Network '+hyper_para.model_type+' loaded.')

	running_cc = 0.0
	running_ls = 0.0

	if(hyper_para.gpu_flag):
		inm.cuda()
		relu.cuda()
		model.cuda()
		classifier.cuda()
	
	model.train()
	classifier.train()
	
	### optimizer for model training (for this work we restrict to only fine-tuning FC layers)
	if(model_type=='vggface'):
		model_optimizer      = optim.Adam(model[-5:].parameters(), lr=hyper_para.lr)
	else:
		model_optimizer      = optim.Adam(model.classifier.parameters(), lr=hyper_para.lr)
	classifier_optimizer = optim.Adam(classifier.parameters(), lr=hyper_para.lr)
	
	# loss functions
	cross_entropy_criterion = nn.CrossEntropyLoss()

	for i in range(int(hyper_para.iterations)):
	# for i in range(int(hyper_para.iterations*no_train_data/hyper_para.batch_size)):
		# print i
		rand_id = np.asarray(random.sample( range(no_train_data), int(hyper_para.batch_size)))
		rand_id = torch.from_numpy(rand_id)

		# get the inputs
		inputs = torch.autograd.Variable(train_data[rand_id].cuda()).float()
		
		# get the labels
		labels = np.concatenate( (np.zeros( (int(hyper_para.batch_size),) ), np.ones( (int(hyper_para.batch_size),)) ), axis=0)
		labels = torch.from_numpy(labels)
		labels = torch.autograd.Variable(labels.cuda()).long()
		
		gaussian_data = np.random.normal(0, hyper_para.sigma, (int(hyper_para.batch_size), hyper_para.D))
		gaussian_data = torch.from_numpy(gaussian_data)

		# forward + backward + optimize
		out1 = model(AddNoise(inputs, hyper_para.sigma1))

		out1 = out1.view(int(hyper_para.batch_size), 1, hyper_para.D)
		out1 = inm(out1)
		out1 = out1.view(int(hyper_para.batch_size), hyper_para.D)
		out2 = torch.autograd.Variable(gaussian_data.cuda()).float()
		out  = torch.cat( (out1, out2),0)
		out  = relu(out)
		out  = classifier(out)
		
		# zero the parameter gradients
		model_optimizer.zero_grad()
		classifier_optimizer.zero_grad()
		 
		cc = cross_entropy_criterion(out, labels) 
		loss = cc
		
		loss.backward()

		model_optimizer.step()
		classifier_optimizer.step()
		
		# print statistics
		running_cc += cc.data
		running_loss += loss.data

		if(hyper_para.verbose==True):
			if (i % (hyper_para.stats_freq) == (hyper_para.stats_freq-1)):    # print every stats_frequency batches
				line = hyper_para.BLUE   + '[' + str(format(i+1, '8d')) + '/'+ str(format(int(hyper_para.iterations), '8d')) + ']' + hyper_para.ENDC + \
					hyper_para.GREEN  + ' loss: '     + hyper_para.ENDC + str(format(running_loss/hyper_para.stats_freq, '1.8f'))  + \
					hyper_para.GREEN  + ' cc: '     + hyper_para.ENDC + str(format(running_cc/hyper_para.stats_freq, '1.8f'))
				print(line)
				running_loss = 0.0
				running_cc = 0.0
			
	classifier.eval()
	model.eval()
	relu.eval()

	area_under_curve, train_features, test_scores, test_features = choose_classifier(dataset, class_number, model_type, model, classifier, D, hyper_para, train_data, test_data, test_label, no_train_data, no_test_data, inm, relu, imagenet_mean, imagenet_std)

	classifier.cpu()
	model.cpu()
	relu.cpu()
	
	torch.save(model,'../../save_folder/saved_models/'+dataset+'/model/'+str(class_number)+'/'+model_type +'_'+
																				str(hyper_para.iterations)+'_'+
																				str(hyper_para.lr)		  +'_'+
																				str(hyper_para.sigma)	  +'.pth')
	
	torch.save(model,'../../save_folder/saved_models/'+dataset+'/classifier/'+str(class_number)+'/'+model_type +'_'+
																					 str(hyper_para.iterations)+'_'+
																					 str(hyper_para.lr)		   +'_'+
																					 str(hyper_para.sigma)     +'.pth')

	scipy.io.savemat('../../save_folder/results/'+dataset+'/'+ str(class_number)  +'/'+ model_type	+'_OCCNN123_'+
							 str(hyper_para.iterations)  +'_'+ str(hyper_para.lr) +'_'+ str(hyper_para.sigma)	 +'.mat',
								{'auc':area_under_curve, 'train_features':train_features, 'test_scores':test_scores,
														 'test_features':test_features,   'test_label':test_label    })

	if(hyper_para.verbose==True):
		print('model, classifier, features and results saved.')

	return area_under_curve


def OC_SVM_linear(dataset, model_type, class_number, hyper_para):

	_, _, relu, mean, cov, imagenet_mean, imagenet_std, _ = get_fuv(hyper_para, model_type)

	if(hyper_para.verbose==True):
		print('Loading dataset '+dataset+'...')

	train_data, test_data, test_label = load_dataset(dataset, class_number, imagenet_mean, imagenet_std, hyper_para)

	if(hyper_para.verbose==True):
		print(dataset+' dataset loaded.')

	no_train_data = np.shape(train_data.numpy())[0]
	no_test_data  = np.shape(test_data.numpy())[0]

	### choose one network which produces D dimensional features
	model = choose_network(model_type, hyper_para.pre_trained_flag)

	### training on gpu
	if(hyper_para.gpu_flag):
		relu.cuda()
		model.cuda()
	
	model.eval()
	relu.eval()

	if(hyper_para.verbose==True):
		print('Extracting training features...')

	train_features = np.memmap('../../temp_files/train_features_temp.bin', dtype='float32', mode='w+', shape=(no_train_data,hyper_para.D))
	train_features = torch.from_numpy(train_features)

	for i in range(no_train_data):
		train_features[i:(i+1)] = (model(torch.autograd.Variable(train_data[i:(i+1)].cuda().contiguous().float(), volatile=True)).float()).data.cpu()
	train_data = None

	if(hyper_para.verbose==True):
		print('Features extracted.')

	if(hyper_para.verbose==True):
		print('Training one class SVM with linear kernel...')
			
	# train one-class svm
	oc_svm_clf = svm.OneClassSVM(kernel='linear', nu=float(hyper_para.N))
	# oc_svm_clf.fit(train_features)
	oc_svm_clf.fit(train_features.numpy())

	if(hyper_para.verbose==True):
		print('One class SVM with Linear kernel trained.')

	## test on the test set
	test_features = np.memmap('../../temp_files/test_features_temp.bin', dtype='float32', mode='w+', shape=(no_test_data,hyper_para.D))
	test_scores   = np.memmap('../../temp_files/test_scores_temp.bin', dtype='float32', mode='w+', shape=(no_test_data,1))
	test_features = torch.from_numpy(test_features)

	k=0
	mean_kwn = np.zeros( (no_test_data,1) )
	for j in range(no_test_data):
		temp = (model(torch.autograd.Variable(test_data[j:(j+1)].cuda().contiguous().float(), volatile=True)).float())
		test_features[k:(k+1)] = temp.data.cpu()
		temp 				   = np.reshape((temp).data.cpu().numpy(), (1, hyper_para.D))
		test_scores[k:(k+1)]   = oc_svm_clf.decision_function(temp)[0]
		
		k = k+1

	test_features  = test_features.numpy()
	train_features = train_features.numpy()

	fpr, tpr, thresholds = metrics.roc_curve(test_label, test_scores)
	
	area_under_curve = metrics.auc(fpr, tpr)
	
	joblib.dump(oc_svm_clf,'../../save_folder/saved_models/'+dataset+'/classifier/'+str(class_number)+'/'+model_type+'_OCSVMlin_'+str(hyper_para.N)+'.pkl')

	scipy.io.savemat('../../save_folder/results/'+dataset+'/'+str(class_number) +'/'+ model_type+'_OCSVMlin_'+str(hyper_para.N)+'.mat',
													{ 'train_features':train_features, 'test_features':test_features, 'test_label':test_label, 'test_scores':test_scores    })

	return area_under_curve


def Bi_SVM_linear(dataset, model_type, class_number, hyper_para):

	_, _, relu, mean, cov, imagenet_mean, imagenet_std, _ = get_fuv(hyper_para)

	train_data, test_data, test_label = load_dataset(dataset, class_number, imagenet_mean, imagenet_std, hyper_para)

	no_train_data = np.shape(train_data.numpy())[0]
	no_test_data  = np.shape(test_data.numpy())[0]

	### choose one network which produces D dimensional features
	model = choose_network(model_type, hyper_para.pre_trained_flag)

	### training on gpu
	if(hyper_para.gpu_flag):
		relu.cuda()
		model.cuda()
	
	model.eval()
	relu.eval()

	train_features = np.memmap('../../temp_files/train_features_temp.bin', dtype='float32', mode='w+', shape=(no_train_data,hyper_para.D))
	train_features = torch.from_numpy(train_features)

	for i in range(no_train_data):
		train_features[i:(i+1)] = (model(torch.autograd.Variable(train_data[i:(i+1)].cuda().contiguous().float(), volatile=True)).float()).data.cpu()
	train_data = None

	# binary svm
	oc_svm_clf = svm.SVC(kernel='linear', C=float(hyper_para.N))
	labels = np.concatenate( (np.ones( (no_train_data,) ), np.zeros( (no_train_data,)) ), axis=0)
	gaussian_temp=np.random.normal(0,hyper_para.sigma, (no_train_data, hyper_para.D))
	train_temp=np.concatenate( (train_features, gaussian_temp), axis=0)
	oc_svm_clf.fit(train_temp, labels)

	## test on the test set
	test_features = np.memmap('../../temp_files/test_features_temp.bin', dtype='float32', mode='w+', shape=(no_test_data,hyper_para.D))
	test_scores   = np.memmap('../../temp_files/test_scores_temp.bin', dtype='float32', mode='w+', shape=(no_test_data,1))
	test_features = torch.from_numpy(test_features)

	k=0
	mean_kwn = np.zeros( (no_test_data,1) )
	for j in range(no_test_data):
		temp = model(torch.autograd.Variable(test_data[j:(j+1)].cuda().contiguous().float(), volatile=True)).float()
		test_features[k:(k+1)] = temp.data.cpu()
		temp 				   = np.reshape(temp.data.cpu().numpy(), (1, hyper_para.D))
		test_scores[k:(k+1)]   = oc_svm_clf.decision_function(temp)[0]
		
		k = k+1

	test_features  = test_features.numpy()
	train_features = train_features.numpy()

	fpr, tpr, thresholds = metrics.roc_curve(test_label, test_scores)

	area_under_curve = metrics.auc(fpr, tpr)

	joblib.dump(oc_svm_clf,'../../save_folder/saved_models/'+dataset+'/classifier/'+str(class_number)+'/'+model_type+'_BiSVMlin_'+str(hyper_para.N)+'.pkl')

	scipy.io.savemat('../../save_folder/results/'+dataset +'/'+str(class_number)+'/'+ model_type+'_BiSVMlin_'+str(hyper_para.N)+'.mat',
													{'auc':area_under_curve, 'train_features':train_features, 'test_scores':test_scores,
																			  'test_features':test_features,  'test_label':test_label    })

	return area_under_curve