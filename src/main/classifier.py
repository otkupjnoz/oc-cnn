
import torch.nn as nn


relu = nn.ReLU()

class classifier_nn(nn.Module):

	def __init__(self,D):
		super(classifier_nn,self).__init__()
		self.fc1 = nn.Linear(D, 2)
		
	def forward(self,x):
		out = x
		out = self.fc1(out)
		return out