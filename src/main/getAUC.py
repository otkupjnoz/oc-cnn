### importing all the important libraries
import numpy as np
import h5py

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import svm

### path variables
score_path = '../../temp_files/scores.mat'
label_path = '../../temp_files/labels.mat'

with h5py.File(score_path, 'r') as f:
	test_features = f['scores'][()]
with h5py.File(label_path, 'r') as f:
	test_label = f['test_label'][()]

fpr, tpr, thresholds = metrics.roc_curve(np.transpose(test_label), np.transpose(test_features))

print metrics.auc(fpr, tpr)

