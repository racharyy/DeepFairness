from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import pickle
import sys
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef



import torch
import torch.nn as nn
import math
import itertools
import random
import pickle
import numpy as np
import torch.optim as optim



class Experiment(object):
	"""docstring for Experiment"""
	def __init__(self, config):
		super(Experiment, self).__init__()
		self.config = config
		


	def set_random_seed(self):
	    
	    seed = self.config['seed']

	    np.random.seed(seed)
	    torch.manual_seed(seed)
	    torch.cuda.manual_seed(seed)


	def setup_data_loader(self,orig_concat_data, cf_concat_data=None):
		
		
		
		N =len(orig_concat_data['input'])
		train_len = int(0.8 * N)
		train_indices = np.random.choice( N,size = train_len,replace=False)
		dev_indices = np.random.choice( train_indices,size = int(0.1*train_len),replace=False)
		train_data_dict, dev_data_dict, test_data_dict ={},{},{}
		for key in ['input','label']:
		    train_data_dict[key] =[]
		    test_data_dict[key] = []
		    dev_data_dict[key] = []

	    for i in range(N):
		    
			inp, label  = orig_concat_data['input'][i], orig_concat_data['label'][i]
		    
		    if i in train_indices:
		        if i in dev_indices:
		            dev_data_dict['input'].append(inp)
		            dev_data_dict['label'].append(label)
		        else:
		            train_data_dict['input'].append(inp)
		            train_data_dict['label'].append(label)
		    else:
				test_data_dict['input'].append(inp)
				test_data_dict['label'].append(label)

		return train_data_dict, dev_data_dict, test_data_dict


	def prep_for_training(self):

		in_size = self.config['trans_dim']+config['u_dim']+config['view_dim']+config['a_dim']
		model = SimpleMLP(in_size = in_size, hidden_size = config['hidden_size'])
		criterion = nn.CrossEntropyLoss()#nn.BCEWithLogitsLoss()
		optimizer = optim.Adam(mlp.parameters(), weight_decay = 0.0001)
		#optimizer = optim.SGD(mlp.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 0)
		return model,optimizer


	def train_epoch(self,model,optimizer):
		pass

	def eval_epoch(self,model,optimizer):
		pass

	def train(self,num_epoch,model,optimizer):
		pass



def main(config):
	pass





