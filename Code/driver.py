from __future__ import absolute_import, division, print_function

import argparse
import yaml

import csv
import logging
import os
import random
import pickle
import sys
import numpy as np

import classification_model

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
  def __init__(self, config):
    super(Experiment, self).__init__()
    self.config = config

    # Prepare the neural network and others
    self.model = getattr(classification_model, self.config['model_type'])(**self.config['model_params'])
    self.loss_fn = getattr(nn, self.config['loss_function_name'])
    self.optimizer = getattr(optim, self.config['optimizer'])(**self.config['optimizer_params'])

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

  def eval_epoch(self):
    pass


  def train(self,num_epoch):
    pass

  def save_model(self, model_filepath):
    pass

  def load_model(self, model_filepath):
    pass


  def run(self):
    '''
    Runs the experiment
    '''
    pass

	def train_epoch(self,model,optimizer):
		pass

	def eval_epoch(self,model,optimizer):
		pass

	def train(self,num_epoch,model,optimizer):
		pass


def main():
  parser = argparse.ArgumentParser('Train and Evaluate Neural Networks')
  parser.add_argument('--conf', dest='config_filepath', 
    help='Full path to the configuration file')
  args = parser.parse_args()
  
  if args.config_filepath and os.path.exists(args.config_filepath):
    conf = yaml.load(open(args.config_filepath), Loader=yaml.FullLoader)
  else:
    raise Exception('Config file not found')

  exp = Experiment(conf)
  exp.run()

if __name__=='__main__':
  main()





