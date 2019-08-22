from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import random
import pickle
import sys
import numpy as np

import deep_fairness.classification_model as models
from deep_fairness.fairyted import Fairytale
from deep_fairness.helper import load_pickle

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

    assert(self.config['model_params']['in_size'] == self.config['trans_dim'] + 
      self.config['u_dim'] + self.config['view_dim'] + self.config['a_dim'])

    # Prepare the neural network and others
    self.model = getattr(models, self.config['model_type'])(**self.config['model_params'])
    self.loss_fn = getattr(nn, self.config['loss_function_name'])
    self.optimizer = getattr(optim, self.config['optimizer'])(self.model.parameters(),
      **self.config['optimizer_params'])

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


  def train_counterfactual_causal_model(self, fairytale_params):
    # Read data
    fairytale_params = {akey:aval for }
    data_dict_path = os.path.join(self.config['input_path'], self.config['input_filename'])
    data_dict = load_pickle(data_dict_path)
    for key in data_dict:
      data_dict[key] = np.array(data_dict[key])

    # Train fairmodel
    self.fairmodel = Fairytale(**fairytale_params)
    mf, trace = self.fairmodel.fit_params()

    # Save the trace for future use
    trace_filename = os.path.join(self.config['output_path'], self.config['causal_model_filename'])
    pickle.dump({'trace':trace}, open(trace_filename, 'wb'))
    return trace

  def load_trace(self):
    trace_filename = os.path.join(self.config['output_path'], self.config['causal_model_filename'])
    if not os.path.exists(trace_filename):
      raise Exception('Trace file not found. Run Experiment.train_counterfactual_causal_model first')
    trace = load_pickle(trace_filename)['trace']
    return trace

  def generate_counterfactual_data(self, trace):
    data_with_u, cfsample = self.fairmodel.counterfactual_generate(trace)
    orig_concat_data, cf_concat_data = self.fairmodel.create_concat_data(data_with_u, cfsample)    
    return orig_concat_data, cf_concat_data


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
    if self.config['train_causal_model']:
      trace = self.train_counterfactual_causal_model(self.config['fairytale_params'])
    else:
      trace = self.load_trace()
    orig_concat_data, cf_concat_data = self.generate_counterfactual_data(trace)
    train, dev, test = self.setup_data_loader(orig_concat_data, cf_concat_data)
      

  def train_epoch(self,model,optimizer):
    pass

  def eval_epoch(self,model,optimizer):
    pass

  def train(self,num_epoch,model,optimizer):
    pass

