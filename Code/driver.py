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


  def setup_data_loader(self,data_loc):
    

      train_data = all_data["train"]
      dev_data= all_data["dev"]
      test_data=all_data["test"]
      
      random.shuffle(train_data)
      random.shuffle(dev_data)
      random.shuffle(test_data)

  def train_epoch(self):
    pass

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





