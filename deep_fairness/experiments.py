from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import random
import pickle
import sys
import time
import copy
import numpy as np

import deep_fairness.classification_model as models
from deep_fairness.fairyted import Fairytale
from deep_fairness.helper import load_pickle, sample_indices, cvt, make_minibatch, counterfactual_loss, calc_acc, MacOSFile

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

    # assert(self.config['model_params']['in_size'] == self.config['trans_dim'] + 
    #   self.config['u_dim'] + self.config['view_dim'] + self.config['a_dim'])

    # Use appropritate device
    if self.config['gpu_num'] < 0:
      self.device = torch.device('cpu')
    else:
      self.device = torch.device('gpu:{0}'.format(self.config['gpu_num']))

    # Load fairmodel
    data_dict_path = os.path.join(self.config['input_path'], self.config['input_filename'])
    data_dict = load_pickle(data_dict_path)
    for key in data_dict:
      data_dict[key] = np.array(data_dict[key])

    if self.config['use_simulated_data']:
      self.fairmodel = Fairytale()
    else:
      self.fairmodel = Fairytale(data=data_dict)

    # Prepare the neural network and others
    self.total_epoch = 0
    self.best_loss = float('inf')
    self.model = getattr(models, self.config['model_type'])(**self.config['model_params']).to(device=self.device)
    self.loss_fn = getattr(nn, self.config['loss_function_name'])()
    self.optimizer = getattr(optim, self.config['optimizer'])(self.model.parameters(),**self.config['optimizer_params'])
    self.scheduler = getattr(optim.lr_scheduler, self.config['scheduler'])(self.optimizer, **self.config['scheduler_params'])
    self.relu = nn.ReLU()
    self.cf_loss = counterfactual_loss


  def set_random_seed(self):      
    seed = self.config['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

  def train_counterfactual_causal_model(self,fit_params_arguments):
    mf, trace = self.fairmodel.fit_params(**fit_params_arguments)
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

  def generate_counterfactual_data(self, trace, num_iter_cf):
    data_with_u, cfsample = self.fairmodel.counterfactual_generate(trace, num_iter_cf=num_iter_cf)
    orig_concat_data, cf_concat_data = self.fairmodel.create_concat_data(data_with_u, cfsample)
    cf_filename = os.path.join(self.config['output_path'],self.config['counterfactual_data_filename'])
    pickle.dump({'orig_concat_data':orig_concat_data, 'cf_concat_data':cf_concat_data}, MacOSFile(open(cf_filename,'wb')))
    return orig_concat_data, cf_concat_data

  def load_counterfactual_data(self):
    cf_filename = os.path.join(self.config['output_path'],self.config['counterfactual_data_filename'])
    data = load_pickle(cf_filename)
    orig_concat_data = data['orig_concat_data']
    cf_concat_data = data['cf_concat_data']
    return orig_concat_data, cf_concat_data

  def test_model(self, orig_concat_data, test_idx):
    self.model.eval()

    
    inputs = orig_concat_data['input'][test_idx,:]
    labels = orig_concat_data['label'][test_idx,:]
    #print(inputs)
    with torch.set_grad_enabled(False):
      outputs = self.model(inputs)
      x,total_acc = calc_acc(outputs, labels)
    #print(outputs)
    average_test_acc = np.mean(total_acc.numpy(),axis=0)
    print("Test Accuracy is :",average_test_acc)
    return x,average_test_acc


  def train_model(self, orig_concat_data, cf_concat_data, train_idx, dev_idx, 
    max_epochs=10, max_iter=10, use_cf=False, minibatch_size=10):


    
    since = time.time()

    best_model_wts = copy.deepcopy(self.model.state_dict())

    for epoch in range(max_epochs):
      # print("max_epochs:",max_epochs)
      # print("max_iter:",max_iter)
      # print("use_cf",use_cf)
      print('Epoch {}/{}'.format(epoch, max_epochs - 1))
      print('-' * 10)

      # Each epoch has a training and validation phase
      for phase in ['train', 'val']:
        if phase == 'train':
          self.model.train()  # Set model to training mode
          indices = train_idx
        else:
          self.model.eval()   # Set model to evaluate mode
          indices = dev_idx

        running_loss = 0.0

        # Iterate over data.
        for iter, a_batch in enumerate(make_minibatch(indices, minibatch_size)):
          
          if iter > max_iter:
            break

          inputs = orig_concat_data['input'][a_batch,:]
          labels = orig_concat_data['label'][a_batch,:]

          if use_cf:
            #print("hamba-------")
            cf_slice = cvt(a_batch)
            cf_inp = cf_concat_data['input'][cf_slice,:]

          # zero the parameter gradients
          self.optimizer.zero_grad()

          # forward
          # track history if only in train
          with torch.set_grad_enabled(phase == 'train'):
            outputs = self.model(inputs)
            # print('++++++++')
            # print(inputs)
            # print('--------')
            # print(outputs)
            # print('--------')
            loss = self.loss_fn(outputs, labels)

            if use_cf:
              #print("hooo=====")
              cf_outputs = self.model(cf_inp)
              # print()
              loss = loss+torch.mean(self.relu(self.cf_loss(cf_outputs,labels)))

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                self.optimizer.step()                
                # try:
                #   self.scheduler.step()
                # except:
                #   self.scheduler.step(metrics=loss)

          # statistics
          running_loss += loss.item()

        epoch_loss = running_loss / (max_iter * minibatch_size)

        print('{} Loss: {:.4f} '.format(phase, epoch_loss))

        # deep copy the model
        if phase == 'val' and epoch_loss < self.best_loss:
          self.best_loss = epoch_loss
          best_model_wts = copy.deepcopy(self.model.state_dict())

      print()

    time_elapsed = time.time() - since

    self.total_epoch += epoch 
    print('Training complete in {:.0f}m {:.0f}s (total epoch = {})'.format(
      time_elapsed // 60, time_elapsed % 60, self.total_epoch))

    # load best model weights
    self.model.load_state_dict(best_model_wts)

  def save_model(self, model_filepath):
    checkpoint = {
      'total_epoch':self.total_epoch,
      'model_state_dict':self.model.state_dict(),
      'optimizer_state_dict':self.optimizer.state_dict(),
      'scheduler_state_dict':self.scheduler.state_dict(),
      'best_loss':self.best_loss
    }
    torch.save(checkpoint, model_filepath)
    

  def load_model(self, model_filepath):
    checkpoint = torch.load(model_filepath)
    self.total_epoch = checkpoint['total_epoch']
    self.best_loss = checkpoint['best_loss']
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


  def run(self):
    '''
    Runs the experiment
    '''

    # Train causal model
    # ==================
    if self.config['train_causal_model']:
      trace = self.train_counterfactual_causal_model(self.config['fit_params_arguments'])
    else:
      trace = self.load_trace()    

    # Generate counterfactual data
    # ============================
    if self.config['generate_counterfactual']:
      orig_concat_data, cf_concat_data = self.generate_counterfactual_data(trace, 
        self.config['num_iter_cf'])
    else:
      orig_concat_data, cf_concat_data = self.load_counterfactual_data()

    # Send to appropriate device
    # ==========================
    for a_key in orig_concat_data:
      orig_concat_data[a_key] = torch.from_numpy(orig_concat_data[a_key].astype(np.float32)).to(device=self.device)
      cf_concat_data[a_key] = torch.from_numpy(cf_concat_data[a_key].astype(np.float32)).to(device=self.device)

    # Divide into train/dev/test
    # ==========================
    train_idx, dev_idx, test_idx = sample_indices(orig_concat_data['input'].shape[0])

    # Neural network training part
    # ============================
    if self.config['train_neural_network']:
      nn_filename = os.path.join(self.config['output_path'], self.config['neural_network_model_filename'])
      if self.config['load_nn_from_file']:
        self.load_model(nn_filename)
      self.train_model(orig_concat_data, cf_concat_data, train_idx, dev_idx, **self.config['trainer_params'])
      self.save_model(nn_filename)

    # Test Neural Network
    # ===================
    if self.config['test_neural_network']:
      nn_filename = os.path.join(self.config['output_path'], self.config['load_nn_filename'])
      self.load_model(nn_filename)
      op,average_test_acc = self.test_model(orig_concat_data, test_idx)
      op = op.numpy()
      inp =  orig_concat_data['input'][test_idx,:].numpy()
      data_dict_predict, data_dict_true = {},{}
      data_dict_true['transcript'] = inp[:,:200]
      data_dict_true['a'] = inp[:,200:207]
      data_dict_true['view'] = inp[:,207]
      data_dict_true['rating'] = orig_concat_data['label'][test_idx,:].numpy()
      data_dict_predict['transcript'] = inp[:,:200]
      data_dict_predict['a'] = inp[:,200:207]
      data_dict_predict['view'] = inp[:,207]
      data_dict_predict['rating'] = op
      with open('Output/test_output.pkl','wb') as f:
        pickle.dump((data_dict_predict,data_dict_true),f)


    