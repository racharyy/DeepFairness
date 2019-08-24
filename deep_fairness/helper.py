import torch
import random
import copy
import random
import math
import os
import pickle
import numpy as np
import time
import glob
import torch.nn as nn

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def dict_to_concat_data(data_dict):

    all_data_dict = {'input':[],'label':[]}

    for i in range(len(data_dict['a'])):
        inp =np.concatenate((data_dict['transcript'][i],data_dict['a'][i],[data_dict['view'][i]],data_dict['u'][i]))
        label = data_dict['rating'][i]
        all_data_dict['input'].append(inp)
        all_data_dict['label'].append(label)

    all_data_dict['input'] = np.array(all_data_dict['input'])
    all_data_dict['label'] = np.array(all_data_dict['label'])

    return all_data_dict

def sample_indices(N):
    train_len = int(0.8 * N)
    train_indices = np.random.choice( N,size = train_len,replace=False)
    dev_indices = np.random.choice( train_indices,size = int(0.1*train_len),replace=False)
    test_indices = set(range(N)).difference(train_indices)
    return train_indices, dev_indices, test_indices

def cvt(ind_list, span=110):
    return np.array([range(i*span,(i+1)*span) for i in ind_list]).flatten()

def make_minibatch(list_index, minibatch_size=10):
    while True:
        np.random.shuffle(list_index)
        yield list_index[:minibatch_size]


def counterfactual_loss(cf_outputs,labels,epsilon=0.1,span=110):
    n = len(cf_outputs)
    return (1.0/n)*(torch.norm(cf_outputs - labels)-epsilon)
















