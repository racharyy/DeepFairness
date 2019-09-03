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

def ab_ret(x,y):# x=a-b, y=a/b
    b=x/(y-1)
    a= b*y
    return (a,b)


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
        inp = np.concatenate((data_dict['transcript'][i],data_dict['a'][i],[data_dict['view'][i]],data_dict['u'][i]))#data_dict['view'][i]
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
    test_indices = list(set(range(N)).difference(train_indices))
    return train_indices, dev_indices, test_indices

def cvt(ind_list, span=11):
    return np.array([range(i*span,(i+1)*span) for i in ind_list]).flatten()

def make_minibatch(list_index, minibatch_size=10):
    while True:
        np.random.shuffle(list_index)
        yield list_index[:minibatch_size]


def counterfactual_loss(cf_outputs,labels,epsilon=0.1,span=11):
    n = len(cf_outputs)

    labels = labels.repeat_interleave(span, axis=0)
    return (1.0/n)*(torch.norm(cf_outputs - labels)-epsilon)


def calc_acc(model_output, target):
    m= nn.Sigmoid()
    x= (m(model_output)>= 0.5).float()
    # print(model_output.shape)
    print(x)
    y = torch.eq(x, target).float()
    return y

import pickle

class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))











