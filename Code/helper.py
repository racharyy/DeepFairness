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