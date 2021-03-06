import argparse
import yaml

import os
import pymc3 as pm 
import matplotlib.pyplot as plt
import numpy as np 
import theano
import theano.tensor as tt
from scipy import stats
import pymc3 as pm 
import sys
sys.path.insert(0,'../')
from deep_fairness.simul_data import model1
from deep_fairness.fairyted import Fairytale
from deep_fairness.counterfactual_generate import counterfactual_sample
from deep_fairness.pymc_model_multivariate import model_fit
from deep_fairness.helper import load_pickle, dict_to_concat_data
from deep_fairness.experiments import Experiment

#from dummyfordiscussion import *

def data_load():
  data_dict = load_pickle('../Data/converted_data_dict.pkl')
  for key in data_dict:
    print(len(data_dict[key]),key)
    data_dict[key] = np.array(data_dict[key])
  fairmodel = Fairytale()
  mf,trace=fairmodel.fit_params()
  data_with_u, cfsample = fairmodel.counterfactual_generate(trace)
  orig_concat_data, cf_concat_data = fairmodel.create_concat_data(data_with_u, cfsample)


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
