import pymc3 as pm 
import matplotlib.pyplot as plt
import numpy as np 
import theano
import theano.tensor as tt
from scipy import stats
import pymc3 as pm 
from simul_data import *
from counterfactual_generate import *
from pymc_model_multivariate import *
from helper import *

from fairyted import *
#from dummyfordiscussion import *

data_dict = load_pickle('../Data/converted_data_dict.pkl')
for key in data_dict:
	print(len(data_dict[key]),key)	
	data_dict[key] = np.array(data_dict[key])
Fairmodel = Fairytale()
mf,trace=Fairmodel.fit_params()
data_with_u, cfsample = Fairmodel.counterfactual_generate(trace)
orig_concat_data, cf_concat_data = Fairmodel.create_concat_data(data_with_u, cfsample)