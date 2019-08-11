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



# a_dim=7
# mu_a = np.zeros(a_dim)
# cov_a = np.eye(a_dim)

num_samples =1000
trans_dim = 60
u_dim = 30
rating_dim = 50
# N=100
# mu_u = np.zeros(u_dim)
# cov_u = np.eye(u_dim)
# mu_trans = np.zeros(trans_dim)
# cov_trans = np.eye(trans_dim)
# mu_rating = np.zeros(rating_dim)
# cov_rating = np.eye(rating_dim)




print('Generating samples from model')

causal_model = model1(u_dim,trans_dim,rating_dim)
data = causal_model.generate(num_samples)

# print("transcript shape ",data["transcript"].shape)
# print("view shape ",data["view"].shape)
# print("a shape ",data["a"].shape)
# print("rating shape ",data["rating"].shape)
# # mu_rand = np.random.normal(size=10)

print('Generation done')


print('Model Fitting started')

mf = model_fit(data,u_dim,'vi')
trace = mf.sample(1000)

print('Model Fitting done')

print('-------------------------------------------------------')
print('| Difference between the true and achieved parameters |')
print('-------------------------------------------------------')
for key in causal_model.params_dic:

    diff = np.linalg.norm(causal_model.params_dic[key]-np.mean(trace[key],axis=0)) #/ s
    print("Difference for ",key," is ", diff)
    

print('Generating counterfactual_sample')

#causal_model.params_dic.keys()
counterfactual_sample(data,trace,u_dim)



