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



class Fairytale(object):
	"""docstring for Fairytale"""
	def __init__(self, data=None,u_dim=10):
		super(Fairytale, self).__init__()
		self.u_dim = u_dim
		if data == None:
			self.simulated = True
			num_samples =100
			trans_dim = 6
			rating_dim = 5
			print('Generating samples from model')
			self.generator = model1(u_dim,trans_dim,rating_dim)
			self.data = self.generator.generate(num_samples)			
			print('Generation done')
		else:
			self.data = data
			self.simulated = False


	def fit_params(self,check_differences=True):

		print('Model Fitting started')

		mf = model_fit(self.data,self.u_dim,'vi')
		trace = mf.sample(1000)
		
		print('Model Fitting done')

		if self.simulated == True and check_differences == True:

			print('-------------------------------------------------------')
			print('| Difference between the true and achieved parameters |')
			print('-------------------------------------------------------')
			for key in self.generator.params_dic:

			    diff = np.linalg.norm(self.generator.params_dic[key]-np.mean(trace[key],axis=0)) #/ s
			    print("Difference for ",key," is ", diff)


		return mf,trace

	def counterfactual_generate(self,trace):

		print('Generating counterfactual_sample')

		data_with_u, cfsample = counterfactual_sample(self.data,trace,self.u_dim)
		return data_with_u, cfsample

	def create_concat_data(self,data_with_u, cfsample):


		assert(data_with_u['transcript'].shape[0]==data_with_u['view'].shape[0])
		assert(data_with_u['transcript'].shape[0]==data_with_u['rating'].shape[0])
		assert(data_with_u['transcript'].shape[0]==data_with_u['a'].shape[0])
		assert(data_with_u['transcript'].shape[0]==data_with_u['u'].shape[0])


		assert(cfsample['transcript'].shape[0]==cfsample['view'].shape[0])
		assert(cfsample['transcript'].shape[0]==cfsample['rating'].shape[0])
		assert(cfsample['transcript'].shape[0]==cfsample['a'].shape[0])
		assert(cfsample['transcript'].shape[0]==cfsample['u'].shape[0])

		
		orig_concat_data, cf_concat_data = dict_to_concat_data(data_with_u), dict_to_concat_data(cfsample)
		
		return orig_concat_data, cf_concat_data

	def classify(self,model,config):
		pass





# print('Model Fitting started')

# mf = model_fit(data,u_dim,'vi')
# trace = mf.sample(1000)




    





