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
<<<<<<< HEAD
#from dummyfordiscussion import *

# a_dim=7
# mu_a = np.zeros(a_dim)
# cov_a = np.eye(a_dim)

# num_samples =1000
# trans_dim = 60
# u_dim = 30
# rating_dim = 50
# N=100
# mu_u = np.zeros(u_dim)
# cov_u = np.eye(u_dim)
# mu_trans = np.zeros(trans_dim)
# cov_trans = np.eye(trans_dim)
# mu_rating = np.zeros(rating_dim)
# cov_rating = np.eye(rating_dim)




class Fairytale(object):
	"""docstring for Fairytale"""
	def __init__(self, data=None,u_dim=1):
		super(Fairytale, self).__init__()
		self.u_dim = u_dim
		if data == None:
			self.simulated = True
			num_samples =1000
			trans_dim = 60
			rating_dim = 50
			print('Generating samples from model')
			self.generator = model1(u_dim,trans_dim,rating_dim)
			self.data = self.generator.generate(num_samples)			
			print('Generation done')
		else:
			self.data = data
			self.simulated = False


	def fit_params(self,check_differences=True):

		print('Model Fitting started')

		mf = model_fit(self.data,self.u_dim,'mcmc')
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

	def counterfactual_generate(self,data):

		print('Generating counterfactual_sample')

		#causal_model.params_dic.keys()
		mf,trace = self.fit_params()
		cfsample = counterfactual_sample(data,trace,self.u_dim)
		return cfsample

	def classify(self,model,config):
		pass
=======
from fairyted import *
#from dummyfordiscussion import *

>>>>>>> ed39994694335a2ed78a95120c30cb6fd6047280


data_dict = load_pickle('../Data/converted_data_dict.pkl')
for key in data_dict:
	print(len(data_dict[key]),key)	
	data_dict[key] = np.array(data_dict[key])
Fairmodel = Fairytale()
mf,trace=Fairmodel.fit_params()
data_with_u, cfsample = Fairmodel.counterfactual_generate(trace)
orig_concat_data, cf_concat_data = Fairmodel.create_concat_data(data_with_u, cfsample)