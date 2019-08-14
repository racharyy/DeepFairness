#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pymc3 as pm 
import matplotlib.pyplot as plt
import numpy as np 
import theano
import theano.tensor as tt
from scipy import stats
import pymc3 as pm 
from simul_data import *
from counterfactual_generate import *



# In[3]:


def model_fit(data,u_dim,method='mcmc'):

    #u_dim = 30
    a_dim=data['a'].shape[1]
    trans_dim = data['transcript'].shape[1]
    rating_dim = data['rating'].shape[1]

    mu_a = np.zeros(a_dim)
    cov_a = np.eye(a_dim)
    mu_u = np.zeros(u_dim)
    cov_u = np.eye(u_dim)
    mu_trans = np.zeros(trans_dim)
    cov_trans = np.eye(trans_dim)
    mu_rating = np.zeros(rating_dim)
    cov_rating = np.eye(rating_dim)

    #num_samples =1000
    
    N=100
    


    with pm.Model() as model:

        #prior sampling
        u = pm.MvNormal('u',mu=mu_u,cov=cov_u,shape=u_dim)
        transcript0 = pm.MvNormal('transcript0',mu=mu_trans,cov=cov_trans,shape=trans_dim)
        view0 = pm.Normal('view0',mu=0,tau=1)

        #effect of u 
        eta_u_transcript = pm.MatrixNormal('eta_u_transcript',colcov=cov_u, rowcov=cov_trans, shape=(trans_dim, u_dim))
        eta_u_view = pm.MvNormal('eta_u_view',mu=mu_u,cov=cov_u,shape=u_dim)
        eta_u_rating = pm.MatrixNormal('eta_u_rating',colcov=cov_u, rowcov=cov_rating, shape=(rating_dim, u_dim))
        
        #effect of protected attribute
        eta_a_transcript = pm.MatrixNormal('eta_a_transcript',colcov=cov_trans, rowcov=cov_a, shape=(a_dim, trans_dim))
        eta_a_view = pm.MvNormal('eta_a_view',mu=mu_a,cov=cov_a,shape=a_dim)
        eta_a_rating = pm.MatrixNormal('eta_a_rating',colcov=cov_rating, rowcov=cov_a, shape=(a_dim,rating_dim))

        #effect of transcript on view and rating
        eta_transcript_view = pm.MvNormal('eta_transcript_view',mu=mu_trans,cov=cov_trans,shape=trans_dim)
        eta_transcript_rating = pm.MatrixNormal('eta_transcript_rating',colcov=cov_rating, rowcov=cov_trans, shape=(trans_dim,rating_dim))

        #effect of view on rating
        eta_view_rating = pm.MvNormal('eta_view_rating',mu=mu_rating,cov=cov_rating,shape=rating_dim)


        sigma_transcript_sq = pm.InverseGamma('sigma_transcript_sq',alpha=1,beta=1)
        sigma_rating_sq = pm.InverseGamma('sigma_rating_sq',alpha=1,beta=1)
        #print(data['a'].shape)
        
        transcript_mean =   tt.dot(eta_u_transcript ,u)+transcript0 + tt.dot(data['a'] , eta_a_transcript)
        transcript = pm.MvNormal('transcript', mu= transcript_mean, cov = sigma_transcript_sq*np.eye(trans_dim), observed = data["transcript"] )
        
        
        view_mean = tt.maximum(1,view0 + tt.dot(eta_u_view , u)+   tt.dot(data['a'],eta_a_view)+tt.dot(transcript, eta_transcript_view)) 
        view = pm.Poisson('view',mu =view_mean, observed = data['view'] )

        rating_mean = tt.dot(eta_u_rating , u) +  tt.dot(data['a'],eta_a_rating) + tt.dot(transcript, eta_transcript_rating) + tt.dot(tt.reshape(view,(-1,1)), tt.reshape(eta_view_rating,(1,-1))) 
        rating = pm.MvNormal('rating', mu= rating_mean, cov = sigma_rating_sq*np.eye(rating_dim), observed = data["rating"] ) 
        


        if method == 'mcmc':   
            step = pm.NUTS()
            #init='adapt_diag'
            trace=pm.sample(100)#,step =step)#,target_accept=0.8)
        elif method == 'vi':
            trace=pm.fit(n=300)
        else:
            print('Not Implemented any other method')
            return
            
    return trace
    


# In[4]:





