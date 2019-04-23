import numpy as np
import theano.tensor as tt
import pymc3 as pm

import seaborn as sns
import matplotlib.pyplot as plt


def gauss(theta):

	return pm.Normal(mu = theta,sd = 1)



with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sd=1)
    theta = pm.Normal('theta', mu=mu, sd=1)

    obs = 


