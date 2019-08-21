import pystan
from simul_model1 import *

import sys
old_stdout = sys.stdout

log_file = open("message.log","w")

sys.stdout = log_file


# schools_dat = {'J': 8,
#                'y': [28,  8, -3,  7, -1,  1, 18, 12],
#                'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}
model1 = model1()
data = model1.generate(1000)

#print((data["transcript"]))

sm = pystan.StanModel(file='model1.stan')
fit = sm.sampling(data=data, iter=2000, chains=4)

la = fit.extract(permuted=True)  # return a dictionary of arrays
true_params = model1.params_dic

for param in true_params:
    
    mean_param = np.mean(la[param],axis=0)
    
    print("True "+ param+"---",true_params[param], "Estimated "+param+"---",mean_param )




# mu = la['mu']

# ## return an array of three dimensions: iterations, chains, parameters
# a = fit.extract(permuted=False)


print(fit)

sys.stdout = old_stdout

log_file.close()



fit.plot()




