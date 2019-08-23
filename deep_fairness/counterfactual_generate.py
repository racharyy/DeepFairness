import pymc3 as pm 
import matplotlib.pyplot as plt
import numpy as np 
import theano
import theano.tensor as tt
from scipy import stats


def counterfactual_sample(data,trace,u_dim,num_extra_unobserved=10, num_iter_cf=10):
    
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
    
    num_original_sample = data['view'].shape[0]
    print('Number of Original Data: ----> ', num_original_sample)
    
    mu_u, cov_u = np.zeros(u_dim), np.eye(u_dim)
    
    transcript0 = trace["transcript0"].mean(axis=0)
    view0 = trace["view0"].mean(axis=0)
    eta_u_transcript =trace["eta_u_transcript"].mean(axis=0)
    eta_u_view = trace["eta_u_view"].mean(axis=0)
    eta_u_rating =trace["eta_u_rating"].mean(axis=0)
    eta_a_transcript = trace["eta_a_transcript"].mean(axis=0)
    eta_a_view = trace["eta_a_view"].mean(axis=0)
    eta_a_rating = trace["eta_a_rating"].mean(axis=0)
    eta_transcript_view = trace["eta_transcript_view"].mean(axis=0)
    eta_transcript_rating = trace["eta_transcript_rating"].mean(axis=0)
    eta_view_rating = trace["eta_view_rating"].mean(axis=0)
    sigma_transcript_sq = trace["sigma_transcript_sq"].mean(axis=0)
    sigma_rating_sq = trace["sigma_rating_sq"].mean(axis=0)
    u_list = []
    
    #ABDUCTION part
    
   
    
    
    
    
    with pm.Model() as model:

        #prior sampling
        u = pm.MvNormal('u',mu=mu_u,cov=cov_u,shape=(num_original_sample,u_dim))
        #u = pm.MvNormal('u',mu=mu_mult,cov=cov_mult,shape=num_original_sample)

        transcript_mean =   tt.dot(u,np.transpose(eta_u_transcript))+transcript0 + tt.dot(data['a'] , eta_a_transcript)
        transcript = pm.MvNormal('transcript', mu= transcript_mean, cov = sigma_transcript_sq*np.eye(trans_dim), observed = data['transcript'] )
        
        
        view_mean = tt.maximum(1,view0 + tt.dot(u,np.transpose(eta_u_view))+   tt.dot(data['a'],eta_a_view)+tt.dot(transcript, eta_transcript_view)) 
        view = pm.Poisson('view',mu =view_mean, observed = data['view'] )

        rating_mean = tt.dot(u,np.transpose(eta_u_rating)) +  tt.dot(data['a'],eta_a_rating) + tt.dot(transcript, eta_transcript_rating) + tt.dot(tt.reshape(view,(-1,1)), tt.reshape(eta_view_rating,(1,-1))) 
        rating = pm.MvNormal('rating', mu= rating_mean, cov = sigma_rating_sq*np.eye(rating_dim), observed = data["rating"] ) 
        
        u_post_mf = pm.fit(n=num_iter_cf)
        new_trace = u_post_mf.sample(num_extra_unobserved)
        u_list=new_trace['u']
        #print(u_list.shape)
    
    data_with_u ={}
    data_with_u['transcript'] = np.repeat(data['transcript'],num_extra_unobserved,axis=0)
    data_with_u['view'] = np.repeat(data['view'],num_extra_unobserved,axis=0)
    data_with_u['rating'] = np.repeat(data['rating'],num_extra_unobserved,axis=0)
    data_with_u['a'] = np.repeat(data['a'],num_extra_unobserved,axis=0)
    data_with_u['u'] = u_list.transpose([1,0,2]).reshape(-1,u_dim)





        
    #ACTION part
    new_data ={} 
    num_counter_fact_a =11
    num_repeat = num_counter_fact_a*num_extra_unobserved

    u_temp, a_temp = [], []
    for i in range(num_original_sample):
        #print("size of u is ",np.array(u_list[:,i,:]).shape)
        cur_u = np.repeat(np.array(u_list[:,i,:]),num_counter_fact_a,axis=0)
        #print(cur_u)
        u_temp.extend(cur_u)
        cur_a = data['a'][i]
        other_a = []
        #print(cur_a)
        x = np.nonzero(cur_a)
        #print(x[0])
        for i in range(3):
            for j in range(3,7):
                a= np.zeros(7)                
                if i!=x[0][0] or j!=x[0][1]:
                    a[i]=1
                    a[j]=1
                    other_a.append(a)

        #a_temp = np.concatenate(a_temp,other_a)
        #print(np.array(other_a).shape)
        other_a = np.repeat([other_a],num_extra_unobserved,axis=0).reshape(-1,7)
        a_temp.extend(other_a)

    new_data['transcript'] = np.repeat(data['transcript'],num_repeat,axis=0)
    new_data['view'] = np.repeat(data['view'],num_repeat,axis=0)
    new_data['rating'] = np.repeat(data['rating'],num_repeat,axis=0)
    

    new_data['u'] = np.array(u_temp)
    new_data['a'] = np.array(a_temp) 



    # for key in new_data:
    #     print(key,"----",new_data[key].shape)
    
    
#     #PREDICTION part
#     new_data["rating"] = np.array([])


    print("Number of Orginal Data with u: ----> ", data_with_u['a'].shape[0])
    print("Number of counterfactual Data: ----> ", new_data['a'].shape[0])
            
    return data_with_u, new_data


