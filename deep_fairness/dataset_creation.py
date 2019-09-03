from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric
import pandas as pd
import os
from helper import *
import matplotlib.pyplot as plt


data_frame = load_pickle('../Data/df_for_metric.pkl')
#data_frame.head()
data_frame = data_frame.iloc[:,2:]



rating_names = ['beautiful', 'confusing', 'courageous', 'fascinating', 'funny', 'informative', 'ingenious', 'inspiring', 'jaw-dropping', 'longwinded', 'obnoxious', 'ok', 'persuasive', 'unconvincing']
protected_attribute_maps = [{0.0: 'male', 1.0: 'female',2.0:'gender_other'}]
default_mappings = {
    'label_maps': [{1.0: 1, 0.0: 0}],
    'protected_attribute_maps': [{0.0: 'white', 1.0: 'black_or_african_american',2.0:'asian',3.0:'race_other'},
                                 {0.0: 'male', 1.0: 'female',2.0:'gender_other'}]
}

prot_attr_dict = {'race':{0.0: 'white', 1.0: 'black_or_african_american',2.0:'asian',3.0:'race_other'},
                                 'gender':{0.0: 'male', 1.0: 'female',2.0:'gender_other'}}

privileged_classes=[lambda x: x == 'white',lambda x: x == 'male']
#privileged_classes=['white','male']
protected_attribute_names=['race', 'gender']
unpriv_list = [[{'gender':0,'race':1},{'gender':0,'race':2},{'gender':0,'race':3}],
    [{'gender':1,'race':1},{'gender':1,'race':2},{'gender':1,'race':3}],
    [{'race':1},{'race':2},{'race':3}],
    [{'gender':1},{'gender':2}],
    [{'gender':1,'race':1},{'gender':2,'race':1}]]
priv_list = [[{'gender':0,'race':0}],
    [{'gender':1,'race':0}],
    [{'race':0}],
    [{'gender':0}],
    [{'gender':0,'race':1}]]


#Data_set_list = [(,)]

a_list, b_list = [], []
unpriv_label_list , priv_label_list = [], []
for (u,p) in zip(unpriv_list,priv_list):
    cur_a, cur_b = [], []
    for label in rating_names:
        #print('Fairness Metric for the label------>',label.upper())
    
        dataset  = StandardDataset(df=data_frame, label_name=label, favorable_classes=[1.0,1.0],
                            protected_attribute_names=protected_attribute_names, privileged_classes=privileged_classes) 
        
       
        dataset_metric = BinaryLabelDatasetMetric(dataset, unprivileged_groups=u, privileged_groups=p)
        


        diff = dataset_metric.mean_difference()
        ratio = dataset_metric.disparate_impact()
        a,b = ab_ret(diff,ratio)
        cur_a.append(a)
        cur_b.append(b)

    a_list.append(cur_a)
    b_list.append(cur_b)
        #print("For Rating",label,"Difference in mean outcomes between unprivileged and privileged groups = %f" %diff)

    
     

    unpriv_label = '+'.join(['-'.join([prot_attr_dict[key][u_el[key]] for key in u_el]) for u_el in u])
    priv_label = '+'.join(['-'.join([prot_attr_dict[key][p_el[key]] for key in p_el]) for p_el in p])

    unpriv_label_list.append(unpriv_label)
    priv_label_list.append(priv_label)

a_list = np.array(a_list)
b_list = np.array(b_list)


# for i in range(14):

#     x= a_list[:,i]
#     y= b_list[:,i]
#     plt.subplot(3,5,i+1)
#     plt.scatter(x,y)
#     xax = np.linspace(0,1,100)
#     plt.plot(xax,xax)
    

# plt.show()

for i in range(len(a_list)):

    xaxis = np.arange(14)
    width = 0.3

    fig, ax = plt.subplots()
    rects1 = ax.bar(xaxis - width/2, a_list[i], width, label=unpriv_label_list[i])
    rects2 = ax.bar(xaxis + width/2, b_list[i], width, label=priv_label_list[i])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Group Probability')
    ax.set_title('Group Fairness for different Label')
    ax.set_xticks(xaxis)
    ax.set_xticklabels(rating_names,rotation=90)
    ax.legend()
        
    fig.tight_layout() 

    plt.show()
