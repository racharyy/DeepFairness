# has to run with python 2 

import pickle
import glob

files_path = glob.glob('/Users/Shouman/Downloads/Data/Rupam_MachineLearning/Ted_data/TED_meta/*.pkl')

def find_id_url(file):
	with open( file, "rb" ) as f:
		data = pickle.load(f)
	return data['talk_meta']['url'],data['talk_meta']['id']

d = {'Video_ID':[], 'url':[]}

for file in files_path:
	url, Id = find_id_url(file)
	d['Video_ID'].append(Id)
	d['url'].append(url)

path = '/Users/Shouman/Downloads/Data/Rupam_MachineLearning/Ted_data/id_url.pkl'
pickle_out = open(path,"wb")
pickle.dump(d, pickle_out)
pickle_out.close()