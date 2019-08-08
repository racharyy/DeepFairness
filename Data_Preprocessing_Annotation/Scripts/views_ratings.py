# has to run with python 2 

import pickle
import glob

files_path = glob.glob('/Users/Shouman/Downloads/Data/Rupam_MachineLearning/Ted_data/TED_meta/*.pkl')

d = {'Video_ID':[], 'totalviews':[], 'beautiful': [], 'funny': [], 'inspiring': [], 'ok': [], 
'fascinating': [], 'total_count': [], 'persuasive': [], 'longwinded': [], 'informative': [],
 'jaw-dropping': [], 'ingenious': [], 'obnoxious': [], 'confusing': [], 'courageous': [], 
 'unconvincing': []}

rating_words = ['beautiful', 'funny', 'inspiring', 'ok', 
'fascinating', 'total_count', 'persuasive', 'longwinded', 'informative',
 'jaw-dropping', 'ingenious', 'obnoxious', 'confusing', 'courageous', 
 'unconvincing']

def add_row(file):
	with open( file, "rb" ) as f:
		data = pickle.load(f)
	data = data['talk_meta']

	d['Video_ID'].append(data['id'])
	d['totalviews'].append(data['totalviews'])
	for word in rating_words:
		if word in data['ratings']:
			d[word].append(data['ratings'][word])
		else:
			d[word].append(0)



for file in files_path:
	add_row(file)

path = '/Users/Shouman/Downloads/Data/Rupam_MachineLearning/Ted_data/views_ratings.pkl'
pickle_out = open(path,"wb")
pickle.dump(d, pickle_out)
pickle_out.close()