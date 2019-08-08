import pickle
import pandas as pd

file_path = '/Users/Shouman/Downloads/Data/Rupam_MachineLearning/Ted_data/id_url.pkl'

with open( file_path, "rb" ) as f:
	data = pickle.load(f)

df = pd.DataFrame(data=data, index=None)
df.to_csv('/Users/Shouman/Downloads/Data/Rupam_MachineLearning/Ted_data/id_url.csv', index=False)

df_1 = pd.read_csv('/Users/Shouman/Downloads/Data/Rupam_MachineLearning/Ted_data/index.csv')

result = pd.merge(df_1, df, on = 'Video_ID')

result.to_csv('/Users/Shouman/Downloads/Data/Rupam_MachineLearning/Ted_data/result.csv', index=False)
