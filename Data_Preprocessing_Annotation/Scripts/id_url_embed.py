import pandas as pd


full_data = pd.read_csv("../Ted_data/result.csv")

def embed_url(s):
	return s.replace("www","embed",1)

full_data['embed_url'] = full_data['url'].apply(embed_url)

data = full_data[['Video_ID','url', 'embed_url']][full_data['Is_a_Talk?']=='Yes']


data.to_csv("id_url_embed.csv",index = False)