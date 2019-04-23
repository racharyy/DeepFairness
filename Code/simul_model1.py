import numpy as np 
from scipy.stats import invgamma


class model1(object):
	"""docstring for model1"""
	def __init__(self):
		super(model1, self).__init__()
		params_dic = {}

		#prior probability
		params_dic["transcript0"] = np.random.normal()  
		params_dic["view0"] = np.random.normal()

		#effect of u on transcript, view and rating
		params_dic["eta_u_transcript"] = np.random.normal()
		params_dic["eta_u_view"] = np.random.normal()
		params_dic["eta_u_rating"] = np.random.normal()

		#effect of protected attiributes on transcript, view and rating
		params_dic["eta_a_transcript"] = np.random.normal(size=7)
		params_dic["eta_a_view"] = np.random.normal(size=7)
		params_dic["eta_a_rating"] = np.random.normal(size=7)

		#effect of transcript on view and rating
		params_dic["eta_transcript_view"] = np.random.normal()
		params_dic["eta_transcript_rating"] = np.random.normal()

		#effect of view on rating
		params_dic["eta_view_rating"] = np.random.normal()

		params_dic["sigma_transcript_sq"] = invgamma(1).rvs()
		params_dic["sigma_rating_sq"] = invgamma(1).rvs()

		self.params_dic = params_dic

	def rand_vect(self):
		i = np.random.randint(4)
		j = np.random.randint(3)
		a = np.zeros(7)
		a[i]=1
		a[4+j]=1
		return a


	def generate(self,num_samples=1000):

		samples = {}
		samples["N"] = num_samples
		samples["K"] = 7
		samples["a"] = []
		samples["transcript"] = []
		samples["view"] = []
		samples["rating"] = []

		params_dic = self.params_dic

		transcript0 = params_dic["transcript0"]
		view0 = params_dic["view0"]
		eta_u_transcript = params_dic["eta_u_transcript"]
		eta_u_view = params_dic["eta_u_view"]
		eta_u_rating =params_dic["eta_u_rating"]
		eta_a_transcript = params_dic["eta_a_transcript"]
		eta_a_view = params_dic["eta_a_view"]
		eta_a_rating = params_dic["eta_a_rating"]
		eta_transcript_view = params_dic["eta_transcript_view"]
		eta_transcript_rating = params_dic["eta_transcript_rating"]
		eta_view_rating = params_dic["eta_view_rating"]
		sigma_transcript_sq = params_dic["sigma_transcript_sq"]
		sigma_rating_sq = params_dic["sigma_rating_sq"]

		sigma_transcript = np.sqrt(sigma_transcript_sq)
		sigma_rating = np.sqrt(sigma_rating_sq)


		for x in xrange(num_samples):
			
			#sample u and a from 
			u = np.random.normal()
			a = self.rand_vect()


			#generate transcript, view and rating
			transcript = np.random.normal(transcript0 + eta_u_transcript * u + np.dot(a , eta_a_transcript), sigma_transcript)
			view = np.random.poisson(np.exp(view0 + eta_u_view * u + np.dot(a , eta_a_view)+transcript * eta_transcript_view));
			rating = np.random.normal(eta_u_rating * u + np.dot(a , eta_a_rating)+transcript* eta_transcript_rating+ view * eta_view_rating, sigma_rating);


			#samples["u"].append(u)
			samples["a"].append(a)
			samples["transcript"].append(transcript)
			samples["view"].append(view)
			samples["rating"].append(rating)
		#samples["a"] = np.array(samples["a"])
		return samples


