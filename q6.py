import scipy
import numpy as np
import numpy.ma as ma
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn import metrics



# Strategy for q6:
	# build array with necessary data field (q8-q13, q16, log of q17),
	# mask the array to clean NaN, inf, and -inf values
	# then run Kmeans and silhouette score at k=5


data = pd.read_table('yelp_reviewers.txt', sep='|')

VA = data[['q7','q8','q9']].values

maskarr = ma.masked_array(VA, np.isnan(VA))
maskmean = maskarr.mean()

mx = ma.masked_array(VA,np.isnan(VA),fill_value=maskmean)

clusters = [2,3,4,5,6,7,8]

s_scores = []

for i in clusters:
	KM = sklearn.cluster.KMeans(n_clusters = clusters[i-2]).fit(mx.filled())
	s_scores.append(sklearn.metrics.silhouette_score(mx.filled(), KM.labels_, metric='euclidean', sample_size=1000))
	print "The Silhouette value for", i, "centroids is:", s_scores[i-2]


