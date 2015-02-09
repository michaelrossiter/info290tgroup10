import scipy
import numpy as np
import numpy.ma as ma
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn import metrics

data = pd.read_table('yelp_reviewers.txt', sep='|')

VA = data[['q11','q12','q13']].values

maskarr = ma.masked_array(VA, np.isnan(VA))
maskmean = maskarr.mean()

mx = ma.masked_array(VA,np.isnan(VA),fill_value=maskmean)

clusters = [2,3,4,5,6,7,8]

s_scores = []

for i in clusters:
	KM = sklearn.cluster.KMeans(n_clusters = clusters[i-2]).fit(mx.filled())
	s_scores.append(sklearn.metrics.silhouette_score(mx.filled(), KM.labels_, metric='euclidean', sample_size=1000))
	print "The Silhouette value for", i, "centroids is:", s_scores[i-2]

print KM.labels_
