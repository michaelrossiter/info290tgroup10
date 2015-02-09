import scipy
import numpy as np
import numpy.ma as ma
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn import metrics

data = pd.read_table('yelp_reviewers.txt', sep='|')

# q7_mean = np.mean(data['q7'])
# print q7_mean
# d7 = data['q7']
# d7[d7 == np.nan] = q7_mean


# q8_mean = np.mean(data['q8'])
# d8 = data['q8']
# d8[d8 == np.nan] = q8_mean

# q9_mean = np.mean(data['q9'])
# d9 = data['q9']
# d9[d9 == np.nan] = q9_mean

# r7 = pd.Series(d7, name="q7")
# r8 = pd.Series(d8, name="q8")
# r9 = pd.Series(d9, name="q9")

# VA = pd.concat([d7,d8,d9], axis=1)

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


