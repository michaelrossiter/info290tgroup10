
import scipy
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn import metrics

data = pd.read_table('yelp_reviewers.txt', sep='|')

VA = data[['q4', 'q5', 'q6']].values
clusters = [2,3,4,5,6,7,8]

s_scores = []

# Clusters = 2
KM = sklearn.cluster.KMeans(n_clusters = clusters[0]).fit(VA)
s_scores.append(sklearn.metrics.silhouette_score(VA, KM.labels_, metric='euclidean', sample_size=1000))

# Clusters = 3 
KM = sklearn.cluster.KMeans(n_clusters = clusters[1]).fit(VA)
s_scores.append(sklearn.metrics.silhouette_score(VA, KM.labels_, metric='euclidean', sample_size=1000))

# Clusters = 4 
KM = sklearn.cluster.KMeans(n_clusters = clusters[2]).fit(VA)
s_scores.append(sklearn.metrics.silhouette_score(VA, KM.labels_, metric='euclidean', sample_size=1000))

# Clusters = 5 
KM = sklearn.cluster.KMeans(n_clusters = clusters[3]).fit(VA)
s_scores.append(sklearn.metrics.silhouette_score(VA, KM.labels_, metric='euclidean', sample_size=1000))

# Clusters = 6 
KM = sklearn.cluster.KMeans(n_clusters = clusters[4]).fit(VA)
s_scores.append(sklearn.metrics.silhouette_score(VA, KM.labels_, metric='euclidean', sample_size=1000))

# Clusters = 7 
KM = sklearn.cluster.KMeans(n_clusters = clusters[5]).fit(VA)
s_scores.append(sklearn.metrics.silhouette_score(VA, KM.labels_, metric='euclidean', sample_size=1000))

# Clusters = 8 
KM = sklearn.cluster.KMeans(n_clusters = clusters[6]).fit(VA)
s_scores.append(sklearn.metrics.silhouette_score(VA, KM.labels_, metric='euclidean', sample_size=1000))


# Results:
for i in clusters:
	print "The Silhouette value for", i, "centroids is:", s_scores[i-2]


