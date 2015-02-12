import math
import scipy
import numpy as np
import numpy.ma as ma
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn import metrics

data = pd.read_table('yelp_reviewers.txt', sep='|')
VA = data[['q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q16a', 'q16b', 'q16c', 'q16d', 'q16e', 'q16f', 'q16g', 
'q16h', 'q16i', 'q17', 'q18_group2', 'q18_group3', 'q18_group5', 'q18_group6', 'q18_group7','q18_group11',
'q18_group13','q18_group14','q18_group15','q18_group16_a','q18_group16_b','q18_group16_c',
'q18_group16_d','q18_group16_e','q18_group16_f','q18_group16_g','q18_group16_h']].values

# Getting natural log of q17
for record in VA:
  if record[15] != 0:
    record[15] = math.log(record[15])

# Getting rid of infinities
n = VA.shape[1]
for i in range(n):
	VA = VA[VA[...,i] != float("inf")]

# Masking/filling
maskarr = ma.masked_array(VA, np.isnan(VA))
maskmean = maskarr.mean()
mx = ma.masked_array(VA,np.isnan(VA),fill_value=maskmean)

silhouettes = []
for i in range(2, 9):
	Kmeans = sklearn.cluster.KMeans(n_clusters=i).fit(mx.filled())
	silhouette = metrics.silhouette_score(mx.filled(), Kmeans.labels_, sample_size=1000)
	silhouettes.append(silhouette)
	print "k=",i,": ", silhouette

best_k = silhouettes.index(max(silhouettes)) + 2
print "Choose k=", best_k



# Part B

print "Part B: Within cluster variance"
wcvs = []
for i in range(2, 9):
	Kmeans = sklearn.cluster.KMeans(n_clusters=i).fit(mx.filled())
	wcv = Kmeans.inertia_
	wcvs.append(wcv)
	print "k=",i,": ", wcv
	