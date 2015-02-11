import math
import scipy
import numpy as np
import numpy.ma as ma
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn import metrics

# Strategy for q6a:
	# build array with necessary data field (q8-q13, q16, log of q17),
	# mask the array to clean NaN, inf, and -inf values
	# then run Kmeans and silhouette score at k=5

data = pd.read_table('yelp_reviewers.txt', sep='|')

VA = data[['q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q16a', 'q16b', 'q16c', 'q16d', 'q16e', 'q16f', 'q16g', 'q16h', 'q16i', 'q17']].values

# getting natural log of q17
for record in VA:
  if record[-1] != 0:
    record[-1] = math.log(record[-1])

maskarr = ma.masked_array(VA, np.isnan(VA))
maskmean = maskarr.mean()
mx = ma.masked_array(VA,np.isnan(VA),fill_value=maskmean)

k = 5
s_score = 0
KM = sklearn.cluster.KMeans(n_clusters = k).fit(mx.filled())
s_score = sklearn.metrics.silhouette_score(mx.filled(), KM.labels_, metric='euclidean', sample_size=1000)
print "The Silhouette value for", k, "centroids is:", s_score


# Part B

VA_B = data[['q14']].values

maskarr = ma.masked_array(VA_B, np.isnan(VA_B))
maskmean = maskarr.mean()
mx = ma.masked_array(VA_B,np.isnan(VA_B),fill_value=maskmean)

new_col = np.zeros((len(KM.labels_),1))
VA_Bclusternum = np.concatenate((VA_B, new_col), 1)

for i in range(0, len(KM.labels_)):
	VA_Bclusternum[i,1] = KM.labels_[i] + 1

df = pd.DataFrame(VA_Bclusternum, columns=["q14","Cluster"])
df_clusters = pd.pivot_table(df, values=['q14'], index=["Cluster"], aggfunc=np.mean)
print df_clusters




