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

clusters = [8]
s_scores = []

KM = sklearn.cluster.KMeans(n_clusters = 8).fit(mx.filled())

new_col = np.zeros((len(KM.labels_),1))
VA4feata = np.concatenate((VA, new_col), 1)

for i in range(0, len(KM.labels_)):
	VA4feata[i,3] = KM.labels_[i] + 1

new_col2 = np.zeros((len(KM.labels_),1))
VA4feat = np.concatenate((VA4feata, new_col2),1)

for i in range(0,len(KM.labels_)):
	VA4feat[i,4] = 1

df = pd.DataFrame(VA4feat, index=None, columns=["q11","q12","q13","Cluster","counter"])

df_clusters_counts = pd.pivot_table(df, values=['counter'], index=["Cluster"], aggfunc=len)
df_clusters_q11 = pd.pivot_table(df, values=['q11'], index=["Cluster"], aggfunc=np.mean)
df_clusters_q12 = pd.pivot_table(df, values=['q12'], index=["Cluster"], aggfunc=np.mean)
df_clusters_q13 = pd.pivot_table(df, values=['q13'], index=["Cluster"], aggfunc=np.mean)

q11array = []
q12array = []
q13array = []


for index,row in df_clusters_q11.iterrows():
	q11array.append(row['q11'])

for index,row in df_clusters_q12.iterrows():
	q12array.append(row['q12'])

for index,row in df_clusters_q13.iterrows():
	q13array.append(row['q13'])


# Want lowest difference in the 3 - alternatively stated, lowest max - min 


rangeArray = zip(q11array, q12array, q13array)

rangeArray2 = np.ptp(rangeArray, axis=1)
minRangeValue = np.min(rangeArray2)

rangeArray3 = rangeArray2.tolist()

indexdf = rangeArray3.index(minRangeValue)

print indexdf + 1

cunti = 1
for indexdf, row in df_clusters_counts.iterrows():
	print "Cluster", cunti, "has this many obs:", (row['counter'])
	if cunti == indexdf + 1:
		print "This is the answer!"
	cunti += 1