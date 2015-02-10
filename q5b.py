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


df_clusters_q11 = pd.pivot_table(df, values=['q11'], index=["Cluster"], aggfunc=np.mean)

df_clusters_q12 = pd.pivot_table(df, values=['q12'], index=["Cluster"], aggfunc=np.mean)

df_clusters_q13 = pd.pivot_table(df, values=['q13'], index=["Cluster"], aggfunc=np.mean)

# Very funny but useless reviewers are where mean funny (q12) - mean useful (q13) is largest

print df_clusters_q12, df_clusters_q13

q12array = []
q13array = []
for index,row in df_clusters_q12.iterrows():
	q12array.append(row['q12'])


for index,row in df_clusters_q13.iterrows():
	q13array.append(row['q13'])

qdiffarray = [q12array - q13array for q12array,q13array in zip(q12array, q13array) ]

maxfunnydiff = np.max(qdiffarray)

print "The funniest but most useless cluster is cluster number", qdiffarray.index(maxfunnydiff) + 1






