import scipy
import numpy as np
import numpy.ma as ma
import pandas as pd
import sklearn
import csv
from sklearn.cluster import KMeans
from sklearn import metrics



# Import the data
data = pd.read_table('yelp_reviewers.txt', sep='|')

# Select the data that we want
VA = data[['q3','q4','q5', 'q6', 'q17', 'q18_group7']].values

# Pre-process the data:
	# Identify max average days between reviews because more frequent is lower score, but 0s should be max
maxTime = max(VA[5])
# print maxTime

	# for votes cool, funny, useful, want the average per review, not total, so replace with per review number
	# also replace 0s for time between reviews (where reviewer only had 1 review) with maxTime.
for i in range(0, len(VA)):
 	# One fucking NaN
 	if np.isnan(VA[i,4]):
 		VA[i,4] = 0 
 	VA[i,1] = VA[i,1] / VA[i,0]
 	VA[i,2] = VA[i,2] / VA[i,0]
 	VA[i,3] = VA[i,3] / VA[i,0]

 	VA[i,4] = VA[i,4] / VA[i,0]
 	# if VA[i,5] == 0:
 	# 	VA[i,5] = maxTime 


# Finding the Nan so we don't have to mask (which seems to induce some error):
# writer = csv.writer(open('noNanNever.csv', 'wb'))

# for i in range(0,len(VA)):
# 	writer.writerow(VA[i])

# Can use masked_array method, but better not to do so:
# maskarr = ma.masked_array(VA, np.isnan(VA))
# maskmean = maskarr.mean()
# mx = ma.masked_array(VA,np.isnan(VA),fill_value=0)



# Test different k values (2-20) for silhouette scores
clusters = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
s_scores = []

for i in clusters:
 	KM = sklearn.cluster.KMeans(n_clusters = clusters[i-2]).fit(VA)
 	s_scores.append(sklearn.metrics.silhouette_score(VA, KM.labels_, metric='euclidean', sample_size=4000))
 	print "The Silhouette value for", i, "centroids is:", s_scores[i-2]

# export silhouette scores at each k to csv to create MS Excel chart - will look forward to learning pyplot with more time
s_score_graph = csv.writer(open('s_graph.csv','wb'))
for i in range(0, len(s_scores)):
	s_score_graph.writerow([i+2,s_scores[i]])




# Going forward with k=6 based on elbow score, so create named representation of k=6
sixClusters = sklearn.cluster.KMeans(n_clusters = 6).fit(VA)
sixCentroids = sixClusters.cluster_centers_

# Create new VA that includes cluster labels
new_col = np.zeros((len(sixClusters.labels_),1))
VAwithlabels = np.concatenate((VA, new_col), 1)

for i in range(0, len(sixClusters.labels_)):
	VAwithlabels[i,6] = sixClusters.labels_[i] + 1

# export centroids and data with cluster labels to csv to create 2D MS Excel charts
centroids_graph = csv.writer(open('centroids_graph.csv','wb'))
for i in range(0, len(sixCentroids)):
 	centroids_graph.writerow(sixCentroids[i])
for i in range(0, len(VAwithlabels)):
	centroids_graph.writerow(VAwithlabels[i])