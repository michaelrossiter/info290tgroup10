import pandas as pd
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import scipy as scipy


df = pd.read_csv('yelp_reviewers.txt',sep="|",header=0,index_col=False)

df2 = pd.concat([df['q4'], df['q5'], df['q6']], axis=1, keys=['q4','q5','q6'])

array = df2.values
features = array

whitened = whiten(features)
kmeans = 2

output = scipy.cluster.vq.kmeans(whitened,kmeans)

print output
