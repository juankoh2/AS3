from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans

import numpy as np

import pandas as pd
mydata= pd.read_csv('wine.data.csv')

X = mydata.iloc[:, 0:6]
# for temp in wine.data:
# print(wine.DESCR)
# print(wine.data[0])

Z = linkage(X,'average')
print(Z)
kclusters = fcluster(Z, 3, criterion='maxclust')
print(kclusters)

km = KMeans(n_clusters=3)
km.fit(X)
print(km.labels_)

from sklearn.metrics import accuracy_score, confusion_matrix
print (accuracy_score(kclusters,km.labels_))

plt.figure()
dendrogram(Z)
plt.show()
