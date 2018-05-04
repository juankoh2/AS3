from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

from sklearn import datasets
wine = datasets.load_wine()
X = wine.data

import pandas as pd
mydata= pd.read_csv('full.csv')

Y = mydata.iloc[:, 0]
# for temp in wine.data:
# print(wine.DESCR)
# print(wine.data[0])

Z = linkage(X,'complete')
# print(Z)
kclusters = fcluster(Z, 3, criterion='maxclust')
print(kclusters)

km = KMeans(n_clusters=3)
km.fit(X)
print(km.labels_)


from sklearn.metrics import accuracy_score, confusion_matrix
print (accuracy_score(Y,kclusters))

# plt.figure()
# dendrogram(Z)
# plt.show()
