from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans

from sklearn import datasets
wine = datasets.load_wine()
X = wine.data

Z = linkage(X,'average')

plt.figure(figsize=(25, 10))
dendrogram(Z)
plt.show()

kclusters = fcluster(Z, 3, criterion='maxclust')
kclusters

km = KMeans(n_clusters=3)
km.fit(X)
km.labels_