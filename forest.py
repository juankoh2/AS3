from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans

from sklearn import datasets
wine = datasets.load_wine()
X = wine.data
# for temp in wine.data:
# print(wine.DESCR)
# print(wine.data[0])

Z = linkage(X,'complete')
print(Z)
kclusters = fcluster(Z, 3, criterion='maxclust')
print(kclusters)

km = KMeans(n_clusters=3)
km.fit(X)
print(km.labels_)

plt.figure()
dendrogram(Z)
plt.show()

