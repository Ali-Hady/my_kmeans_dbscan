import numpy as np

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

class DBScan:
    
    def __init__(self, eps, min_pts, metric=euclidean_distance):
        self.eps = eps
        self.min_pts = min_pts
        self.metric = metric
        self._labels = None
        
    @staticmethod
    def __neighbors_gen(X, point, eps, metric):
        neighbors = []
        
        for i in range(X.shape[0]):
            if metric(X[point], X[i]) < eps:
                neighbors.append(i)
                
        return neighbors
    
    @staticmethod
    def __expand(X, labels, point, neighbors, current, eps, min_pts, metric):
        labels[point] = current
        
        for i in range(len(neighbors)):
            next_point = neighbors[i]
            
            if labels[next_point] == -1: # not enough neighbors already -> a border point
                labels[next_point] = current
                
            elif labels[next_point] == 0: # possibility for enough neighbors -> next core point | border point
                labels[next_point] = current
                next_neighbors = DBScan.__neighbors_gen(X, next_point, eps, metric)
                
                if len(next_neighbors) >= min_pts:
                    neighbors += next_neighbors # update neighbors with neighbors of neighbors
                    
    @staticmethod
    def __driver(X, labels, eps, min_pts, metric=euclidean_distance):
        current = 0
        
        for i in range(X.shape[0]):
            if labels[i] != 0:
                continue
            
            neighbors = DBScan.__neighbors_gen(X, i, eps, metric)
            if len(neighbors) < min_pts:
                labels[i] = -1
            else:
                current += 1 # a new cluster is formed
                DBScan.__expand(X, labels, i, neighbors, current, eps, min_pts, metric)
                
        return labels
    
    def fit_predict(self, X):
        labels = np.zeros(X.shape[0])
        self._labels = DBScan.__driver(X, labels, self.eps, self.min_pts, self.metric)
        
        return self._labels
    
    

from sklearn.datasets import load_iris
iris = load_iris()

# Scale the data
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(iris.data)

eps = 0.5
minPts = 5
labels = DBScan(eps, minPts).fit_predict(X)

# Visualize the clusters
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X[:, 0], X[:, 1], c=labels)
ax.set_xlabel('Sepal length')
ax.set_ylabel('Sepal width')
plt.show()
                
                
                
        