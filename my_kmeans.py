import numpy as np
import matplotlib.pyplot as plt

class KMeansClustering:
    
    def __init__(self, k=3):
        self.k = k
        self.centroids = None
        
    @staticmethod
    def euclidean_distance(data_point, centroids):
        '''vectorized approach to euclidean distance -> works on two vectors
        return: one dimentional np array of len = centroids num
        '''
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis = 1)) #both dimensions column wise
    
    def fit(self, X, max_iterations=200):
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0),
                                           size=(self.k, X.shape[1]))
        
        for _ in range(max_iterations):
            y = np.zeros(X.shape[0])
            
            for i, data_point in enumerate(X):
                distances = KMeansClustering.euclidean_distance(data_point, self.centroids)
                cluster_index = np.argmin(distances)
                # remember each distance is indexed by cluster number, we get index of min dist
                y[i] = cluster_index
            
            # indices of all point belonging to each cluster
            cluster_indices = []
            
            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i)) # group all points to their cluster
            
            
            cluster_centers = []
    
            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0: # the "i"th cluster has no points
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0]) 
                    # [0] just to take a 1-d array instead of 2d
                    # print(cluster_centers) 
                    # index 0 -> point1 [x1, y1]
                    # index 1 -> point2 [x2, y2]
                    # mean = [(x1 + x2) / 2, (y1 + y2) / 2] -> effect of axis 0
            
            cluster_centers = np.array(cluster_centers)        
            
            if np.max(self.centroids - cluster_centers) < 0.0001:
                break
            else:
                self.centroids = cluster_centers
                
        return y
                    
random_points = np.random.randint(0, 100, (100, 2))                    

kmeans = KMeansClustering(3)
labels = kmeans.fit(random_points)

plt.scatter(random_points[:, 0], random_points[:, 1], c=labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c=range(len(kmeans.centroids)),
            marker="*", s=200)
plt.show()