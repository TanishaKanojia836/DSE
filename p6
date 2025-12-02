#Perform density-based clustering algorithm on a downloaded dataset and evaluate the cluster
quality by changing the algorithm's parameters.
#  DENSITY-BASED CLUSTERING (DBSCAN)

from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd

# Load dataset
iris = load_iris()
X = iris.data

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to run DBSCAN
def run_dbscan(X, eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    # DBSCAN assigns -1 to noise points
    unique_clusters = set(labels)

    # silhouette score only valid when >=2 clusters
    if len(unique_clusters) > 1 and -1 not in unique_clusters:
        sil = silhouette_score(X, labels)
    else:
        sil = "Not Applicable"

    print(f"\nParameters: eps={eps}, min_samples={min_samples}")
    print("Clusters formed:", unique_clusters)
    print("No. of clusters:", len(unique_clusters) - (1 if -1 in unique_clusters else 0))
    print("Silhouette Score:", sil)

# Parameter variations

eps_values = [0.3, 0.5, 0.7]
min_samples_values = [3, 5, 10]

for eps in eps_values:
    for ms in min_samples_values:
        run_dbscan(X_scaled, eps, ms)
