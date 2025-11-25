# Task 8: K-Means Clustering – Synthetic Dataset (No external file)
# Tools: Scikit-learn, NumPy, Pandas, Matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# -----------------------------------------------------
# 1. Create Synthetic Dataset
# -----------------------------------------------------
print("Creating synthetic dataset...")

X, y_true = make_blobs(
    n_samples=300,       # number of data points
    centers=4,           # true number of blobs
    cluster_std=1.2,     # spread
    random_state=42
)

# Convert to DataFrame for convenience
df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
print("Dataset created successfully!")
print(df.head())

# -----------------------------------------------------
# 2. Standardize the data
# -----------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# -----------------------------------------------------
# 3. Elbow Method to find optimal K
# -----------------------------------------------------
wcss = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, wcss, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# -----------------------------------------------------
# 4. Fit K-Means with K = 4 (expected optimum)
# -----------------------------------------------------
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

df['Cluster'] = cluster_labels
print("\nCluster labels added to dataset:")
print(df.head())

# -----------------------------------------------------
# 5. Silhouette Score
# -----------------------------------------------------
sil_score = silhouette_score(X_scaled, cluster_labels)
print(f"\nSilhouette Score for K = {optimal_k}: {sil_score:.3f}")

# -----------------------------------------------------
# 6. PCA for 2D Visualization
# -----------------------------------------------------
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1],
            c=cluster_labels, s=70, cmap='viridis')
plt.title("K-Means Clustering Visualization (PCA Reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()
