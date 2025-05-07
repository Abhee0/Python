import numpy as np
import matplotlib.pyplot as plt

# Given points
points = np.array([
    [0.1, 0.6],   # P1
    [0.15, 0.71], # P2
    [0.08, 0.9],  # P3
    [0.16, 0.85], # P4
    [0.2, 0.3],   # P5
    [0.25, 0.5],  # P6
    [0.24, 0.1],  # P7
    [0.3, 0.2]    # P8
])

# Initial centroids
centroid1 = np.array([0.1, 0.6])  # m1 = P1
centroid2 = np.array([0.3, 0.2])  # m2 = P8

# Euclidean distance function
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

# Assign points to clusters
def assign_clusters(points, centroid1, centroid2):
    cluster1 = []
    cluster2 = []
    for point in points:
        d1 = euclidean_distance(point, centroid1)
        d2 = euclidean_distance(point, centroid2)
        if d1 < d2:
            cluster1.append(point)
        else:
            cluster2.append(point)
    return np.array(cluster1), np.array(cluster2)

# Perform one iteration
cluster1, cluster2 = assign_clusters(points, centroid1, centroid2)

# Calculate updated centroids
centroid1_new = np.mean(cluster1, axis=0)
centroid2_new = np.mean(cluster2, axis=0)

# Output cluster results
print("Cluster 1:")
print(cluster1)
print("Cluster 2:")
print(cluster2)

print(f"\nUpdated Centroid 1 (m1): {centroid1_new}")
print(f"Updated Centroid 2 (m2): {centroid2_new}")

# Determine which cluster P6 belongs to
p6 = np.array([0.25, 0.5])
d1_p6 = euclidean_distance(p6, centroid1_new)
d2_p6 = euclidean_distance(p6, centroid2_new)
p6_cluster = 'C1' if d1_p6 < d2_p6 else 'C2'
print(f"\nP6 belongs to: {p6_cluster}")

# Population around m2 (Cluster 2)
print(f"Population around m2 (Cluster 2): {len(cluster2)}")

# Optional: Plotting the clusters and centroids
plt.scatter(*zip(*cluster1), c='blue', label='Cluster 1 (C1)')
plt.scatter(*zip(*cluster2), c='green', label='Cluster 2 (C2)')
plt.scatter(*centroid1_new, c='blue', marker='X', s=100, label='Centroid C1 (new)')
plt.scatter(*centroid2_new, c='green', marker='X', s=100, label='Centroid C2 (new)')
plt.title("K-Means Clustering (1 Iteration)")
plt.legend()
plt.grid(True)
plt.show()
