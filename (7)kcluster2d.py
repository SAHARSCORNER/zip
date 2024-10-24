import numpy as np
import matplotlib.pyplot as plt

# 2D Data
data_2d = np.array([[2, 3], [3, 4], [10, 11], [11, 12], [12, 15],
                    [20, 25], [25, 30], [30, 35], [28, 30]])
k = 2

# Randomly initialize centroids
np.random.seed(0)  # For reproducibility
initial_centroids_indices = np.random.choice(data_2d.shape[0], size=k, replace=False)
centroids = data_2d[initial_centroids_indices]

prev_clusters = [None] * k
clusters = [None] * k
j = 0

while True:
    j += 1
    print(f"\nIteration {j}:")

    # Assign clusters based on closest centroid
    distances = np.linalg.norm(data_2d[:, np.newaxis] - centroids, axis=2)
    cluster_assignments = np.argmin(distances, axis=1)

    # Store the current clusters
    clusters = [data_2d[cluster_assignments == i] for i in range(k)]

    # Calculate new centroids
    new_centroids = np.array([cluster.mean(axis=0) if len(cluster) > 0 else centroids[i] for i, cluster in enumerate(clusters)])

    print("Clusters:")
    for i, cluster in enumerate(clusters):
        print(f"k{i + 1}:", cluster)

    print("Centroids:", new_centroids)

    # Check for convergence
    if np.array_equal(centroids, new_centroids):
        break

    centroids = new_centroids

# Final clusters output
print("\nFinal clusters are:")
for i, cluster in enumerate(clusters):
    print(f"k{i + 1}:", cluster)

# Plotting the clusters
for i, cluster in enumerate(clusters):
    plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i + 1}')

plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=200, label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid()
plt.show()
