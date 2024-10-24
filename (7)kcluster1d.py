import numpy as np

# 1D Data
data_1d = [2, 3, 4, 10, 11, 12, 20, 25, 30]
D_1d = np.array(data_1d)

# Number of clusters
k = 2
# Randomly choose initial centroids
c1 = np.random.choice(D_1d)
c2 = np.random.choice(D_1d)

prev_k1 = [0]
prev_k2 = []
k1 = []
k2 = []
j = 0

while prev_k1 != k1:
    prev_k1 = k1.copy()  # Use copy to avoid reference issues
    k1 = []
    k2 = []
    j += 1
    print(f"\nIteration {j}:")
    for i in D_1d:
        if abs(i - c1) < abs(i - c2):
            k1.append(i)
        else:
            k2.append(i)
    c1 = round(sum(k1) / len(k1)) if k1 else c1  # Update centroid if k1 is not empty
    c2 = round(sum(k2) / len(k2)) if k2 else c2  # Update centroid if k2 is not empty

    print("k1:", k1)
    print("k2:", k2)
    print("c1:", c1, ", c2:", c2)

print("\nFinal clusters are:")
print("k1:", k1)
print("k2:", k2)
