import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

label_lookup = {"blanket": 0, "brick": 1, "grass": 2, "stones": 3}
color_lookup = {0: "blue", 1: "red", 2: "green", 3: "black"}

# Load Train Feature Vectors
train_x = np.zeros((36, 25))
train_y = [""] * 36
train_colors = ["black"] * 36
train_labels = ["NONE"] * 36
with open("build/debug/train_features.csv", "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        vals = line.split(",")[:-1]
        train_x[i, :] = vals[1:]
        train_y[i] = vals[0]
        train_colors[i] = color_lookup[label_lookup[vals[0]]]
        train_labels[i] = vals[0] + "_" + str(1 + (i % 9))

# Load Test Feature Vectors    
test_x = np.zeros((12, 25))
test_labels = ["NONE"] * 12
with open("build/debug/test_features.csv", "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        vals = line.split(",")[:-1]
        test_x[i, :] = vals[1:]
        test_labels[i] = str(i+1)

# PCA
pca = PCA(n_components=3)
pca.fit(train_x)
train_x_pca = pca.transform(train_x)
test_x_pca = pca.transform(test_x)

# Train PCA Plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(train_x_pca[:, 0], train_x_pca[:, 1], train_x_pca[:, 2], c=train_colors, marker='x')
for i, label in enumerate(train_labels):
    ax.text(train_x_pca[i, 0], train_x_pca[i, 1], train_x_pca[i, 2], label, size=8, zorder=1, color='k') 
plt.title("PCA on Train Dataset")
plt.show()

# Test PCA Plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(test_x_pca[:, 0], test_x_pca[:, 1], test_x_pca[:, 2], marker='x')
for i, label in enumerate(test_labels):
    ax.text(test_x_pca[i, 0], test_x_pca[i, 1], test_x_pca[i, 2], label, size=8, zorder=1, color='k') 
plt.title("PCA on Test Dataset")
plt.show()

# Test Classification using Mahalanobis distance
nn = NearestNeighbors(n_neighbors=1, algorithm="brute", metric="mahalanobis", metric_params={"VI": np.cov(train_x_pca)})
neighbors = nn.fit(train_x_pca)
neighbors, indices = neighbors.kneighbors(test_x_pca)
classification = [train_y[int(i)] for i in indices]
print("classification", classification)