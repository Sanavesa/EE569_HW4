import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import DistanceMetric

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

# Plot how well PCA compressed the dataset
def calculate_pca_reconstruction_accuracy(covariance_matrix, label=""):
    pca_reconstruction_acc = 0
    for idx, eigenvector in enumerate(pca.components_):
        eigenvalue = np.dot(eigenvector.T, np.dot(covariance_matrix, eigenvector))
        print(f"{label}'s eigenvalue #{idx+1}", eigenvalue)
        pca_reconstruction_acc += eigenvalue
    trace = np.trace(covariance_matrix)
    print(f"{label}'s trace", trace)
    pca_reconstruction_acc /= trace
    print(f"{label}'s pca_reconstruction_acc", pca_reconstruction_acc)
calculate_pca_reconstruction_accuracy(np.cov(train_x.T), "train")
calculate_pca_reconstruction_accuracy(np.cov(test_x.T), "test")

# Train PCA Plot
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(train_x_pca[:9, 0], train_x_pca[:9, 1], train_x_pca[:9, 2], c=train_colors[:9], marker="x", label="Blanket")
ax.scatter(train_x_pca[9:18, 0], train_x_pca[9:18, 1], train_x_pca[9:18, 2], c=train_colors[9:18], marker="o", label="Brick")
ax.scatter(train_x_pca[18:27, 0], train_x_pca[18:27, 1], train_x_pca[18:27, 2], c=train_colors[18:27], marker="*", label="Grass")
ax.scatter(train_x_pca[27:, 0], train_x_pca[27:, 1], train_x_pca[27:, 2], c=train_colors[27:], marker="s", label="Stones")
# for i, label in enumerate(train_labels):
#     ax.text(train_x_pca[i, 0], train_x_pca[i, 1], train_x_pca[i, 2], label, size=8, zorder=1, color="k") 
ax.set_title("PCA on Train Dataset")
ax.set_xlabel("$1^{st}$ PC")
ax.set_ylabel("$2^{nd}$ PC")
ax.set_zlabel("$3^{rd}$ PC")
ax.legend()
plt.show()

# Test PCA Plot
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(test_x_pca[:, 0], test_x_pca[:, 1], test_x_pca[:, 2], marker="x")
for i, label in enumerate(test_labels):
    ax.text(test_x_pca[i, 0], test_x_pca[i, 1], test_x_pca[i, 2], label, size=8, zorder=1, color="k") 
ax.set_title("PCA on Test Dataset")
ax.set_xlabel("$1^{st}$ PC")
ax.set_ylabel("$2^{nd}$ PC")
ax.set_zlabel("$3^{rd}$ PC")
plt.show()

# Test Classification using Mahalanobis distance
mahalanobis = DistanceMetric.get_metric("mahalanobis", V=np.cov(train_x_pca.T))
mahalanobis_matrix = mahalanobis.pairwise(test_x_pca, train_x_pca)
indices = np.argmin(mahalanobis_matrix, axis=1)
classification = [train_y[int(i)] for i in indices]
correct_classification = ["grass", "blanket", "blanket", "stones", "stones", "grass", "brick", "stones", "brick", "brick", "blanket", "grass"]

for idx, (output, correct_output) in enumerate(zip(classification, correct_classification)):
    if output == correct_output:
        print(f"{idx+1}. {output}")
    else:
        print(f"{idx+1}. {output}\tWrong: {correct_output}")

matching_classifications = len([i for i, j in zip(classification, correct_classification) if i == j])
print(f"Accuracy: {matching_classifications}/{len(classification)}")

"""
1. blanket	Wrong: grass
2. blanket
3. blanket
4. stones
5. stones
6. grass
7. brick
8. brick	Wrong: stones
9. brick
10. brick
11. stones	Wrong: blanket
12. grass
Accuracy: 9/12
"""