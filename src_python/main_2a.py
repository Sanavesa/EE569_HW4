from msilib.schema import File
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt

# Specify the color for each cluster
COLORS = {
    0: np.array([107, 143, 159], dtype=int),
    1: np.array([114, 99, 107], dtype=int),
    2: np.array([175, 128, 74], dtype=int),
    3: np.array([167, 57, 32], dtype=int),
    4: np.array([144, 147, 104], dtype=int),
    5: np.array([157, 189, 204], dtype=int),
}

def cluster_to_color(cluster):
    return COLORS[cluster]

for window_size in ["17", "35", "47", "65", "79", "93"]:
    filename = f"D:/Programming/Github/EE569_HW4/build/Debug/Mosaic_a_features_{window_size}.csv"
    print("-"*50, filename, "-"*50)

    # Load feature vectors
    with open(filename, "r") as f:
        lines = f.readlines()
        width, height, channels = tuple(map(int, lines[0].split(",")))
        features = np.zeros((height * width, channels))
        for i, line in enumerate(lines[1:]):
            vals = tuple(map(float, line.split(",")[:-1]))
            features[i, :] = np.asarray(vals)

    # Normalize features by L5L5T
    features /= features[:, 0].reshape(-1, 1)

    # Discard L5L5T dimension
    features = np.delete(features, obj=0, axis=1) # drop the 1st column

    ########################## K-Means on 24D ##########################
    kmeans = KMeans(n_clusters=6).fit(features)

    counts = np.bincount(kmeans.labels_)
    print("K-Means Cluster counts", sorted(counts))
    print()

    predictions = np.array(kmeans.labels_).reshape(height, width)
    segmentation_image = np.zeros((height, width, 3), dtype=np.uint8)
    for h in range(height):
        for w in range(width):
            segmentation_image[h, w, :] = cluster_to_color(predictions[h, w])

    plt.imsave(filename.replace(".csv", "_kmeans.png"), segmentation_image)
    plt.show()