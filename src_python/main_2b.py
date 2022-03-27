import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

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

output_lines = []

for window_size in ["17", "35", "47", "65", "79", "93"]:
    filename = f"D:/Programming/Github/EE569_HW4/build/Debug/Mosaic_b_features_{window_size}.csv"
    print("-"*50, filename, "-"*50)
    output_lines.append("-"*50 + filename + "-"*50)
    
    # Load feature vectors
    with open(filename, "r") as f:
        lines = f.readlines()
        width, height, channels = tuple(map(int, lines[0].split(",")))
        features = np.zeros((height * width, channels))
        for i, line in enumerate(lines[1:]):
            vals = tuple(map(float, line.split(",")[:-1]))
            features[i, :] = np.asarray(vals)

    # Normalize features by L5L5T
    # features /= features[:, 0].reshape(-1, 1)

    # Discard L5L5T dimension
    features = np.delete(features, obj=0, axis=1) # drop the 1st column

    for pca_comp in [3, 6, 12, 18, 24]:
        ########################## PCA ##########################
        if pca_comp == 24:
            features_pca = features
        else:
            pca = PCA(n_components=pca_comp)
            pca.fit(features)
            features_pca = pca.transform(features)

        ########################## K-Means  ##########################
        print("PCA Num Components", pca_comp)
        kmeans = KMeans(n_clusters=6).fit(features_pca)

        counts = np.bincount(kmeans.labels_)
        print("K-Means Cluster counts", sorted(counts))

        predictions = np.array(kmeans.labels_).reshape(height, width)
        segmentation_image = np.zeros((height, width, 3), dtype=np.uint8)
        for h in range(height):
            for w in range(width):
                segmentation_image[h, w, :] = cluster_to_color(predictions[h, w])

        plt.imsave(filename.replace(".csv", f"_kmeans_pca{pca_comp}.png"), segmentation_image)
        
        ########################## K-Means  ##########################
        gmm = GaussianMixture(n_components=6).fit_predict(features_pca)

        counts = np.bincount(gmm)
        print("GMM Cluster counts", sorted(counts))
        print()

        predictions = np.array(gmm).reshape(height, width)
        segmentation_image = np.zeros((height, width, 3), dtype=np.uint8)
        for h in range(height):
            for w in range(width):
                segmentation_image[h, w, :] = cluster_to_color(predictions[h, w])

        plt.imsave(filename.replace(".csv", f"_gmm_pca{pca_comp}.png"), segmentation_image)